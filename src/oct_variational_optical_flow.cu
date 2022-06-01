#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ctime>
#include "image.h"
#include "goesread.h"
#include "offlags.h"

//header files from the CUDA samples
#include <cooperative_groups.h>
#include <helper_functions.h>
#include <helper_cuda.h>
using namespace std;
namespace cg = cooperative_groups;

//Function: oct_variational_optical_flow.cu
//Purpose: This is CUDA-based code designed to perform variational optical flow computation given two image objects
//
//Author: Jason Apke

//const int threadsPerBlock=512; //Set this to determine how many threads you will use in each gpu block
#define threadsPerBlock 128
//#define threadsPerBlock 256
//#define threadsPerBlock 512
__device__
float oct_bc_cu(float x, int nx, bool &bc)
{
    bc=false;
    if(x < 0)
    {
        x = 0;
        bc=true;
    }
    if(x >= nx)
    {
        x = nx-1;
        bc=true;
    }
    return x;
}
//A square function designed to reduce mem/block (pow is expensive)
__device__
float jsq(float x)
{
    return x*x;
}

__device__
void zoom_size(int nx, int ny, int &nxx, int &nyy, double factor)
{
    nxx = (int)((double)nx* factor + 0.5);
    nyy = (int)((double)ny* factor + 0.5);
}

__device__
float oct_binterp_coefs_cu (float x, float y,float x1, float x2, float y1, float y2,float f11,float f21,float f12,float f22,float & p1, float & p2,float & p3,float & p4)
{
    //All about efficiency, only compute bilinear terms once
    p1 = (x2-x)/(x2-x1);
    p2 = (x-x1)/(x2-x1);
    p3 = ((y2-y)/(y2-y1));
    p4 = ((y-y1)/(y2-y1));
    return p3*((p1)*f11+(p2)*f21)+p4*((p1)*f12+(p2)*f22);
}
__device__
float oct_coef_binterp_cu(float p1, float p2, float p3, float p4, float f11,float f21,float f12,float f22)
{
    //uses the bilinear terms computed from the function above
    return p3*((p1)*f11+(p2)*f21)+p4*((p1)*f12+(p2)*f22);
}
//Robust function derivative for smoothness constraint
__device__
float oct_PSI_smooth_cu(float x,int doq)
{
    //doq means do quadratic (for graduated non-convexity minimization)
    float answer;
    if(doq == 0)
    {
        answer= 1./(sqrtf((x+1E-6)));
    } else
    {
        answer=1.; 
    }
    return answer;
}
//experimental weighting function
__device__
float scw(float a, float b, float sigma)
{
    //an experimental smoothness constraint weighting
    float amb = a-b;
    return exp(-1.*(amb*amb)/sigma);
}
//Robust function derivative for data constraint
__device__
float oct_PSI_data_cu(float x,int doq)
{
    float answer;
    if(doq == 0)
    {
        answer=1./(sqrt(x+1E-6));
    } else
    {
        answer=1.;
    }
    return answer;
}

//A function for matrix multiplication
__device__
float multiply_row( int rowsize,int rowstart, float *Aval, int *Acol, float *x0,int nx, long An)
{
    float sum = 0;
    for(int i = rowstart; i < rowstart+rowsize; i++)
    {
        sum += Aval[i]*x0[Acol[i]];
    }
    return sum;

}
//A function for multiplying matrices by vectors
__device__
void jMatXVec(  float *Aval,int *Arow, int *ArowSP, int *Acol, float *x0, long An,int nx, float *ans, const cg::grid_group &grid)
{
    int row_begin, row_end;
    for (int k = grid.thread_rank(); k < nx; k+=grid.size())
    {
        row_begin = ArowSP[k];
        if(k < nx-1)
        {
            row_end = ArowSP[k+1];
        } else
        {
            row_end = An;
        }
        ans[k] = multiply_row(row_end-row_begin,row_begin,Aval,Acol,x0,nx,An);
    }
}
//A quick inverter kernel for the M matrix
__device__
void jDiagInv(float *M, int nx, const cg::grid_group &grid)
{
    for (int k = grid.thread_rank(); k < nx; k+=grid.size())
    {
        M[k] = 1./M[k];
    }

}
//A dot product within GPUs using shared memory and atomic add functions for speed
__device__ void jVecXVec(float *vecA, float *vecB, float *result, int size, const cg::thread_block &cta, const cg::grid_group &grid)
{
   __shared__ float tmp[threadsPerBlock];

    float temp_sum = 0.0;
    for (int i=grid.thread_rank(); i < size; i+=grid.size())
    {
        temp_sum += (float) (vecA[i] * vecB[i]);
    }
    tmp[cta.thread_rank()] = temp_sum;

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    float beta  = temp_sum;
    float temp;

    for (int i = tile32.size() / 2; i > 0; i >>= 1) {
        if (tile32.thread_rank() < i) {
            temp       = tmp[cta.thread_rank() + i];
            beta       += temp;
            tmp[cta.thread_rank()] = beta;
        }
        cg::sync(tile32);
    }
    cg::sync(cta);

    if (cta.thread_rank() == 0) {
        beta  = 0.0;
        for (int i = 0; i < cta.size(); i += tile32.size()) {
            beta  += tmp[i];
        }
        atomicAdd(result, beta);
    }
}
//A scalar times a vector function
__device__
void jScaleXVec(  float a, float *vec1,float *vec2, int nx, const cg::grid_group &grid)
{

    for (int k = grid.thread_rank(); k < nx; k+=grid.size())
    {
        vec2[k] = a*vec1[k];  
    }
}
//A vector addition function with a bonus d*a scalar multiplication (speeds things up here)
__device__
void jVecPVec(  float *a, float *b,float *c, float d,int nx, const cg::grid_group &grid)
{
    for (int k = grid.thread_rank(); k < nx; k+=grid.size())
    {
        c[k] = d*a[k]+b[k];  
    }
}
//Kernel that fills a gaussian kernel for zooming
__device__ 
void fill_GK(float *GK, float factor,int filtsize, int filtsize2, const cg::grid_group &grid)
{
    float sigma,r,s;
    if(grid.thread_rank() == 0)
    {
        sigma = 0.6*sqrt(1.0/(factor*factor)-1.0); 
        s = 2.0 * sigma * sigma;

        float sum = 0.0;
        for (int x = -filtsize; x <= filtsize; x++) {
            r = x ;
            GK[x + filtsize] = (exp(-(r * r) / s)) / (3.14159265358979323846 * s);
            sum += GK[x + filtsize];

        }

        for (int i = 0; i < filtsize2; ++i)
            GK[i] /= sum;

    }
}
//bicubic interpolation sub-function 3
__device__
float oct_cell_cu(
    float v[4],  //interpolation points
    float x      //point to be interpolated
)
{
    return  v[1] + 0.5 * x * (v[2] - v[0] +
            x * (2.0 *  v[0] - 5.0 * v[1] + 4.0 * v[2] - v[3] +
            x * (3.0 * (v[1] - v[2]) + v[3] - v[0])));
}
//bicubic interpolation sub-function 2
__device__
float oct_bicubic_cell_cu (
    float p[4][4], //array containing the interpolation points
    float x,       //x position to be interpolated
    float y        //y position to be interpolated
)
{
    float v[4];
    v[0] = oct_cell_cu(p[0], y);
    v[1] = oct_cell_cu(p[1], y);
    v[2] = oct_cell_cu(p[2], y);
    v[3] = oct_cell_cu(p[3], y);

    return oct_cell_cu(v, x);
}
//bicubic interpolation sub-function 1
__device__
float oct_bicubic_cu(float * input, float uu, float vv, int nx, int ny)
{
    int sx = 1;
    int sy = 1;
    int x, y, mx, my, dx,dy,ddx,ddy;
    bool bc;

    //I always use reflecting boundary conditions, so this should work
    x =   (int) oct_bc_cu((float)((int) uu),nx,bc);
    y =   (int) oct_bc_cu((float)((int) vv),ny,bc);
    mx =  (int) oct_bc_cu((float)((int) (uu-sx)),nx,bc);
    my =  (int) oct_bc_cu((float)((int) (vv-sy)),ny,bc);
    dx =  (int) oct_bc_cu((float)((int) (uu+sx)),nx,bc);
    dy =  (int) oct_bc_cu((float)((int) (vv+sy)),ny,bc);
    ddx = (int) oct_bc_cu((float)((int) (uu+2*sx)),nx,bc);
    ddy = (int) oct_bc_cu((float)((int) (vv+2*sy)),ny,bc);
    int nxtmy = nx*my;
    int nxty = nx*y;
    int nxtdy = nx*dy;
    int nxtddy = nx*ddy;
    //Below may be less than ideal for GPU memory access issues, may cause slowdowns
    const float p11 = input[mx  + nxtmy];
    const float p12 = input[x   + nxtmy];
    const float p13 = input[dx  + nxtmy];
    const float p14 = input[ddx + nxtmy];

    const float p21 = input[mx  + nxty];
    const float p22 = input[x   + nxty];
    const float p23 = input[dx  + nxty];
    const float p24 = input[ddx + nxty];

    const float p31 = input[mx  + nxtdy];
    const float p32 = input[x   + nxtdy];
    const float p33 = input[dx  + nxtdy];
    const float p34 = input[ddx + nxtdy];

    const float p41 = input[mx  + nxtddy];
    const float p42 = input[x   + nxtddy];
    const float p43 = input[dx  + nxtddy];
    const float p44 = input[ddx + nxtddy];

    float pol[4][4] = {
        {p11, p21, p31, p41},
        {p12, p22, p32, p42},
        {p13, p23, p33, p43},
        {p14, p24, p34, p44}
    };
    float f = oct_bicubic_cell_cu(pol,uu-x,vv-y);
    return f;


}
//A horizontal gaussian convolution smoothing filter
__device__
void convh(float *image, float *imageout,float *GK, int nx, int ny,int nc, float factor,int filtsize, const cg::grid_group &grid)
{
    bool bc;
    long xityi = nx*ny;
    for (int lxyza= grid.thread_rank(); lxyza < xityi*nc; lxyza += grid.size())
    {
        int c = lxyza/xityi;
        int tv = lxyza-c*xityi;
        int i = (tv) % nx;
        float wsum = 0;
        for(int kk2 = -filtsize; kk2 < filtsize; ++kk2)
        {
            int iiv = (int) oct_bc_cu((float)i+kk2,nx,bc);
            wsum = wsum+GK[kk2+filtsize]*image[lxyza+(iiv-i)]; 
        }
        imageout[lxyza] = wsum;
    }
}
//A vertical gaussian convolution smoothing filter
__device__
void convv(float *imageout,float *Is,float *GK, int nx, int ny,int nc, float factor,int filtsize, const cg::grid_group &grid)
{
    bool bc;
    long xityi = nx*ny;
    //Vertical convolution
    for (int lxyza= grid.thread_rank(); lxyza < xityi*nc; lxyza += grid.size())
    {
        int c = lxyza/xityi;
        int tv = lxyza-c*xityi;
        int i = (tv) % nx;
        int j = (tv-i)/nx; 
        float wsum = 0;
        for(int kk2 = -filtsize; kk2 < filtsize; ++kk2)
        {
            int jjv = (int) oct_bc_cu((float)j+kk2,ny,bc);
            wsum = wsum+GK[kk2+filtsize]*imageout[lxyza+nx*(jjv-j)]; 
        }
        Is[lxyza] = wsum;
    }
}
//A zoom out function, where the input image Is is blurred with the gaussian filters above
__device__
void zoom_out (float *imageout,float *Is, int nx, int ny,int nc, float factor, const cg::grid_group &grid)
{
    //now interpolate

    int nxx, nyy;
    long nxxtnyy;
    nxx = (int)((double)nx* factor + 0.5);
    nyy = (int)((double)ny* factor + 0.5);
    nxxtnyy = nxx*nyy;
    for (int lxyza= grid.thread_rank(); lxyza < nxxtnyy*nc; lxyza += grid.size())
    {
        int c = lxyza/nxxtnyy;
        int tv = lxyza-c*nxxtnyy;
        int ii = (tv) % nxx;
        int jj = (tv-ii)/nxx; 
        int i2 = ii/factor;
        int j2 = jj/factor;
        //int ii = lxyza % nxx;
        //lxyz = ii + xi * jj

        //int jj = (lxyza-ii)/nxx;

        //float iv = oct_bc_cu((float)(ii+up0p0),xi,bc);
        /////Below is the BILINEAR version, I am keeping in case speed/GPU computation is an issue//////////
        //bool bc;
        //float iv = oct_bc_cu(i2,nx,bc);
        //float jv = oct_bc_cu(j2,ny,bc); 
        //int jv1 = int(jv);
        //int iv1 = int(iv);
        //if(jv1 == ny-1)
        //    jv1 = ny-2;
        //if(iv1 == nx-1)
        //    iv1 = nx-2;
        ////Important jason this used to be bicubic interpolation
        //float p1,p2,p3,p4;
        //float xitjv1 = nx*jv1;
        //long c1 = iv1+xitjv1; //xityitc[c]; //iv1,jv1,c
        //long c2 = c1+1; //iv1+1, jv1, c
        //long c3 = c1+nx; //iv1,jv1+1,c
        //long c4 = c3+1;
        ////g  = oct_binterp_coefs_cu(iv,jv,iv1,iv1+1,jv1,jv1+1,Is[c1],Is[c2],Is[c3],Is[c4],p1,p2,p3,p4);
        //float fv1,fv2;
        //p1 = ((iv1+1)-iv);  //This is bilinear interpolation but the denominator is 0 ///((iv1+1)-(iv1));
        //p2 = (iv-(iv1)); ///(x2-x1);
        //fv1 = (p1)*Is[c1]+(p2)*Is[c2];
        //fv2 = (p1)*Is[c3]+(p2)*Is[c4];
        //p3 = (jv1+1-jv); ///(y2-y1));
        //p4 = (jv-jv1); ///(y2-y1));
        ////ans = ((y2-y)/(y2-y1))*fv1+((y-y1)/(y2-y1))*fv2;
        //imageout[lxyza]=  p3*fv1+p4*fv2;
        /////////////////////////////////////////////////////////////////////////////////end bilinear option
        //Interpolation is now done with bicubic since it is sufficiently fast on the RTX 6000s/GTX 1080
        imageout[lxyza] =  oct_bicubic_cu(Is,i2,j2, nx, ny);
    }
}
//A function to compute the gradients of input image geo1
__device__
void oct_compgrad_cu (float *geo1, float *gradxarr, float *gradyarr,int xi,int yi,int nc, const cg::grid_group &grid)
{
    long lxyz_pdx,lxyz_mdx,lxyz_pdy,lxyz_mdy,lxyz_pdx2,lxyz_mdx2,lxyz_mdy2,lxyz_pdy2;
    bool bc;
    long xityi = xi*yi;
    long xityitnc = xityi*nc;

    for (int lxyza= grid.thread_rank(); lxyza < xityitnc; lxyza += grid.size())
    {
        int c = lxyza/xityi;
        int tv = lxyza-c*xityi;
        int i = (tv) % xi;
        int j = (tv-i)/xi; //this should work
        long xitj = xi*j;
        long xityitc = xityi*c;
        int jp1 = (int) oct_bc_cu((float)j+1,yi,bc);
        int jp2 = (int) oct_bc_cu((float)j+2,yi,bc);
        int jm1 = (int) oct_bc_cu((float)j-1,yi,bc);
        int jm2 = (int) oct_bc_cu((float)j-2,yi,bc);

        int ip1 = (int) oct_bc_cu((float)i+1,xi,bc);
        int ip2 = (int) oct_bc_cu((float)i+2,xi,bc);
        int im1 = (int) oct_bc_cu((float)i-1,xi,bc);
        int im2 = (int) oct_bc_cu((float)i-2,xi,bc);
        lxyz_pdx = (ip1)+xitj+xityitc;
        lxyz_pdx2 = (ip2)+xitj+xityitc;
        lxyz_mdx = (im1)+xitj+xityitc;
        lxyz_mdx2 = (im2)+xitj+xityitc;

        lxyz_pdy = i+xi*jp1+xityitc;
        lxyz_pdy2 = i+xi*jp2+xityitc;
        lxyz_mdy = i+xi*jm1+xityitc;
        lxyz_mdy2 = i+xi*jm2+xityitc;

        gradxarr[lxyza] = (-geo1[lxyz_pdx2]+8.*geo1[lxyz_pdx]-8.*geo1[lxyz_mdx]+geo1[lxyz_mdx2])/12.0; //(dist);

        gradyarr[lxyza] = (-geo1[lxyz_pdy2]+8.*geo1[lxyz_pdy]-8.*geo1[lxyz_mdy]+geo1[lxyz_mdy2])/12.0; //(dist);
    }
}
//A function to zoom in flow estimates to the next pyramid levels
//Careful, this zoom in function is SPECIFICALLY for flow, that is, it will multiply the result by input sf
__device__
void zoom_in(float * flow, float * flowout,int nx,int ny,int nxx,int nyy,float sf, const cg::grid_group &grid)
{
    const float factorx = ((float)nxx / nx);
    const float factory = ((float)nyy / ny);

    for (int lxyza= grid.thread_rank(); lxyza < nxx*nyy; lxyza += grid.size())
    {
        int ii = lxyza % nxx;
        int jj = (lxyza-ii)/nxx;
        float i2 =  (float) ((ii / factorx)-(0.5-0.5/factorx));
        float j2 =  (float) ((jj / factory)-(0.5-0.5/factory));
        flowout[lxyza] = oct_bicubic_cu(flow, i2, j2, nx, ny)/sf;
    }
}
//The function to hold the full conjugate gradient scheme together, with LOTS of inputs (arrays needed for minimization)
extern "C" __global__ void octConjugateGradient(  float *Aval,int *Arow, int *ArowSP, int *Acol, float *x0,
        float *uval, float *vval, float *uvalt, float *vvalt,float *uhval,float *vhval,
        float *geo1,float *geo2,float *geo10,float *geo20,float *CTHc,float *CTH, float *GK,float *gradxarrg2,
        float *gradyarrg2,float *gradxxarr,float *gradxyarr,float *gradyyarr,
        float *gradxarr,float *gradyarr,int nchan,double alpha,double lambdadalpha,float lambdaco,
        long An,int nx,int nx2, int ny2,
        float *bcu,float *Mval,int *Mrow,float *z0,
        float *p0, float *residc,float tol,int iters,float *rk, float *zktrk,float *z0tr0,
        float *rkTzk,float *pkTApk,float *dummyvec,
        int liters,int kiters,int filtsize,int filtsize2, float scaleFactor,float scsig,bool dozim,bool dodiscrete){
    float Bk,alphak,factor;
    float a5,a6,a7,a8;
    float psis1,psis2,psis3,psis4,psistot,psistotq,psisnmiu,psisnmiv,psisnmiuq,psisnmivq;
    float wm1p0,wp0m1,wp0p1,wp1p0;
    int xi2,xio,yio,xi,yi;
    long xityi,xityitnc; 
    cg::thread_block cta = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    //The outer iteration loop
    for(int k = 0; k < kiters; k++){
        factor = pow(scaleFactor,kiters-k-1);
        zoom_size(nx2,ny2,xi,yi,factor);  //determines xi and yi
        xi2 = 2*xi;
        xityi = xi*yi;
        xityitnc = xityi*nchan;
        //double lambdac = lambdaco*(pow(0.5,k));
        float lambdac = lambdaco*(pow(0.5,k));
        cg::sync(grid);
        //Zoom in the previous guess if needed, otherwise fill in the first guess flow values

        if(k > 0){
            zoom_in(uvalt,uval,xio,yio,xi,yi,scaleFactor,grid);
            cg::sync(grid);
            zoom_in(vvalt,vval,xio,yio,xi,yi,scaleFactor,grid);
            cg::sync(grid);
        }
        if(k == kiters-1)
        {
            for (int lxyza= grid.thread_rank(); lxyza < xityitnc; lxyza += grid.size())
            {
                geo1[lxyza] = geo10[lxyza];
                geo2[lxyza] = geo20[lxyza]; 
            }
            cg::sync(grid);
            for (int lxyza= grid.thread_rank(); lxyza < xityi; lxyza += grid.size())
            {
                uvalt[lxyza] = uhval[lxyza];
                vvalt[lxyza] = vhval[lxyza];
            }
            cg::sync(grid);
        } else
        {
            //Zoom out the images to the level of the pyramid
            float sigma = 1.0/sqrt(2.*factor); 
            //capping the blur filter size for speed
            filtsize = 2*sigma;
            if(filtsize < 5)
                filtsize = 5;
            filtsize2 = 2*filtsize+1;
            fill_GK(GK,factor,filtsize,filtsize2,grid);
            //Zoom out geo10
            cg::sync(grid);
            convh(geo10, geo1, GK, nx2, ny2, nchan, factor, filtsize, grid);
            cg::sync(grid);
            convv(geo1, dummyvec, GK, nx2, ny2, nchan, factor, filtsize,grid);
            cg::sync(grid);
            zoom_out (geo1,dummyvec, nx2, ny2,nchan,factor,grid);
            cg::sync(grid);

            //Zoom out geo20
            convh(geo20, geo2, GK, nx2, ny2, nchan, factor, filtsize, grid);
            cg::sync(grid);
            convv(geo2, dummyvec, GK, nx2, ny2, nchan, factor, filtsize,grid);
            cg::sync(grid);
            zoom_out (geo2,dummyvec, nx2, ny2,nchan,factor,grid);
            cg::sync(grid);
            
            //Zoom out flows for U hat/V hat value
            convh(uhval, uvalt, GK, nx2, ny2, 1, factor, filtsize, grid);
            cg::sync(grid);
            convv(uvalt, dummyvec, GK, nx2, ny2, 1, factor, filtsize,grid);
            cg::sync(grid);
            zoom_out (uvalt,dummyvec, nx2, ny2,1,factor,grid);
            cg::sync(grid);
            
            convh(vhval, vvalt, GK, nx2, ny2, 1, factor, filtsize, grid);
            cg::sync(grid);
            convv(vvalt, dummyvec, GK, nx2, ny2, 1, factor, filtsize,grid);
            cg::sync(grid);
            zoom_out (vvalt,dummyvec, nx2, ny2,1,factor,grid);
            cg::sync(grid);
            for (int lxyza= grid.thread_rank(); lxyza < xityi; lxyza += grid.size())
            {
                uvalt[lxyza] *= factor;
                vvalt[lxyza] *= factor; //scaling pixel displacements to current zoom level
            }
            cg::sync(grid);
            if(dodiscrete)
            {
                //zoom out the cloud-top heights needed for discrete regularization
                convh(CTHc, CTH, GK, nx2, ny2, 1, factor, filtsize, grid);
                cg::sync(grid);
                convv(CTH, dummyvec, GK, nx2, ny2, 1, factor, filtsize,grid);
                cg::sync(grid);
                zoom_out (CTH,dummyvec, nx2, ny2,1,factor,grid);
                cg::sync(grid);
            }
        }
        if(k == 0)
        {
            for (int lxyza= grid.thread_rank(); lxyza < xityi; lxyza += grid.size())
            {
                uval[lxyza] = uvalt[lxyza]; //set initial estimate to first guess
                vval[lxyza] = vvalt[lxyza];
            }
            cg::sync(grid);
            
        } 
        //Compute the relavant gradients
        oct_compgrad_cu(geo1,gradxarr,gradyarr,xi,yi,nchan,grid);
        cg::sync(grid);
        oct_compgrad_cu(geo2,gradxarrg2,gradyarrg2,xi,yi,nchan,grid);
        cg::sync(grid);
        oct_compgrad_cu(gradxarrg2,gradxxarr,gradxyarr,xi,yi,nchan,grid);
        cg::sync(grid);
        //if speed becomes an issue, we may remove redundant gradient solve here (gradxyarr) -J. Apke 2/18/2022
        oct_compgrad_cu(gradyarrg2,gradxyarr,gradyyarr,xi,yi,nchan,grid);
        cg::sync(grid);

        //double nd = 1.,nd2=1.;
        nx = xityi*2;
        //update An as well
        An = 12*xityi-4*xi-4*yi;

        //Graduated non-convexity iterations (always 3 steps)

        for(int gnc=0; gnc < 3; gnc++)
        {
            double al1 = 1.-0.5*gnc;
            //Inner iterations
            for(int l=0; l < liters; l++)
            {   

                for(int lxyza = grid.thread_rank(); lxyza < xityi; lxyza += grid.size())
                {
                    long lxyz = lxyza+lxyza;
                    long lxyzap1 = lxyza+1;
                    long lxyzam1 = lxyza-1;
                    long lxyzp1p0 = lxyzap1;
                    long lxyzp1p1 = lxyzap1+xi;
                    long lxyzp1m1 = lxyzap1-xi;
                    long lxyzp0p1 = lxyza+xi;
                    long lxyzp0m1 = lxyza-xi;
                    long lxyzm1p0 = lxyzam1;
                    long lxyzm1p1 = lxyzam1+xi;
                    long lxyzm1m1 = lxyzam1-xi;
                    int ii = lxyza % xi;
                    int jj = (lxyza-ii)/xi;
                    //Boundary conditions for the smoothness constraints
                    //May move below to it's own function

                    if(ii == 0)
                    {
                        lxyzm1p0 = lxyzm1p0+2;
                        lxyzm1p1 = lxyzm1p1+2;
                        lxyzm1m1 = lxyzm1m1+2;
                    }
                    if(ii == xi-1)
                    {
                        lxyzp1p0 = lxyzp1p0-2;
                        lxyzp1p1 = lxyzp1p1-2;
                        lxyzp1m1 = lxyzp1m1-2;
                    }
                    if(jj == 0)
                    {
                        lxyzp1m1 = lxyzp1m1+xi+xi;
                        lxyzp0m1 = lxyzp0m1+xi+xi;
                        lxyzm1m1 = lxyzm1m1+xi+xi;
                    }
                    if(jj == yi-1)
                    {
                        lxyzp1p1 = lxyzp1p1-xi-xi;
                        lxyzp0p1 = lxyzp0p1-xi-xi;
                        lxyzm1p1 = lxyzm1p1-xi-xi;
                    }

                    //Defining some of the nearby flow values to determine descritized smoothness constraint
                    float up1p0,up0p0,up1p1,up1m1,up0p1,up0m1,um1p1,um1p0,um1m1;
                    float vp1p0,vp0p0,vp1p1,vp1m1,vp0p1,vp0m1,vm1p1,vm1p0,vm1m1;
                    up1p0 = uval[lxyzp1p0]; 
                    up0p0 = uval[lxyza]; 
                    up1p1 = uval[lxyzp1p1]; 
                    up1m1 = uval[lxyzp1m1]; 
                    up0p1 = uval[lxyzp0p1]; 
                    up0m1 = uval[lxyzp0m1]; 
                    um1p1 = uval[lxyzm1p1]; 
                    um1p0 = uval[lxyzm1p0]; 
                    um1m1 = uval[lxyzm1m1];



                    vp1p0 = vval[lxyzp1p0]; 
                    vp0p0 = vval[lxyza]; 
                    vp1p1 = vval[lxyzp1p1]; 
                    vp1m1 = vval[lxyzp1m1]; 
                    vp0p1 = vval[lxyzp0p1]; 
                    vp0m1 = vval[lxyzp0m1]; 
                    vm1p1 = vval[lxyzm1p1]; 
                    vm1p0 = vval[lxyzm1p0]; 
                    vm1m1 = vval[lxyzm1m1]; 


                    float Uip1 = jsq(up1p0-up0p0)+jsq(0.25*((up1p1-up1m1) + (up0p1-up0m1)))+jsq(vp1p0-vp0p0)+jsq(0.25*((vp1p1-vp1m1) + (vp0p1-vp0m1)));
                    float Uim1 = jsq(up0p0-um1p0)+jsq(0.25*((um1p1-um1m1) + (up0p1-up0m1)))+jsq(vp0p0-vm1p0)+jsq(0.25*((vm1p1-vm1m1) + (vp0p1-vp0m1)));
                    float Ujp1 = jsq(up0p1-up0p0)+jsq(0.25*((up1p1-um1p1) + (up1p0-um1p0)))+jsq(vp0p1-vp0p0)+jsq(0.25*((vp1p1-vm1p1) + (vp1p0-vm1p0)));
                    float Ujm1 = jsq(up0p0-up0m1)+jsq(0.25*((up1m1-um1m1) + (up1p0-um1p0)))+jsq(vp0p0-vp0m1)+jsq(0.25*((vp1m1-vm1m1) + (vp1p0-vm1p0)));


                    //This discrete flag was me messing around with the regularizer- J. Apke 2/21/2022
                    if(dodiscrete){
                        wm1p0 = scw(CTH[lxyza],CTH[lxyzm1p0],scsig);
                        wp0m1 = scw(CTH[lxyza],CTH[lxyzp0m1],scsig);
                        wp0p1 = scw(CTH[lxyza],CTH[lxyzp0p1],scsig);
                        wp1p0 = scw(CTH[lxyza],CTH[lxyzp1p0],scsig);
                        float wsum = wm1p0+wp0m1+wp0p1+wp1p0;
                        wm1p0 = wm1p0/wsum;
                        wp0m1 = wp0m1/wsum;
                        wp0p1 = wp0p1/wsum;
                        wp1p0 = wp1p0/wsum;



                        psis1 = wm1p0*oct_PSI_smooth_cu(jsq(up0p0-um1p0)+jsq(vp0p0-vm1p0),0);
                        psis2 = wp0m1*oct_PSI_smooth_cu(jsq(up0p0-up0m1)+jsq(vp0p0-vp0m1),0);
                        psis3 = wp1p0*oct_PSI_smooth_cu(jsq(up0p0-up1p0)+jsq(vp0p0-vp1p0),0);
                        psis4 = wp0p1*oct_PSI_smooth_cu(jsq(up0p0-up0p1)+jsq(vp0p0-vp0p1),0);
                        psistot = psis1+psis2+psis3+psis4;

                        psistotq = 1.; //psis1q+psis2q+psis3q+psis4q;
                        psisnmiu = psis1*(up0p0-um1p0)+psis2*(up0p0-up0m1) + psis3*(up0p0-up1p0)+psis4*(up0p0-up0p1);
                        psisnmiv = psis1*(vp0p0-vm1p0)+psis2*(vp0p0-vp0m1) + psis3*(vp0p0-vp1p0)+psis4*(vp0p0-vp0p1);

                        psisnmiuq = wm1p0*(up0p0-um1p0) + wp0m1*(up0p0-up0m1) + wp1p0*(up0p0-up1p0) + wp0p1*(up0p0-up0p1); //psis1q*(uval[lxyz1])+psis2q*(uval[lxyz2]);
                        psisnmivq = wm1p0*(vp0p0-vm1p0) + wp0m1*(vp0p0-vp0m1) + wp1p0*(vp0p0-vp1p0) + wp0p1*(vp0p0-vp0p1); //psis1q*(vval[lxyz1])+psis2q*(vval[lxyz2]);
                    } else
                    {
                        psis1 = oct_PSI_smooth_cu(Uim1,0);
                        psis2 = oct_PSI_smooth_cu(Ujm1,0);
                        psis3 = oct_PSI_smooth_cu(Uip1,0);
                        psis4 = oct_PSI_smooth_cu(Ujp1,0);
                        psistot = psis1+psis2+psis3+psis4;
                        psistotq = 4.; 
                        psisnmiu = psis1*(um1p0)+psis2*(up0m1) + psis3*(up1p0)+psis4*(up0p1);
                        psisnmiv = psis1*(vm1p0)+psis2*(vp0m1) + psis3*(vp1p0)+psis4*(vp0p1);

                        psisnmiuq = um1p0 + up0m1 + up1p0 + up0p1; 
                        psisnmivq = vm1p0 + vp0m1 + vp1p0 + vp0p1; 
                    }

                    float vr1 = 0, vr2 = 0, vr4 = 0, vr5 = 0, vr6 = 0,intcomp =0; //Variables for filling sparse mat and
                    float vr12 = 0, vr22 = 0, vr42 = 0, vr52 = 0, vr62 = 0,intcomp2 =0; //Variables for filling sparse mat with Grad Constancy Separate RF
                    bool bc2=false,bc3=false;
                    bool bc;
                    //These are warped positions, so iv = ii+u estimate, with boundary conditions 
                    float iv = oct_bc_cu((float)(ii+up0p0),xi,bc);
                    if(bc) bc2=true;
                    float jv = oct_bc_cu((float)(jj+vp0p0),yi,bc);
                    if(bc) bc3=true;
                    int iv1 = int(iv);
                    int jv1 = int(jv);
                    if(iv1 == xi-1)
                    {
                        iv1= xi-2;
                    }
                    if(jv1 == yi-1)
                    {
                        jv1 = yi-2;
                    }
                    long xitjv1 = xi*jv1;


                    for(int c = 0; c < nchan; c++)
                    {
                        long xityitc = xityi*c;
                        long lxyz3d = ii+xi*jj+xityitc; 

                        long c1 = iv1+xitjv1+xityitc; 
                        long c2 = c1+1; 
                        long c3 = c1+xi; 
                        long c4 = c3+1;
                        float p1,p2,p3,p4;
                        //This step could be done every K iteration instead if x0 is used to hold the perturbations,
                        //just some thoughts for later - J. Apke 2/18/2022
                        float g2  = oct_binterp_coefs_cu(iv,jv,iv1,iv1+1,jv1,jv1+1,geo2[c1],geo2[c2],geo2[c3],geo2[c4],p1,p2,p3,p4); //after this, we are done w/ geo2
                        float Ix  = oct_coef_binterp_cu(p1,p2,p3,p4,gradxarrg2[c1],gradxarrg2[c2],gradxarrg2[c3],gradxarrg2[c4]);
                        float Iy  = oct_coef_binterp_cu(p1,p2,p3,p4,gradyarrg2[c1],gradyarrg2[c2],gradyarrg2[c3],gradyarrg2[c4]);
                        float Ixx = oct_coef_binterp_cu(p1,p2,p3,p4,gradxxarr[c1], gradxxarr[c2],gradxxarr[c3],gradxxarr[c4]);
                        float Ixy = oct_coef_binterp_cu(p1,p2,p3,p4,gradxyarr[c1], gradxyarr[c2],gradxyarr[c3],gradxyarr[c4]);
                        float Iyy = oct_coef_binterp_cu(p1,p2,p3,p4,gradyyarr[c1], gradyyarr[c2],gradyyarr[c3],gradyyarr[c4]);
                        //Boundary condition gradient value settings
                        if(bc2)
                        {
                            Ix=0.;
                            Ixx=0.;
                            Ixy=0.;
                        }
                        if(bc3)
                        {
                            Iy=0.;
                            Ixy=0.;
                            Iyy=0.;
                        }
                        //Here are the image derivatives needed for Brox variational optical flow

                        float It     = g2-geo1[lxyz3d]; 
                        float Ixt    = Ix-gradxarr[lxyz3d]; 
                        float Iyt    = Iy-gradyarr[lxyz3d]; 
                        //float yin    = It;
                        //float yinx   = Ixt;
                        //float yiny   = Iyt;
                        float IxIx   = Ix*Ix;
                        float IyIy   = Iy*Iy;
                        float IxxIxx = Ixx*Ixx;
                        float IxyIxy = Ixy*Ixy;
                        float IyyIyy = Iyy*Iyy;
                        float na, nb, nc;
                        //Zimmer normalization constants
                        if(dozim)
                        {
                            na = 1./(IxIx + IyIy+1.);
                            nb = 1./(IxxIxx+IxyIxy+1.);
                            nc = 1./(IxyIxy+IyyIyy+1.);
                        } else
                        {
                            na = 1.; nb = 1.; nc = 1.;
                        }
                        intcomp  += na*It*It;
                        intcomp2 += (nb*Ixt*Ixt+nc*Iyt*Iyt); 
                        //intcomp  += na*yin*yin;
                        //intcomp2 += (nb*yinx*yinx+nc*yiny*yiny); 

                        vr1  += (na*IxIx);
                        vr12 += (nb*IxxIxx+nc*IxyIxy); 

                        //float IxIy   = na*Ix*Iy;
                        //float IxxIxy = nb*Ixx*Ixy;
                        //float IxyIyy = nc*Iyy*Ixy;
                        //float LaVal  = (IxxIxy+IxyIyy); 
                        vr2  +=  na*Ix*Iy;
                        vr22 += (nb*Ixx*Ixy+nc*Iyy*Ixy);
                        //vr3  += (IxIy);
                        //vr32 += (LaVal);
                        vr4  += (na*IyIy);
                        vr42 += ((nb*IxyIxy+nc*IyyIyy)); 
                        float natIt = -na*It;
                        float nbtIxt = nb*Ixt;
                        float nctIyt = nc*Iyt;
                        vr5  += natIt*Ix;
                        vr52 += -(nbtIxt*Ixx+nctIyt*Ixy);
                        vr6  += natIt*Iy;
                        vr62 += -(nbtIxt*Ixy+nctIyt*Iyy);
                    }
                    //with the above calculated, we are ready to compute individual terms in the big sparse matrix
                    float psid = oct_PSI_data_cu(intcomp,0)/alpha; //Note for ease of computation I am combining psid and alpha here
                    //float psidq = 1./alpha; 
                    float psid2 = lambdadalpha*oct_PSI_data_cu(intcomp2,0);
                    //float psidq2 = lambdadalpha; 


                    float a1 =(float) ((al1)*((vr1)/alpha +lambdadalpha*(vr12)+lambdac + psistotq)+(1-al1)*(psid*(vr1)+psid2*vr12+lambdac + psistot));
                    float a2 = (float)((al1)*((vr2)/alpha +lambdadalpha*vr22)+(1-al1)*(psid*(vr2)+psid2*vr22));
                    //We can get rid of a3 now, its the same as a2
                    //float a3 = (float)((al1)*(psidq*(vr2)+psidq2*vr22)+(1-al1)*(psid*(vr2)+psid2*vr22));
                    //float a3 = (float)((al1)*(psidq*(vr3)+psidq2*vr32)+(1-al1)*(psid*(vr3)+psid2*vr32));
                    float a4 = (float)((al1)*((vr4)/alpha+lambdadalpha*vr42+lambdac + psistotq)+(1-al1)*(psid*(vr4)+psid2*vr42+lambdac + psistot));
                    //discrete test
                    if(dodiscrete)
                    {
                        a5 = (float) (-1*(al1*wm1p0+(1-al1)*(psis1)));
                        //Terms multiplied by du and dv at i, j-1 in the du and dv equation
                        a6 = (float) (-1*(al1*wp0m1+(1-al1)*(psis2)));

                        //Terms multipled by du and dv at i+1, j in the du and dv equation
                        a7 = (float) (-1*(al1*wp1p0+(1-al1)*(psis3)));
                        //Terms multiplied by du and dv at i, j+1 in the du and dv equation
                        a8 = (float) (-1*(al1*wp0p1+(1-al1)*(psis4))); //JASON NOTE I REMOVED THE PSISQs FROM THESE!!!!
                    } else
                    {
                        //smoothness constraint terms in the sparse matrix
                        a5 = (float) (-1*(al1+(1-al1)*(psis1)));
                        //Terms multiplied by du and dv at i, j-1 in the du and dv equation
                        a6 = (float) (-1*(al1+(1-al1)*(psis2)));

                        //Terms multipled by du and dv at i+1, j in the du and dv equation
                        a7 = (float) (-1*(al1+(1-al1)*(psis3)));
                        //Terms multiplied by du and dv at i, j+1 in the du and dv equation
                        a8 = (float) (-1*(al1+(1-al1)*(psis4))); //JASON NOTE I REMOVED THE PSISQs FROM THESE!!!!
                    }

                    //now for the "hard" part.  With the terms set, it is time to find rowt and rowu
                    long rowt = 0;
                    
                    if(jj > 0) rowt += lxyza-xi;

                    if(ii == 0)
                    {
                        rowt += lxyza-jj; 
                    } else
                    {
                        rowt += lxyza-jj-1;
                    }

                    //count a1 and a2 inserts here
                    rowt += lxyza+lxyza;

                    //counts every i < xi-1
                    rowt += lxyza-jj;
                    //counts every j less than yi-1
                    if(jj < yi-1)
                    {
                        rowt += lxyza;
                    } else
                    {
                        rowt += xityi-xi;
                    }

                    if(jj > 0) rowt += lxyza-xi;
                    if(ii == 0)
                    {
                        rowt += lxyza-jj; 
                    } else
                    {
                        rowt += lxyza-jj-1;
                    }
                    rowt += lxyza+lxyza;
                    
                    rowt += lxyza-jj;
                    

                    if(jj < yi-1)
                    {
                        rowt += lxyza;
                    } else
                    {
                        rowt += xityi-xi;
                    }
                    long rowu = lxyz;
                    long lxyz1 = lxyz-2; //This is supposed to be to the adjacent x
                    long lxyz2 = lxyz-xi2; //This is supposed to be to the adjacent y
                    long lxyz3 = lxyz+2; //This is supposed to be to the adjacent x
                    long lxyz4 = lxyz+xi2; //This is supposed to be to the adjacent y
                    if(ii == 0)
                        lxyz1 += 4;
                    if(jj == 0)
                        lxyz2 += (xi2+xi2);
                    if(ii == xi-1)
                        lxyz3 = lxyz3-4;
                    if(jj == yi-1)
                        lxyz4 = lxyz4-(xi2+xi2);
                    //OK, now time to actually fill the sparse matrices A and M
                    int rowset = 0;
                    if(jj > 0){
                        if(jj<yi-1)
                        {
                            Aval[rowt]=a6;
                        } else
                        {
                            Aval[rowt]=a6+a8;
                        }
                        Arow[rowt] = lxyz;
                        Acol[rowt] = lxyz2;
                        ArowSP[rowu]=rowt;
                        rowu++;
                        rowset = 1;
                        rowt++;
                    }

                    if(ii > 0){
                        if(ii<xi-1)
                        {
                            Aval[rowt]=a5;
                        } else
                        {
                            Aval[rowt]=a5+a7;
                        }
                        Arow[rowt] = lxyz;
                        Acol[rowt] = lxyz1;
                        if(rowset == 0)
                        {
                            ArowSP[rowu]=rowt;
                            rowu++;
                            rowset = 1;
                        }
                        rowt++;
                    }

                    Aval[rowt]=a1;
                    Arow[rowt] = lxyz;
                    Acol[rowt] = lxyz;
                    if(rowset ==0)
                    {
                        ArowSP[rowu]=rowt;
                        rowu++;
                    }
                    Mval[lxyz] = a1; //Aval[rowt]; 
                    Mrow[lxyz] = lxyz; // Arow[rowt]; //should be the same, just less global mem access
                    rowt++;
                    Aval[rowt]=a2;
                    Arow[rowt] = lxyz;
                    Acol[rowt] = lxyz+1;
                    rowt++;
                    if(ii < xi-1)
                    {
                        if(ii > 0){
                            Aval[rowt]=a7;
                        } else {
                            Aval[rowt]=a7+a5;
                        }
                        Arow[rowt] = lxyz;
                        Acol[rowt] = lxyz3;
                        rowt++;
                    }

                    if(jj < yi-1)
                    {
                        if(jj > 0)
                        {
                            Aval[rowt]=a8;
                        } else {
                            Aval[rowt]=a8+a6;
                        }
                        Arow[rowt] = lxyz;
                        Acol[rowt] = lxyz4;
                        rowt++;
                    }
                    rowset = 0;
                    if(jj > 0)
                    {
                        if(jj<yi-1)
                        {
                            Aval[rowt]=a6;
                        } else
                        {
                            Aval[rowt]=a6+a8;
                        }
                        Arow[rowt] = lxyz+1;
                        Acol[rowt] = lxyz2+1;
                        ArowSP[rowu]=rowt;
                        rowu++;
                        rowset = 1;
                        rowt++;
                    }
                    if(ii > 0)
                    {
                        if(ii<xi-1)
                        {
                            Aval[rowt]=a5;
                        } else
                        {
                            Aval[rowt]=a5+a7;
                        }
                        Arow[rowt] = lxyz+1;
                        Acol[rowt] = lxyz1+1;
                        if(rowset == 0)
                        {
                            ArowSP[rowu]=rowt;
                            rowu++;
                            rowset = 1;
                        }
                        rowt++;
                    }
                    Aval[rowt]=a2;
                    Arow[rowt] = lxyz+1;
                    Acol[rowt] = lxyz;
                    if(rowset == 0)
                    {
                        ArowSP[rowu]=rowt;
                        rowu++;
                    }
                    rowt++;
                    Aval[rowt]=a4;
                    Arow[rowt] = lxyz+1;
                    Acol[rowt] = lxyz+1;
                    Mval[lxyz+1] = a4; //Aval[rowt]; 
                    Mrow[lxyz+1] = lxyz+1; //Arow[rowt];
                    rowt++;
                    if(ii < xi-1)
                    {
                        if(ii > 0)
                        {
                            Aval[rowt]=a7;
                        } else {
                            Aval[rowt]=a7+a5;
                        }
                        Arow[rowt] = lxyz+1;
                        Acol[rowt] = lxyz3+1;
                        rowt++;
                    }
                    if(jj < yi-1)
                    {
                        if(jj > 0)
                        {
                            Aval[rowt]=a8;
                        } else {
                            Aval[rowt]=a8+a6;
                        }
                        Arow[rowt] = lxyz+1;
                        Acol[rowt] = lxyz4+1;
                        rowt++;
                    }
                    if(dodiscrete)
                    {
                        bcu[lxyz] = (float) (al1*((vr5)/alpha+lambdadalpha*vr52-psisnmiuq)+
                           (1.-al1)*(psid*(vr5)+psid2*vr52-psisnmiu));
                        bcu[lxyz+1]=(float) (al1*((vr6)/alpha+lambdadalpha*vr62-psisnmivq)+
                            (1-al1)*(psid*(vr6)+psid2*vr62-psisnmiv));

                    } else
                    {
                        float val2 = lambdac*(uval[lxyza]-uvalt[lxyza]);
                        bcu[lxyz] = (float) (al1*((vr5)/alpha+lambdadalpha*vr52-val2+psisnmiuq-psistotq*uval[lxyza])+
                           (1.-al1)*(psid*(vr5)+psid2*vr52-val2+psisnmiu-psistot*uval[lxyza]));
                        val2 = lambdac*(vval[lxyza]-vvalt[lxyza]); //This is new I guess
                        bcu[lxyz+1]=(float) (al1*((vr6)/alpha+lambdadalpha*vr62-val2+psisnmivq-psistotq*vval[lxyza])+
                            (1-al1)*(psid*(vr6)+psid2*vr62-val2+psisnmiv-psistot*vval[lxyza]));
                    }



                }
                
                
                cg::sync(grid);

                //Now that Sparse Matrices A and M are set, and vector bcu is set, time for the 
                //pre-conditioned conjugate gradient algorithm

                jMatXVec(Aval,Arow,ArowSP, Acol,x0,An,nx,dummyvec,grid);
                //Wait for the threads to finish, next answers are dependent
                cg::sync(grid);

                
                for(int i = grid.thread_rank(); i < nx; i += grid.size())
                {
                    bcu[i]= bcu[i] - dummyvec[i];
                }
                cg::sync(grid);
                jDiagInv(Mval,nx,grid);// We will change numblocks as we go
                cg::sync(grid);
                jMatXVec(Mval,Mrow,Mrow,Mrow,bcu,nx,nx,z0,grid);///////////DEFINE z0
                cg::sync(grid);
                for(int j = grid.thread_rank(); j < nx; j += grid.size())
                {
                    p0[j] = z0[j];
                }
                cg::sync(grid);
                if(threadIdx.x == 0 && blockIdx.x == 0)
                    *residc = 0.0;
                jVecXVec(bcu,bcu,residc,nx,cta,grid);
                cg::sync(grid);

                int ki = 0;

                while(((*residc) > tol) && (ki < iters)){
                    if(ki > 0){
                        if(threadIdx.x == 0 && blockIdx.x == 0)
                            *z0tr0 = 0.0;
                        jVecXVec(z0,bcu,z0tr0,nx,cta,grid); //Should be floats
                        cg::sync(grid);

                        jMatXVec(Mval,Mrow,Mrow,Mrow,rk,nx,nx,z0,grid); ///z0 updated here!!!
                        cg::sync(grid);
                        if(threadIdx.x == 0 && blockIdx.x == 0)
                            *zktrk = 0.0;
                        jVecXVec(z0,rk,zktrk,nx,cta,grid);
                        cg::sync(grid);
                        Bk = (*zktrk) / (*z0tr0);

                        jVecPVec(p0,z0,p0,Bk,nx,grid); //!!!p0 updated here!!!!
                        cg::sync(grid);

                        //Now that this is all done, reset bcu and loop again
                        for(int j = grid.thread_rank(); j < nx; j += grid.size()){
                            bcu[j] = rk[j];
                        }
                        cg::sync(grid);
                    }
                    if(threadIdx.x == 0 && blockIdx.x == 0)
                        *rkTzk = 0.0;
                    jVecXVec(bcu,z0,rkTzk,nx,cta,grid);
                    cg::sync(grid);


                    jMatXVec(Aval,Arow,ArowSP,Acol,p0,An,nx,dummyvec,grid); //replacing pkTA with Atx0 for memory
                    cg::sync(grid);
                    if(threadIdx.x == 0 && blockIdx.x == 0)
                        *pkTApk = 0.0;
                    jVecXVec(p0,dummyvec,pkTApk,nx,cta,grid);
                    cg::sync(grid);

                    
                    alphak = (*rkTzk) / (*pkTApk);
                    jMatXVec(Aval,Arow,ArowSP,Acol, p0,An,nx,dummyvec,grid);
                    cg::sync(grid);
                    jVecPVec(p0,x0,x0,alphak,nx,grid);
                    cg::sync(grid);
                    jVecPVec(dummyvec,bcu,rk,-1.*alphak,nx,grid);
                    cg::sync(grid);
                    if(threadIdx.x == 0 && blockIdx.x == 0)
                        *residc = 0.0;
                    jVecXVec(rk,rk,residc,nx,cta,grid);

                    cg::sync(grid);
                    ki++;
                }
                cg::sync(grid);
                //At this point, it is solved, so update the u/v vectors accordingly
                for(int kii = grid.thread_rank(); kii < xityi; kii += grid.size())
                {
                    int kii2 = kii+kii;
                    uval[kii] = uval[kii]+x0[kii2];
                    vval[kii] = vval[kii]+x0[kii2+1]; 
                    x0[kii2] = 0.; //x0 contains the residuals
                    bcu[kii2] = 0.;
                    x0[kii2+1] = 0.;
                    bcu[kii2+1] = 0.;

                }
                cg::sync(grid);
            }//endfor liters (inner iterations)
        }//endfor GNC (Graduated Non-Convexity iterations)
        cg::sync(grid);
        //re-use u/vvalt arrays to save on storage and pass flow values that need to be zoomed in
        for (int lxyza= grid.thread_rank(); lxyza < xityi; lxyza += grid.size())
        {
            uvalt[lxyza] = uval[lxyza];
            vvalt[lxyza] = vval[lxyza];
        }
        cg::sync(grid);
        xio = xi; //set the previous resolution values
        yio = yi;
        cg::sync(grid);
    }//endfor kiters (outer iterations)
}
//CPU wrapper
void oct_variational_optical_flow(Image geo1i,Image geo2i,float *CTH,float *uarr, float *varr,int nx, int ny,int nc, OFFlags args)
{

    //Most of these will be cuda arrays
    float *Aval,*bcu,*Mval,*x0,*z0,*p0,*rk,*dummyvec;
    float *geo1,*geo2,*geo10,*geo20,*GK,*gradxarrg2,*gradyarrg2,*gradxxarr,*gradxyarr,*gradyyarr,*gradxarr,*gradyarr;
    float *uval, *vval,*uhval,*vhval,*uvalt,*vvalt,*CTHc,*CTHo;
    float *rkTzk,*pkTApk,*zktrk,*z0tr0,*residc;

    int *Arow,*Acol,*Mrow,*ArowSP;
    int nchan = geo1i.nchannels;


    long xityi = nx*ny;
    long xityi2 = xityi*2;
    long An=12*xityi-4*nx-4*ny; 
    double alpha = args.alpha;
    double lambdadalpha = args.lambda/alpha;
    int kiters = args.kiters;
    int liters = args.liters;
    int iters  = args.cgiters; //conjugate gradient max iterations
    bool dozim = true;
    if(args.dozim == 0) dozim = false;
    float lambdac = args.lambdac/alpha;
    //double lambdac = args.lambdac/alpha;
    float scsig = (float) args.scsig;
    
    
    float scaleFactor= (float) args.scaleF;
    float factor = pow(scaleFactor,kiters-1);
    float sigma = 0.6*sqrt(1.0/(factor*factor)-1.0); 
    int filtsize = (int) 2*sigma;
    if(filtsize < 5)
        filtsize = 5; //min gauss filter size
    int filtsize2 = 2*filtsize+1;

    //Need to add a set device here for dual-GPU machines
    //Boilerplate statement to test how many gpus you have and select one to use (based on args)
    int devcount=0;
    cudaGetDeviceCount(&devcount);
    //cout << "This device has " << devcount << " gpus\n";
    int dev = args.setdevice; //default device is the first gpu, but some have multiple gpus
    if(devcount == 0)
    {
        cout << "No gpus available for use, exiting\n";
        exit(0);
    }
    if(dev > (devcount-1))
    {
        cout << "Warning: setdevice set to non-existent GPU, setting to default GPU 1\n";
        dev = 0;
    }
    cudaSetDevice(dev);
    //Future iterations of OCTANE may have error checking on the cuda functions below -J. Apke 2/21/2022

    cudaMallocManaged((void **) &GK, filtsize2*sizeof(float));

    //There are currently too many arrays declared for the gpu, so we are going to work to dual-purpose arrays where possible
    cudaMallocManaged((void **) &Aval, An*sizeof(float));
    cudaMallocManaged((void **) &Arow, An*sizeof(int));
    cudaMallocManaged((void **) &Acol, An*sizeof(int));
    cudaMallocManaged((void **) &uval, xityi*sizeof(float));//These are the in-out u and v arrays
    cudaMallocManaged((void **) &vval, xityi*sizeof(float));
    cudaMallocManaged((void **) &uvalt, xityi*sizeof(float));//these arrays are required for zoomins/first guess storage
    cudaMallocManaged((void **) &vvalt, xityi*sizeof(float));
    cudaMallocManaged((void **) &uhval, xityi*sizeof(float));//These hold the first guess values
    cudaMallocManaged((void **) &vhval, xityi*sizeof(float));
    
    cudaMallocManaged((void **) &geo1,       xityi*nchan*sizeof(float));
    cudaMallocManaged((void **) &geo2,       xityi*nchan*sizeof(float));
    cudaMallocManaged((void **) &geo10,       xityi*nchan*sizeof(float));
    cudaMallocManaged((void **) &geo20,       xityi*nchan*sizeof(float));
    cudaMallocManaged((void **) &gradxarrg2, xityi*nchan*sizeof(float));
    cudaMallocManaged((void **) &gradyarrg2, xityi*nchan*sizeof(float));
    cudaMallocManaged((void **) &gradxxarr,  xityi*nchan*sizeof(float));
    cudaMallocManaged((void **) &gradxyarr,  xityi*nchan*sizeof(float));
    cudaMallocManaged((void **) &gradyyarr,  xityi*nchan*sizeof(float));
    cudaMallocManaged((void **) &gradxarr,   xityi*nchan*sizeof(float));
    cudaMallocManaged((void **) &gradyarr,   xityi*nchan*sizeof(float));

    cudaMallocManaged((void **) &ArowSP, xityi2*sizeof(int));
    cudaMallocManaged((void **) &bcu, xityi2*sizeof(float));
    //Arrays needed for CG operations
    cudaMallocManaged((void **) &x0, xityi2*sizeof(float));
    cudaMallocManaged((void **) &p0, xityi2*sizeof(float));
    cudaMallocManaged((void **) &z0, xityi2*sizeof(float));
    cudaMallocManaged((void **) &rk, xityi2*sizeof(float));
    cudaMallocManaged((void **) &Mval, xityi2*sizeof(float));
    cudaMallocManaged((void **) &Mrow, xityi2*sizeof(int));
    bool dodiscrete=false;
    if(dodiscrete)
    {
        cudaMallocManaged((void **) &CTHc,       xityi*sizeof(float));
        cudaMallocManaged((void **) &CTHo,       xityi*sizeof(float));
    } else
    {
        cudaMallocManaged((void **) &CTHc,       1*sizeof(float)); //declare dummy arrays otherwise, don't waste memory
        cudaMallocManaged((void **) &CTHo,       1*sizeof(float));
    }
    //Below is a dummy storage vector, designed for some of the cuda functions used
    if(nchan < 2)
    {
        cudaMallocManaged((void **) &dummyvec, xityi2*sizeof(float));
    } else {
        cudaMallocManaged((void **) &dummyvec, xityi*nchan*sizeof(float));
    }
    cudaMallocManaged((void **) &rkTzk, sizeof(float));
    *rkTzk = 0.0;
    cudaMallocManaged((void **) &pkTApk, sizeof(float));
    *pkTApk = 0.0;
    cudaMallocManaged((void **) &zktrk, sizeof(float));
    *zktrk = 0.0;
    cudaMallocManaged((void **) &z0tr0, sizeof(float));
    *z0tr0 = 0.0;
    cudaMallocManaged((void **) &residc, sizeof(float));
    *residc = 0.0;

    for(int i=0; i < xityi; i++)
    {
        uval[i] = uarr[i];
        vval[i] = varr[i];
        uhval[i] = uarr[i];
        vhval[i] = varr[i];
        if(dodiscrete)
        {
            CTHc[i] = CTH[i];
            CTHo[i] = CTH[i];
        }
    }
    for(int i=0; i < xityi2; i++)
    {
        x0[i] =  0.; 
    }
    for(int i=0; i < xityi*nchan; i++)
    {
        geo1[i] =   geo1i.data[i];
        geo2[i] = geo2i.data[i];
        geo10[i] =  geo1i.data[i];
        geo20[i] = geo2i.data[i];
    }
    float tol = 0.0001*0.0001;

    //below stores all the arguments for the big CG function
    //Careful modifying these

    void *kernelArgs[] = {
        (void*) &Aval,
        (void*) &Arow, 
        (void*) &ArowSP, 
        (void*) &Acol, 
        (void*) &x0,
        (void*) &uval,
        (void*) &vval,
        (void*) &uvalt,
        (void*) &vvalt,
        (void*) &uhval,
        (void*) &vhval,
        (void*) &geo1,
        (void*) &geo2,
        (void*) &geo10,
        (void*) &geo20,
        (void*) &CTHc,
        (void*) &CTHo,
        (void*) &GK,
        (void*) &gradxarrg2,
        (void*) &gradyarrg2,
        (void*) &gradxxarr,
        (void*) &gradxyarr,
        (void*) &gradyyarr,
        (void*) &gradxarr,
        (void*) &gradyarr,
        (void*) &nchan,
        (void*) &alpha,
        (void*) &lambdadalpha,
        (void*) &lambdac,
        (void*) &An,
        (void*) &xityi2,
        (void*) &nx,
        (void*) &ny,
        (void*) &bcu,
        (void*) &Mval,
        (void*) &Mrow,
        (void*) &z0,
        (void*) &p0, 
        (void*) &residc,
        (void*) &tol,
        (void*) &iters,
        (void*) &rk,
        (void*) &zktrk,
        (void*) &z0tr0,
        (void*) &rkTzk,
        (void*) &pkTApk,
        (void*) &dummyvec,
        (void*) &liters,
        (void*) &kiters,
        (void*) &filtsize,
        (void*) &filtsize2,
        (void*) &scaleFactor,
        (void*) &scsig,
        (void*) &dozim,
        (void*) &dodiscrete,
    };
	int sMemSize = sizeof(double) * threadsPerBlock;
    int numBlocksPerSm = 0;
    int numThreads = threadsPerBlock;

    //Wrapped in an error checker in case your device cannot perform cooperative group processing
    //Also checks to see if you allocated too many threads/block for your GPU memory
    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, octConjugateGradient, numThreads, sMemSize));
    int numSms = 20;
    cout << "Number of Blocks per SM (there are 20 SMs) " << numBlocksPerSm << endl;
    

    dim3 dimGrid(numSms*numBlocksPerSm, 1, 1), dimBlock(threadsPerBlock, 1, 1);
    cudaDeviceProp deviceProp;


    cudaGetDeviceProperties(&deviceProp,dev);
    checkCudaErrors(cudaLaunchCooperativeKernel((void *)octConjugateGradient, dimGrid,dimBlock, kernelArgs, sMemSize, NULL));
    cudaDeviceSynchronize();
    //the conjugate gradient algorithm returns the optical flow output, which is stored back in the uarr/varr CPU array
    for(int i=0; i < xityi; i++)
    {
        uarr[i] = uval[i];
        varr[i] = vval[i];
    }
    

    cudaFree(Aval); 
    cudaFree(Arow);
    cudaFree(uval); 
    cudaFree(vval);
    cudaFree(uhval); 
    cudaFree(vhval);
    cudaFree(uvalt); 
    cudaFree(vvalt);
    cudaFree(ArowSP);
    cudaFree(Acol);

    cudaFree(CTHc); 
    cudaFree(CTHo); 
    cudaFree(geo1);
    cudaFree(geo2);
    cudaFree(geo10);
    cudaFree(geo20);
    cudaFree(GK);
    cudaFree(gradxarrg2);
    cudaFree(gradyarrg2);
    cudaFree(gradxxarr);
    cudaFree(gradxyarr);
    cudaFree(gradyyarr);
    cudaFree(gradxarr);
    cudaFree(gradyarr);
    cudaFree(bcu);
    cudaFree(x0);
    cudaFree(p0);
    cudaFree(z0);
    cudaFree(rk);
    cudaFree(Mval);
    cudaFree(dummyvec);
}
