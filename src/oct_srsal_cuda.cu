#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "oct_bicubic.h"
#include "oct_gaussian.h"
#include "offlags.h"
using namespace std;
//Function: oct_latlon2xy
//Purpose: This is a C++ function with a cpu based (and gpu based) bilateral filter, meant to match
//the objective analysis performed by Apke et al. 2018: Relationships between deep convection updraft
//characteristics and satellite based super rapid scan mesoscale atmospheric motion vecter derived flow
//
//Author: Jason Apke, Updated 11/27/2019
//A cuda device boundary condition check, note, different from bc in variational_optical_flow
__device__
int oct_bc_cuda(int x, int nx)
{
    //Reflecting boundary conditions, now computed on GPU
    if(x < 0)
    {
        x = 0-x;
    }
    if(x >= nx)
    {
        x = nx - (x-nx+1);
    }
    return x;
}
__device__ 
long oct_lxyz(int x, int y, int nx)
{
    return x+y*nx;
}
__global__
void octsrsalcuda(  int n, double *upix, double *vpix,double *GK, double *CTHsub21,int *icarr, int *jcarr, 
                    int nx, int ny, int filtsize, double sigpix2, double *upix2, double *vpix2)
{

    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int stride= blockDim.x*gridDim.x;
    for (int lxyz = index; lxyz < n; lxyz+=stride)
    {
        int ic=icarr[lxyz];
        int jc=jcarr[lxyz];

        float pixc = CTHsub21[lxyz];
        double au = 0;
        double av = 0;
        double a2 = 0;
        for(int kc = 0; kc < 2*filtsize+1; kc++)
        {
            for(int lc = 0; lc <2*filtsize+1 ; lc++)
            {
                //first determine the color distance in radiance space I suppose
                int kcv = kc-filtsize;
                int lcv = lc-filtsize;
                int ivc = (int) oct_bc_cuda(ic+kcv,nx);
                int jvc = (int) oct_bc_cuda(jc+lcv,ny);
                long lxyz2 = oct_lxyz(ivc,jvc,nx); 
                float pixl = CTHsub21[lxyz2];
                double pixm = pixl-pixc;
                double a1 = GK[kc]*GK[lc]*exp((pixm)*(pixm)*sigpix2);
                a2 += a1;
                au += upix[lxyz2]*a1;
                av += vpix[lxyz2]*a1;
            }
        }
        upix2[lxyz] = (au/a2);
        vpix2[lxyz] = (av/a2);
    }
}

void oct_srsal_cu(float *upix,float *vpix,float *CTHsub21,int nx, int ny, OFFlags args)
{
    double sigpix = 20.;
    double sigpix2 = -1./(sigpix*sigpix*2.);

    double filtsigma=9; 
    int filtsize=2*filtsigma; 
    double *GK;
    GK = new double[2*filtsize+1];
    oct_getGaussian_1D(GK,2*filtsize+1,filtsigma);
    int n=nx*ny;
    double *u, *v, *GK1D, *CTH,*up2,*vp2;
    int *icarr,*jcarr;
    int devcount=0;
    cudaGetDeviceCount(&devcount);
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
    cudaMallocManaged(&u, nx*ny*sizeof(double));
    cudaMallocManaged(&icarr, nx*ny*sizeof(int));
    cudaMallocManaged(&jcarr, nx*ny*sizeof(int));
    cudaMallocManaged(&v, nx*ny*sizeof(double));
    cudaMallocManaged(&CTH, nx*ny*sizeof(double));
    cudaMallocManaged(&up2, nx*ny*sizeof(double));
    cudaMallocManaged(&vp2, nx*ny*sizeof(double));
    cudaMallocManaged(&GK1D, (2*filtsize+1)*sizeof(double));

    for(int l=0; l<ny; l++)
    {
        long nxtj = nx*l;
        for(int i=0; i<nx; i++)
        {
            //Fill the arrays
            long lxyz = i+nxtj;
            u[lxyz] = upix[lxyz];
            v[lxyz] = vpix[lxyz];
            CTH[lxyz] = (double) CTHsub21[lxyz];
            up2[lxyz] = 0;
            vp2[lxyz] = 0;
            icarr[lxyz] = i;
            jcarr[lxyz] = l;
        }
    }
    for(int j = 0; j< 2*filtsize+1; j++) GK1D[j] = GK[j];
    cudaDeviceSynchronize();
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1)/blockSize;
    octsrsalcuda<<<numBlocks,blockSize>>>(n, u, v,GK1D, CTH,icarr,jcarr,nx,ny, filtsize, sigpix2, up2, vp2);
    cudaDeviceSynchronize();
    for(int k=0; k<n; k++)
    {
        upix[k] = up2[k];
        vpix[k] = vp2[k];
    }


    cudaFree(u);
    cudaFree(v);
    cudaFree(CTH);
    cudaFree(up2);
    cudaFree(vp2);
    cudaFree(GK1D);
    cudaFree(icarr);
    cudaFree(jcarr);
    delete [] GK;
}
