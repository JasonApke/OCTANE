#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "oct_bicubic.h"
#include "image.h"
#include "goesread.h"
#include "offlags.h"
using namespace std;
//Purpose: This is a cuda function to convert pixel u-v to navigated u/v speeds, includes polar/mercator options 
//Author: Jason Apke, Updated 2/23/2022
__device__
double oct_haversine_cuda (float lat1,float lon1,float lat2,float lon2,double rad,double rad2)
{
    const double earthrad = 6371000.00;
    double a,c,r,dlat,dlon;

    dlon = lon2 - lon1;
    dlat = lat2 - lat1;
    a = (pow(sin(dlat*rad2),2) + cos(lat1*rad) * cos(lat2*rad) * pow((sin(dlon*rad2)),2));
    c = 2. * atan2(sqrt(a),sqrt(1-a));
    r = earthrad*c;

    return r;
} 
__device__
void oct_navpixel_uv_cuda(GOESNAVVar geo,double *xv,int xi, int yi, double dt,double *r,double DTOR,double DTOR2,bool dp,bool dm)
{
    const double PI=3.14159265359;
    double xVal,yVal,dist;
    double latv[2],lonv[2],sds[2];
    sds[0] = 0.;
    sds[1] = 0.;
    if(dp){

        for(int iv=0; iv<2; ++iv)
        {
            if(iv == 0)
            {
                xVal = (xi)*geo.xScale+geo.xOffset;
                yVal = (yi)*geo.yScale+geo.yOffset;
            } else {
                xVal = (xv[0]*dt+xi)*geo.xScale+geo.xOffset;
                yVal = (xv[1]*dt+yi)*geo.yScale+geo.yOffset;
            }
            double rho = sqrt(xVal*xVal+yVal*yVal);
            double c = asin(rho/geo.R);

            if(geo.lat1 > 89.9999)
            {
                lonv[iv] = geo.lon0*DTOR+atan2(xVal,-yVal);
            } else
            {
                lonv[iv] = geo.lon0*DTOR+atan2(xVal*sin(c),(rho*cos(geo.lat1*DTOR)*cos(c)-yVal*sin(geo.lat1*DTOR)*sin(c)));
            }
            if(rho > 0.0000001)
            {
                latv[iv] = asin(cos(c)*sin(geo.lat1*DTOR)+(yVal*sin(c)*cos(geo.lat1*DTOR)/rho));
            } else
            {
                latv[iv] = geo.lat1*DTOR;
            }

            latv[iv] = latv[iv]/DTOR;
            lonv[iv] = lonv[iv]/DTOR;
        }

    } else
    {
        if(dm){

            for(int iv=0; iv<2; ++iv)
            {

                if(iv == 0)
                {
                    xVal = (xi)*geo.xScale+geo.xOffset;
                    yVal = (yi)*geo.yScale+geo.yOffset;
                } else {
                    xVal = (xv[0]*dt+xi)*geo.xScale+geo.xOffset;
                    yVal = (xv[1]*dt+yi)*geo.yScale+geo.yOffset;
                }
                latv[iv] = PI/2.-2.*atan(exp(-yVal/geo.R));
                lonv[iv] = xVal/geo.R + geo.lon1;
                latv[iv] = latv[iv]/DTOR; //convert to radians
                lonv[iv] = lonv[iv]/DTOR;
            }
        } else
        {
            double a, b, c,d,e, rs, sx, sy, sz; 
            double H;
            H = geo.pph+geo.req;

            for(int iv=0; iv<2; ++iv)
            {
                
                if(iv == 0)
                {
                    xVal = (xi)*geo.xScale+geo.xOffset;
                    yVal = (yi)*geo.yScale+geo.yOffset;
                } else {
                    xVal = (xv[0]*dt+xi)*geo.xScale+geo.xOffset;
                    yVal = (xv[1]*dt+yi)*geo.yScale+geo.yOffset;
                }
                sds[iv] = xVal*xVal+yVal*yVal;


                a = pow((sin(xVal)),2)+pow(cos(xVal),2)*(pow((cos(yVal)),2)+(pow(geo.req,2))/(pow(geo.rpol,2))*pow((sin(yVal)),2));
                b = -2.* H*cos(xVal)*cos(yVal);
                c = pow(H,2) - pow(geo.req,2);
                d = (pow(b,2) - 4.*a*c);
                if(d >= 0)
                {
                    rs = (-b - sqrt(d))/(2.*a);
                    sx = rs*cos(xVal)*cos(yVal);
                    sy = -rs*sin(xVal);
                    sz = rs*cos(xVal)*sin(yVal);
                    e = (pow((H-sx),2) +pow(sy,2));
                    if(sz == 0 || e <= 0 || H-sx == 0)
                    {
                        latv[iv] = -999.; 
                        lonv[iv] = -999.;
                        latv[iv] = -999.;
                        lonv[iv] = -999.;
                    } else{

                        latv[iv] = atan((pow(geo.req,2))/(pow(geo.rpol,2))*(sz/sqrt(e)));
                        lonv[iv] = geo.lam0 - atan(sy/(H-sx));
                        latv[iv] = latv[iv]/DTOR;
                        lonv[iv] = lonv[iv]/DTOR;
                    }
                } else
                {
                        latv[iv] = -999.; 
                        lonv[iv] = -999.;
                        latv[iv] = -999.;
                        lonv[iv] = -999.;
                }
            }
        }
    }

    //compute U
    if((latv[0] < -998) || (latv[1] < -998) || (sds[0] > 0.021))
    {
        r[0] = 0.;
        r[1] = 0.; //u and v are technically off the surface of the earth or outside my subpoint distance threshold, but I am setting them to 0 as a fill
    } else{
        //approximating distances with spherical motion
        //here is zonal motion (u)
        dist = oct_haversine_cuda(latv[0],lonv[0],latv[0],lonv[1],DTOR,DTOR2);
        if(lonv[1] >= lonv[0])
        {
            r[0] = dist/dt;
        } else
        {
            r[0] = -dist/dt;
        }
        //here is meridional motion (v)
        dist = oct_haversine_cuda(latv[0],lonv[0],latv[1],lonv[0],DTOR,DTOR2);
        if(latv[1] >= latv[0])
        {
            r[1] = dist/dt;
        } else
        {
            r[1] = -dist/dt;
        }
    }



}
__global__
void octnavcalcuda(  int n, int *icarr, int *jcarr,double DTOR, double DTOR2,GOESNAVVar nav1,
        double t1, double t2,double *u1, double *v1, short *ur,short *vr, 
        double *upix, double *vpix,bool dp,bool dm)
{
    double dans[2], xans[2];
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int stride= blockDim.x*gridDim.x;
    for (int lxyz = index; lxyz < n; lxyz+=stride)
    {
        int ii = icarr[lxyz];
        int jj = jcarr[lxyz];

            if(u1[lxyz] > -9998.)
            {
                dans[0] = u1[lxyz]/(t2-t1);
                dans[1] = v1[lxyz]/(t2-t1);


                oct_navpixel_uv_cuda(nav1,dans,ii+nav1.minX,jj+nav1.minY,t2-t1,xans,DTOR,DTOR2,dp,dm);



                ur[lxyz] = (short) (100*(xans[0]));
                vr[lxyz] = (short) (100*(xans[1])); 
                double spdval = sqrt(xans[0]*xans[0]+xans[1]*xans[1]);

                double dirval = (270.-atan2(-dans[1],dans[0])/DTOR)*DTOR; //confusing I know, but the first dimension is the y in GOES data
                if(dp)
                {
                    //For polar, we usually try and track sea ice movement which is slow and tough to scale appropriately, I save it as a double
                    upix[lxyz] = xans[0]; 
                    vpix[lxyz] = xans[1]; 
                }
                else{
                    upix[lxyz] = (-spdval*sin(dirval));
                    vpix[lxyz] = (-spdval*cos(dirval));
                }

            } else
            {
                xans[0] = -9999.;
                xans[1] = -9999.;
                ur[lxyz] = (short) (-32768);
                vr[lxyz] = (short) (-32768);
            }
    } // end nav loop

}
__global__
void octuv2xy(  int n, double *u, double *v, double *lat, double *lon,double *x1v, double *y1v,double secs,
        double req,double req2,double rpol,double rpol2,double eval,double lam0,double pph,float xscale, float xoffset,float yscale,float yoffset)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int stride= blockDim.x*gridDim.x;
	const double R = 6371000.0;
    const double pi=3.14159265;
    double rad = pi/180.;
    double H = pph+req;
    for (int lxyz = index; lxyz < n; lxyz+=stride)
    {
		double u1 = u[lxyz];
        double v1 = v[lxyz];
        double latvalv = lat[lxyz];
        double lonvalv = lon[lxyz];
        //haversine bearing function
		double dist = sqrt(pow(u1,2.0) + pow(v1,2.0))*(secs);
		double brng = (180.+(90.-(atan2(-v1,-u1)/rad)))*rad;
		double latorig = latvalv*rad;
		latvalv = asin(sin(latorig)*cos(dist/R) + cos(latorig)*sin(dist/R)*cos(brng));
		lonvalv = lonvalv*rad+ (atan2((sin(brng)*sin(dist/R)*cos(latorig)),(cos(dist/R)-sin(latorig)*sin(latvalv))));


        double thtc = atan(((rpol2)/(req2))*tan(latvalv));
        double rc = rpol/sqrt(1.-(eval)*pow(cos(thtc),2.));

        double sx = H - rc*cos(thtc)*cos(lonvalv-lam0);
        double sy = -rc*cos(thtc)*sin(lonvalv-lam0);
        double sz = rc*sin(thtc);
        if((H*(H-sx)) >= (sy*sy+((req2)/(rpol2)*sz*sz)))
        {
            x1v[lxyz] = (asin(-sy/(sqrt(sx*sx+sy*sy+sz*sz)))-xoffset)/xscale;
            y1v[lxyz] = (atan(sz/sx)-yoffset)/yscale;
            //}
        } else{
            x1v[lxyz] = -999.;
            y1v[lxyz] = -999.; //Fill values

        }
    }
}
//A cuda function to convert 
void oct_pix2uv_cuda(GOESVar &goesData,double t2,float *uarr, float *varr, short *ur, short *vr,short *ur2,short *vr2,OFFlags args)
{
    int   n=goesData.nav.nx*goesData.nav.ny;
    int   *icarr,*jcarr;
    short *urcu,*vrcu; //,*spdacu;
    double *upixcu,*vpixcu,*u1cu,*v1cu;
    double pi=3.14159265;
    double rad = pi/180.;
    double rad2 = rad/2.;
    bool dp=false,dm=false;
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
    if(args.dopolar==1) dp =true;
    if(args.domerc==1) dm =true;

    //Declare the arrays in a memory space both the CPU and GPU can see
    if((((goesData.nav.xOffset - goesData.nav.g2xOffset)*(goesData.nav.xOffset - goesData.nav.g2xOffset)) < (0.00001*0.00001)) && (((goesData.nav.yOffset - goesData.nav.g2yOffset)*(goesData.nav.yOffset - goesData.nav.g2yOffset)) < (0.00001*0.00001)))
    {
    if(args.pixuv == 0)
    {
        cudaMallocManaged(&urcu, n*sizeof(short));
        cudaMallocManaged(&vrcu, n*sizeof(short));
        cudaMallocManaged(&icarr, n*sizeof(int));
        cudaMallocManaged(&jcarr, n*sizeof(int));
        cudaMallocManaged(&upixcu, n*sizeof(double));
        cudaMallocManaged(&vpixcu, n*sizeof(double));
        cudaMallocManaged(&u1cu, n*sizeof(double));
        cudaMallocManaged(&v1cu, n*sizeof(double));

        for(int j=0; j<goesData.nav.ny; j++)
        {
            long nxtj = goesData.nav.nx*j;
            for(int i=0; i<goesData.nav.nx; i++)
            {
                //Fill the arrays
                long lxyz   = i+nxtj;
                icarr[lxyz] = i;
                jcarr[lxyz] = j;
                u1cu[lxyz] = uarr[lxyz];
                v1cu[lxyz] = varr[lxyz];

            }
        }
        cudaDeviceSynchronize();
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1)/blockSize;
        octnavcalcuda<<<numBlocks,blockSize>>>(  
                            n, icarr,jcarr,rad, rad2,goesData.nav,
                            goesData.t, t2,u1cu,v1cu,urcu,vrcu, 
                            upixcu, vpixcu,dp,dm);
        cudaDeviceSynchronize();
        for(int k=0; k<n; k++)
        {
            //Fill the arrays
            ur[k] = urcu[k];
            vr[k] = vrcu[k];
            ur2[k] = (short)(100*uarr[k]);
            vr2[k] = (short)(100*varr[k]);
        }

        cudaFree(urcu);
        cudaFree(vrcu);
        cudaFree(upixcu);
        cudaFree(vpixcu);
        cudaFree(icarr);
        cudaFree(jcarr);
        cudaFree(u1cu);
        cudaFree(v1cu);
        goesData.dT = (float) (t2-goesData.t); //time change in seconds
    } else {
        for(int k=0; k<n; k++)
        {
            //Fill the arrays
            ur[k] = (short)(100*uarr[k]); //scaled for storage purposes, output upix/vpix for unscaled pixel motion
            vr[k] = (short)(100*varr[k]);
        }
        goesData.dT = (float) (t2-goesData.t); //time change in seconds

    }
    } else {
        cout << "MOVE WARNING: Sector Moved, setting motions to 0 " << goesData.nav.xOffset << " " << goesData.nav.g2xOffset << " " << goesData.nav.yOffset << " " << goesData.nav.g2yOffset << endl;
        for(int k=0; k<n; k++)
        {
            ur[k] = 0.;
            vr[k] = 0.;
            ur2[k] = (short)(100*0.);
            vr2[k] = (short)(100*0.);
        }
        goesData.dT = (float) (t2-goesData.t); //time change in seconds

    }
}

void oct_uv2pix(GOESVar &goesData,float *u, float *v, double t2,OFFlags args)
{
    int   n=goesData.nav.nx*goesData.nav.ny;
    double *ucu,*vcu,*latcu,*loncu,*x1v,*y1v;
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
    cudaMallocManaged(&ucu, n*sizeof(double));
    cudaMallocManaged(&vcu, n*sizeof(double));
    cudaMallocManaged(&latcu, n*sizeof(double));
    cudaMallocManaged(&loncu, n*sizeof(double));
    cudaMallocManaged(&x1v, n*sizeof(double));
    cudaMallocManaged(&y1v, n*sizeof(double));

    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1)/blockSize;

    for(int i=0; i<goesData.nav.nx; i++)
    {
        for(int j=0; j<goesData.nav.ny; j++)
        {
            //Fill the arrays
            long lxyz   = i+j*goesData.nav.nx;
            ucu[lxyz] = u[lxyz];
            vcu[lxyz] = v[lxyz];
            latcu[lxyz] = goesData.latVal[lxyz];
            loncu[lxyz] = goesData.lonVal[lxyz];
            x1v[lxyz] = goesData.x[i]; //15.;
            y1v[lxyz] = goesData.y[j];
        }
    }
    double req2 = goesData.nav.req * goesData.nav.req;
    double rpol2 = goesData.nav.rpol * goesData.nav.rpol;
	double eval = sqrt((req2-rpol2)/(req2));
    bool dg=true;
    if((goesData.nav.xOffset == goesData.nav.g2xOffset) && (goesData.nav.yOffset == goesData.nav.g2yOffset))
    {

        if(args.domerc == 1)
        {
            dg = false;
        }
        eval = eval*eval;
        if(dg){
            octuv2xy<<<numBlocks,blockSize>>>(    
                n, ucu, vcu, latcu, loncu,x1v, y1v,t2-goesData.t,
                goesData.nav.req,req2,goesData.nav.rpol,rpol2,eval,goesData.nav.lam0,goesData.nav.pph,goesData.nav.xScale,goesData.nav.xOffset,goesData.nav.yScale,goesData.nav.yOffset);
            cudaDeviceSynchronize();
        } else
        {
            octuv2xy<<<numBlocks,blockSize>>>(    
                n, ucu, vcu, latcu, loncu,x1v, y1v,t2-goesData.t,
                goesData.nav.req,req2,goesData.nav.rpol,rpol2,eval,goesData.nav.lam0,goesData.nav.pph,goesData.nav.xScale,goesData.nav.xOffset,goesData.nav.yScale,goesData.nav.yOffset);
            cudaDeviceSynchronize();
        }
                
        for(int i2=0; i2<goesData.nav.nx; i2++)
        {
            for(int j2=0; j2<goesData.nav.ny; j2++)
            {
                int k = i2+j2*goesData.nav.nx;
                if(x1v[k] > -998.)
                {
                    u[k] = x1v[k]-goesData.x[i2];
                    v[k] = y1v[k]-goesData.y[j2];
                } else{
                    u[k] = 0.;
                    v[k] = 0.; //weight to stationary if off the map
                }
            }
        }
    } else{
        for(int i2=0; i2<goesData.nav.nx; i2++)
        {
            for(int j2=0; j2<goesData.nav.ny; j2++)
            {
                int k = i2+j2*goesData.nav.nx;
                //This happens when they move the mesosector, just set everything to 0 in this case
                u[k] = 0.;
                v[k] = 0.;
            }
        }

    }
    cudaFree(ucu);
    cudaFree(vcu);
    cudaFree(latcu);
    cudaFree(loncu);
    cudaFree(x1v);
    cudaFree(y1v);
}
