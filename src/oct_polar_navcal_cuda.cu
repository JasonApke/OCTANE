#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "oct_bicubic.h"
#include "offlags.h"
using namespace std;
//Purpose: This is a navigation and calibration function for remapped polar files accelerated by the cuda library 
//Author: Jason Apke, Updated 11/27/2019

__global__
void octpolarnavcalcuda(  int n, int *icarr, int *jcarr,int minx, int maxx, int miny, short *x, short *y, float *data2,
                     float xScale, float xOffset, float yScale, float yOffset,
                     float R, float lon0, float lat1, int donav,float *data3,float *lat, float *lon,long *lxyzarr)
{
    double rho,c;
    const double PI=3.14159265359;
    const double DTOR = PI/180.;
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int stride= blockDim.x*gridDim.x;
    for (int lxyzn = index; lxyzn < n; lxyzn+=stride)
    {
        long lxyz=lxyzarr[lxyzn];
        int i=icarr[lxyz];
        int j=jcarr[lxyz];
        long lxyz2=(i-minx)+(maxx-minx)*(j-miny);
		double xVal = x[i]*xScale+xOffset;
		double yVal = y[j]*yScale+yOffset;
		float dVal = data2[lxyz]; 

		if(donav == 1)
		{
            //This is the navigation code for orthonormal polar grid
            rho = sqrt(xVal*xVal+yVal*yVal);
            c = asin(rho/R);

            if(lat1 > 89.99999)
            {
                lon[lxyz2] = lon0+atan2(xVal,-yVal);
            } else
            {
                lon[lxyz2] = lon0+atan2(xVal*sin(c),(rho*cos(lat1)*cos(c)-yVal*sin(lat1)*sin(c)));
            }
            if(rho > 0.0000001)
            {
            lat[lxyz2] = asin(cos(c)*sin(lat1)+(yVal*sin(c)*cos(lat1)/rho));
            } else
            {
                lat[lxyz2] = lat1;
            }

			lat[lxyz2] = lat[lxyz2]/DTOR;
			lon[lxyz2] = lon[lxyz2]/DTOR;
		}
		else
		{
			lat[lxyz2] = 0.;
			lon[lxyz2] = 0.;
		}
		data3[lxyz2] = dVal;



    }
}

void oct_polar_navcal_cuda(float *data2,short *data2s, short *x, short *y, short *xs, short *ys, int nx, 
                     int ny, int minx, int maxx, int miny, int maxy, float *data3, float *lat, 
                     float *lon,float xScale, float xOffset, float yScale,
                     float yOffset, float lon0, float lat1, 
                     float R,int donav,int chan,OFFlags args)
{
    int   n=nx*ny,n2=(maxx-minx)*(maxy-miny);
    short *xcu, *ycu;
    float *data3cu,*latcu,*loncu,*data2cu;
    int   *icarr,*jcarr;
	long  *lxyzarr;
    const double PI=3.14159265359;
    const double DTOR = PI/180.;
    
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

    //Declare the arrays in a memory space both the CPU and GPU can see
    cudaMallocManaged(&data2cu, n*sizeof(float));
    cudaMallocManaged(&icarr,   n*sizeof(int));
    cudaMallocManaged(&jcarr,   n*sizeof(int));
    //These arrays are sometimes subset for speed
    cudaMallocManaged(&data3cu, n2*sizeof(float));
    cudaMallocManaged(&lxyzarr, n2*sizeof(long));
    cudaMallocManaged(&latcu,   n2*sizeof(float));
    cudaMallocManaged(&loncu,   n2*sizeof(float));
    
    cudaMallocManaged(&xcu,   nx*sizeof(short));
    cudaMallocManaged(&ycu,   ny*sizeof(short));
    for(int i=0; i<nx; i++)
    {
        for(int l=0; l<ny; l++)
        {
            //Fill the arrays
            long lxyz   = i+nx*l;
            data2cu[lxyz]     = data2[lxyz];
            icarr[lxyz]     = i;
            jcarr[lxyz]     = l;
            xcu[i] = x[i];
            ycu[l] = y[l];
            if(i >=minx && i < maxx && l >=miny && l < maxy)
            {
                long lxyz2 = (i-minx)+(maxx-minx)*(l-miny);
                data3cu[lxyz2]     = 0.;
                data2s[lxyz2] = 0; //data2[lxyz];
                lxyzarr[lxyz2]= lxyz;
                if(l==miny)
                {
                    xs[i-minx] = x[i];
                }
                if(i==minx)
                {
                    ys[l-miny] = y[l];
                }
            }
        }
    }
    cudaDeviceSynchronize();
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1)/blockSize;
    octpolarnavcalcuda<<<numBlocks,blockSize>>>(n2,icarr,jcarr,minx, maxx, miny, xcu, ycu, data2cu,
                     xScale, xOffset, yScale, yOffset,
                     R, lon0*DTOR, lat1*DTOR, donav,data3cu,latcu, loncu,lxyzarr);
    cudaDeviceSynchronize();
    long nxtnytnc = (chan-1)*n2;
    for(int k=0; k<n2; k++)
    {
        data3[k+nxtnytnc] = data3cu[k];
        lat[k] = latcu[k];
        lon[k] = loncu[k];
    }


    cudaFree(data2cu);
    cudaFree(icarr);
    cudaFree(jcarr); 
    cudaFree(data3cu);
    cudaFree(lxyzarr);
    cudaFree(latcu); 
    cudaFree(loncu); 
    cudaFree(xcu);
    cudaFree(ycu);
}
