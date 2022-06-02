#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "oct_bicubic.h"
#include "offlags.h"
using namespace std;
//Purpose: This is a mercator file navigation and calibration function accelerated by the cuda library 
//Author: Jason Apke, Updated 11/27/2019

__global__
void octmercnavcalcuda(  int n, int *icarr, int *jcarr,int minx, int maxx, int miny, short *x, short *y, float *data2,
                     float xScale, float xOffset, float yScale, float yOffset,
                     float R, float lon0,  int donav,float *data3,float *lat, float *lon,long *lxyzarr)
{
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
		float dVal = data2[lxyz]; //*radScale+radOffset;

		if(donav == 1)
		{
            //This is the navigation code for mercator grid (all you need is earth radius and lon0 for sphere)
            lon[lxyz2] = xVal/R + lon0;
            lat[lxyz2] = PI/2. - 2. * atan(exp(-yVal/R));

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

void oct_merc_navcal_cuda(float *data2,short *data2s, short *x, short *y, short *xs, short *ys, int nx, 
                     int ny, int minx, int maxx, int miny, int maxy, float *data3, float *lat, 
                     float *lon,float xScale, float xOffset, float yScale,
                     float yOffset, float lon0, 
                     float R,int donav,OFFlags args)
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
                data2s[lxyz2] = 0; 
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
    octmercnavcalcuda<<<numBlocks,blockSize>>>(n2,icarr,jcarr,minx, maxx, miny, xcu, ycu, data2cu,
                     xScale, xOffset, yScale, yOffset,
                     R, lon0*DTOR, donav,data3cu,latcu, loncu,lxyzarr);
    cudaDeviceSynchronize();
    for(int k=0; k<n2; k++)
    {
        data3[k] = data3cu[k];
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
