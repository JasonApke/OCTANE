#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "oct_bicubic.h"
#include "offlags.h"
using namespace std;
//Purpose: This is a navigation and calibration function accelerated by the cuda library 
//Author: Jason Apke, Updated 11/27/2019

__global__
void octnavcalcuda(  int n, int *icarr, int *jcarr,int minx, int maxx, int miny, short *x, short *y, short *data2,
                     float xScale, float xOffset, float yScale, float yOffset, float radScale, float radOffset,
                     float rpol, float req, float H, float lam0,float fk1,float fk2, float bc1, float bc2, 
                     float kap1, int donav,float *data3,float *lat, float *lon,int cal,int datf,long *lxyzarr,
                     float maxin, float minin, float maxout, float minout,float subpoint_slope,float subpoint_int)
{
    const double PI=3.14159265359;
    const double DTOR = PI/180.;
    double a,b,c,rs,sx,sy,sz,dataF;
    float sdsconst;
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int stride= blockDim.x*gridDim.x;
    for (int lxyzn = index; lxyzn < n; lxyzn+=stride)
    {
        long lxyz=lxyzarr[lxyzn];
        int i=icarr[lxyz];
        int j=jcarr[lxyz];
        long lxyz2=(i-minx)+(maxx-minx)*(j-miny);
        //
		double xVal = x[i]*xScale+xOffset;
		double yVal = y[j]*yScale+yOffset;
        double subpoint_dist = xVal*xVal+yVal*yVal;
		float dVal = data2[lxyz]*radScale+radOffset;

		if(donav == 1)
		{
			a = pow((sin(xVal)),2)+pow(cos(xVal),2)*(pow((cos(yVal)),2)+(pow(req,2))/(pow(rpol,2))*pow((sin(yVal)),2));
			b = -2.* H*cos(xVal)*cos(yVal);
			c = pow(H,2) - pow(req,2);
			rs = (-b - sqrt((pow(b,2) - 4.*a*c)))/(2.*a);
			sx = rs*cos(xVal)*cos(yVal);
			sy = -rs*sin(xVal);
			sz = rs*cos(xVal)*sin(yVal);
			lat[lxyz2] = atan(double((pow(req,2))/(pow(rpol,2)))*(sz/sqrt((pow((H-sx),2) +pow(sy,2)))));
			lon[lxyz2] = lam0 - atan(sy/(H-sx));
			lat[lxyz2] = lat[lxyz2]/DTOR;
			lon[lxyz2] = lon[lxyz2]/DTOR;
		}
		else
		{
			lat[lxyz2] = 0.;
			lon[lxyz2] = 0.;
		}
		if(cal == 0)
		{
			//for raw counts, you have to convert back from radiance
			dataF=dVal;
			datf = 1;
		}
		if(cal == 1)
		{
			dataF=(fk2/(log((fk1/dVal)+1.))-bc1)/bc2;
			datf = 1;
		}
		if(cal == 2)
		{
			dataF=kap1*dVal;
			datf = 1;
		}
		if(cal == 3)
		{
			dataF=dVal;
			datf = 1;
		}
		if(datf == 0)
		{
			dataF=dVal;
		}
        //Future feature will be to make the subpoint distance something one can set
        if(subpoint_dist < 0.021){
            sdsconst = 1.;
        } else
        {
            if(subpoint_dist >= 0.0212)
            {
                sdsconst = 0.;
            } else{
                sdsconst = subpoint_slope*subpoint_dist+subpoint_int;
            }
        }
        //data normalization occurs here, note sdsconst is a variable from 0 to 1 which filters away points too close to the limb
        data3[lxyz2] = sdsconst*(((dataF-minin)/(maxin-minin))*(maxout-minout)+minout);



    }
}

void oct_navcal_cuda(short *data2,short *data2s, short *x, short *y, short *xs, short *ys, int nx, 
                     int ny, int minx, int maxx, int miny, int maxy, float *data3, float *lat, 
                     float *lon,string cal,int datf,float xScale, float xOffset, float yScale,
                     float yOffset, float radScale, float radOffset, float rpol, float req, 
                     float H, float lam0,float fk1,float fk2,float bc1,float bc2,
                     float kap1,float maxin, float minin, float maxout, float minout, int donav,OFFlags args)
{
    int   n=nx*ny,n2=(maxx-minx)*(maxy-miny);
    short *data2cu,*xcu, *ycu;
    float *data3cu,*latcu,*loncu;
    int   *icarr,*jcarr;
	long  *lxyzarr;
    int cali;

    //Declare the arrays in a memory space both the CPU and GPU can see
    if(cal=="RAW") cali=0;
    if(cal=="TEMP") cali = 1;
    if(cal=="REF") cali = 2;
    if(cal=="BRIT") cali = 3;
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
    cudaMallocManaged(&data2cu, n*sizeof(short));
    cudaMallocManaged(&icarr,   n*sizeof(int));
    cudaMallocManaged(&jcarr,   n*sizeof(int));
    //These arrays are sometimes subset for speed
    cudaMallocManaged(&data3cu, n2*sizeof(float));
    cudaMallocManaged(&lxyzarr, n2*sizeof(long));
    cudaMallocManaged(&latcu,   n2*sizeof(float));
    cudaMallocManaged(&loncu,   n2*sizeof(float));
    
    cudaMallocManaged(&xcu,   nx*sizeof(short));
    cudaMallocManaged(&ycu,   ny*sizeof(short));
    long maxxmminx = maxx-minx;
    for(int l=0; l<ny; l++)
    {
        long nxtl = nx*l;
        long maxxvalue = maxxmminx*(l-miny);

        for(int i=0; i<nx; i++)
        {
            //Fill the arrays
            long lxyz   = i+nxtl;
            data2cu[lxyz]     = data2[lxyz];
            icarr[lxyz]     = i;
            jcarr[lxyz]     = l;
            xcu[i] = x[i];
            ycu[l] = y[l];
            if(i >=minx && i < maxx && l >=miny && l < maxy)
            {
                long lxyz2 = (i-minx)+maxxvalue;
                data3cu[lxyz2]     = 0.;
                data2s[lxyz2] = data2[lxyz];
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
    float subpoint_slope = 1./(0.021-0.0212);
    float subpoint_int = 1.-0.021*subpoint_slope;
    cudaDeviceSynchronize();
    //Block size may be something I will add as a user setting -J. Apke 2/23/2022
    int blockSize = 512;
    int numBlocks = (n + blockSize - 1)/blockSize;
    octnavcalcuda<<<numBlocks,blockSize>>>(n2, icarr, jcarr,minx, maxx, miny, xcu, ycu, data2cu,
                                           xScale, xOffset, yScale, yOffset, radScale, radOffset,
                                           rpol, req, H, lam0,fk1,fk2, bc1, bc2, 
                                           kap1, donav,data3cu,latcu,loncu,cali,datf,lxyzarr,
                                           maxin, minin, maxout, minout,subpoint_slope,subpoint_int);
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
