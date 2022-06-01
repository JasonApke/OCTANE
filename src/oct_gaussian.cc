#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "util.h"
#include "oct_gaussian.h"
#include "oct_bc.h"
using namespace std;
//Purpose: A collection of functions for gaussian smoothing on the CPU.
//Note: Variational_Optical_Flow now performs this on the GPU (faster), so below
//are only used for file-read image smoothing now
void oct_getGaussian(double **GKernel,int wk,double sigma)
{
    double r, s = 2.0 * sigma * sigma;

    // sum is for normalization 
    double sum = 0.0;
    int wk2 = (wk-1)/2;
    // generating wkxwk kernel 
    for (int x = -wk2; x <= wk2; x++) {
        for (int y = -wk2; y <= wk2; y++) {
            r = sqrt(x * x + y * y);
            GKernel[x + wk2][y + wk2] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += GKernel[x + wk2][y + wk2];
        }
    }

    // normalising the Kernel 
    for (int i = 0; i < wk; ++i)
        for (int j = 0; j < wk; ++j)
            GKernel[i][j] /= sum;
}

void oct_getGaussian_1D(double *GKernel,int wk,double sigma)
{
    double r, s = 2.0 * sigma * sigma;

    double sum = 0.0;
    int wk2 = (wk-1)/2;
    for (int x = -wk2; x <= wk2; x++) {
            r = x ;
            GKernel[x + wk2] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += GKernel[x + wk2];
    }

    for (int i = 0; i < wk; ++i)
        GKernel[i] /= sum;
}
void oct_gaussian(double * image, int nx, int ny,double sigma)
{
    double * geosub112;
    double *GK;
    bool bc;
    int filtsize = (int) 2*sigma;
    if (filtsize <5)
        filtsize=5; 
    geosub112 = new double[nx*ny];

    GK = new double [2*filtsize+1];
    oct_getGaussian_1D(GK,2*filtsize+1,sigma);
    for(int ii2 = 0; ii2<nx; ++ii2)
    {
        for (int jj2 = 0; jj2<ny; ++jj2)
        {
            long lxyz = ii2+nx*jj2;
            double wsum;
            double wnum = 0.;
            wsum = 0;
            //Horizontal convolution
            for(int kk2 = -filtsize; kk2 < filtsize; ++kk2)
            {
                //weighted average
                int iiv = (int) oct_bc<int>(ii2+kk2,nx,bc);
                int lxyz_en = iiv+nx*jj2; //(ii2+kk2)+nx*(jj2+ll2);
                wsum = wsum+GK[kk2+filtsize]*image[lxyz_en];
            }
            //average
            //vertical convolution
            geosub112[lxyz] = wsum;
            

        }
    }
    for(int ii23 = 0; ii23<nx; ++ii23)
    {
        for (int jj23 = 0; jj23<ny; ++jj23)
        {
            long lxyz = ii23+nx*jj23;
            double wsum = 0;
            //VERTICAL CONVOLUTION
            for(int ll2 = -filtsize; ll2 < filtsize; ++ll2)
            {
                int jjv = (int) oct_bc<int>(jj23+ll2,ny,bc);
                long lxyz_en = ii23+nx*jjv;
                wsum = wsum+GK[ll2+filtsize]*geosub112[lxyz_en];
            }
            image[lxyz] = wsum;
        }
    }


    delete [] geosub112;
    delete [] GK;
}
void oct_gaussian2(double ** image, int nx, int ny,double sigma,int nchan)
{
    //A gaussian function specifically designed for multi-channel
    double * geosub112;
    double *GK;
    bool bc;
    int filtsize = (int) 2*sigma; 
    if (filtsize <5)
        filtsize=5; 
    geosub112 = new double[nx*ny];

    GK = new double [2*filtsize+1];
    oct_getGaussian_1D(GK,2*filtsize+1,sigma);
    for(int nc = 0; nc< nchan; nc++)
    {
    for(int ii2 = 0; ii2<nx; ++ii2)
    {
        for (int jj2 = 0; jj2<ny; ++jj2)
        {
            long lxyz = ii2+nx*jj2;
            double wsum;
            double wnum = 0.;
            wsum = 0;
            //Horizontal convolution
            for(int kk2 = -filtsize; kk2 < filtsize; ++kk2)
            {
                //weighted average
                int iiv = (int) oct_bc<int>(ii2+kk2,nx,bc);
                int lxyz_en = iiv+nx*jj2; 
                wsum = wsum+GK[kk2+filtsize]*image[lxyz_en][nc];
            }
            geosub112[lxyz] = wsum;
            

        }
    }
    for(int ii23 = 0; ii23<nx; ++ii23)
    {
        for (int jj23 = 0; jj23<ny; ++jj23)
        {
            long lxyz = ii23+nx*jj23;
            double wsum = 0;
            //VERTICAL CONVOLUTION
            for(int ll2 = -filtsize; ll2 < filtsize; ++ll2)
            {
                int jjv = (int) oct_bc<int>(jj23+ll2,ny,bc);
                long lxyz_en = ii23+nx*jjv;
                wsum = wsum+GK[ll2+filtsize]*geosub112[lxyz_en];
            }
            image[lxyz][nc] = wsum;
        }
    }
    }//end channel number move


    delete [] geosub112;
    delete [] GK;
}
void oct_gaussian_2d(double * image, int nx, int ny,double sigma)
{
    double * geosub11;
    double **GK;
    int filtsize = 5; // convolution filter size
    bool bc;
    GK = dMatrix(2*filtsize+1,2*filtsize+1);
    geosub11 = new double[nx*ny];

    GK = dMatrix(2*filtsize+1,2*filtsize+1);
    oct_getGaussian(GK,2*filtsize+1,sigma);
    if(GK[0][0] != GK[0][0])
    {
        cout << "Ok, Kernel Failed Here " << sigma << endl;
        exit(0);
    }
    for(int ii2 = 0; ii2<nx; ++ii2)
    {
        for (int jj2 = 0; jj2<ny; ++jj2)
        {
            long lxyz = ii2+nx*jj2;
            //so now we want to perform gauss smoothing
            double wsum = 0.;
            double wnum = 0.;
            for(int kk2 = -filtsize; kk2 < filtsize; ++kk2)
            {
                for(int ll2 = -filtsize; ll2 < filtsize; ++ll2)
                {
                    //weighted average
                    int iiv = (int) oct_bc<int>(ii2+kk2,nx,bc);
                    int jjv = (int) oct_bc<int>(jj2+ll2,ny,bc);
                    int lxyz_en = iiv+nx*jjv;
                    wsum = wsum+GK[kk2+filtsize][ll2+filtsize]*image[lxyz_en];
                }
            }
            //average
            geosub11[lxyz] = wsum;
            if( wsum != wsum)
            {
                for(int kk22 = -filtsize; kk22 < filtsize; ++kk22)
                {
                    for(int ll2 = -filtsize; ll2 < filtsize; ++ll2)
                    {
                        int iiv = (int) oct_bc<int>(ii2+kk22,nx,bc);
                        int jjv = (int) oct_bc<int>(jj2+ll2,ny,bc);
                        int lxyz_en = iiv+nx*jjv;
                        cout <<iiv << " " << jjv << " " << GK[kk22+filtsize][ll2+filtsize]<< " " << image[lxyz_en]<< endl;;
                    }
                }
                cout << "Gaussian smoothing problem, may be an issue with the data, exiting \n";
                exit(0);
            }

        }
    }
    for(int kk2 = 0; kk2 < nx*ny; kk2++)
        image[kk2] = geosub11[kk2];


    delete [] geosub11;
    free_dMatrix(GK,2*filtsize+1);
}
