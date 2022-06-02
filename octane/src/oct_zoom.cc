#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "oct_bicubic.h"
#include "oct_gaussian.h"
#include "image.h"
using namespace std;
//Purpose: These are a collection of zoom in/out functions designed for scaling the datasets
//to the same resolutions where needed

void oct_zoom_size(int nx, int ny, int &nxx, int &nyy, double factor)
{
    nxx = (int)((double)nx* factor + 0.5);
    nyy = (int)((double)ny* factor + 0.5);
}
void oct_zoom_out(double * image, double * imageout, int nx, int ny, double factor,int verb)
{
    //define temp image for smoothing
    int inout = 2;
    double *Is;

    Is = new double [nx*ny];
    for (int i = 0; i < nx*ny; i++){
            Is[i] = image[i];
    }

    int nxx, nyy;
    oct_zoom_size(nx, ny, nxx,nyy,factor);
    //Smooth the image first
    const double sigma = 0.6*sqrt(1.0/(factor*factor)-1.0); 
    oct_gaussian(Is, nx, ny, sigma);
    //now interpolate
    for(int jj = 0; jj < nyy; jj++)
    {
        long nxxtjj = nxx*jj;
        for(int ii = 0; ii < nxx; ii++)
        {
            const double i2  = (double) ii / factor;
            const double j2  = (double) jj / factor;
            double g = oct_bicubic(Is,i2,j2, nx, ny,inout);
            if(verb == 0){
                imageout[ii+nxxtjj] = g;
            }else{
                imageout[ii+nxxtjj] = g*factor; //This is for the flow, the flow needs to be scaled down
            }
        }
    }
    delete [] Is;
}
void oct_zoom_out_float(float * image, float * imageout, int nx, int ny, double factor,int verb,int cnum)
{
    int inout = 2;
    double *Is;
    double g;

    Is = new double [nx*ny];
    for (int i = 0; i < nx*ny; i++){
            Is[i] = image[i];
    }
    if(verb == 1) exit(0);

    int nxx, nyy;
    oct_zoom_size(nx, ny, nxx,nyy,factor);
    long cnumt = cnum*nxx*nyy;
    //Smooth the image first
    const double sigma = 0.6*sqrt(1.0/(factor*factor)-1.0); 
    oct_gaussian(Is, nx, ny, sigma);
    //now interpolate
    for(int jj = 0; jj < nyy; jj++)
    {
        long nxxtjj = nxx*jj;
        for(int ii = 0; ii < nxx; ii++)
        {
            const double i2  = (double) ii / factor;
            const double j2  = (double) jj / factor;
            if(factor < 0.999999)
            {
                g = oct_bicubic(Is,i2,j2, nx, ny,inout);
            } else
            {
                g = image[ii+nxxtjj];
            }
            imageout[ii+nxxtjj+cnum] = g;
        }
    }
    delete [] Is;
}
void oct_zoom_out_2d(double * image, double ** imageout, int nx, int ny, double factor,int verb)
{
    //define temp image for smoothing
    int inout = 2;
    double *Is;

    Is = new double [nx*ny];
    for (int i = 0; i < nx*ny; i++){
            Is[i] = image[i];
    }
    if(verb == 1) exit(0);

    int nxx, nyy;
    oct_zoom_size(nx, ny, nxx,nyy,factor);
    //Smooth the image first
    const double sigma = 0.6*sqrt(1.0/(factor*factor)-1.0); 
    oct_gaussian(Is, nx, ny, sigma);
    //now interpolate
    for(int ii = 0; ii < nxx; ii++)
    {
        for(int jj = 0; jj < nyy; jj++)
        {
            const double i2  = (double) ii / factor;
            const double j2  = (double) jj / factor;
            double g = oct_bicubic(Is,i2,j2, nx, ny,inout);
            imageout[ii][jj] = g;
        }
    }
    delete [] Is;
}
//this is an image zoom out
void oct_zoom_out_image(double **image, Image &imageout, int nx, int ny, double factor,int verb)
{
    //define temp image for smoothing
    int inout = 2;
    double *Is;
    
    Is = new double [nx*ny];
    for (int c=0; c<imageout.nchannels; c++)
    {
    for (int i = 0; i < nx*ny; i++){
            Is[i] = image[i][c];
    }

    int nxx, nyy;
    oct_zoom_size(nx, ny, nxx,nyy,factor);
        long nxxtnyytc = nxx*nyy*c;
    //Smooth the image first
    const double sigma = 0.6*sqrt(1.0/(factor*factor)-1.0); 
    oct_gaussian(Is, nx, ny, sigma);
    //now interpolate
    for(int jj = 0; jj < nyy; jj++)
    {
        long nxxtjj = nxx*jj;
        for(int ii = 0; ii < nxx; ii++)
        {
            const double i2  = (double) ii / factor;
            const double j2  = (double) jj / factor;
            double g = oct_bicubic(Is,i2,j2, nx, ny,inout);
            imageout.data[ii+nxxtjj+nxxtnyytc] = g;
        }//endfor jj
    }//endfor ii
    } //endfor c iterations
    delete [] Is;
}
void oct_zoom_in(double * flow, double * flowout,int nx,int ny,int nxx,int nyy)
{
    // compute the zoom factor
    int inout = 1;
    const double factorx = ((double)nxx / nx);
    const double factory = ((double)nyy / ny);

    // re-sample the image using bicubic interpolation
    //Option for parallel processing here
    for (int jj1 = 0; jj1 < nyy; jj1++)
    {
        double j2 =  (double) jj1 / factory;
        int nxxtjj1 = nxx*jj1;
        for (int i1 = 0; i1 < nxx; i1++)
        {
            double i2 =  (double) i1 / factorx;

            //double g = bicubic_interpolation_at(I, j2, i2, nx, ny, false);
            double g = oct_bicubic(flow, i2, j2, nx, ny,inout);
            //Iout[i1 * nxx + jj1] = g;
            flowout[i1+nxxtjj1] = g;
        }
    }

}
//I use this to zoom out the calibrated radiance values
void oct_zoom_in_float(float * flow, float * flowout,int nx,int ny,int nxx,int nyy,int cnum,int interp)
{
    // compute the zoom factor
    int inout = 1;
    float g;
    const float factorx = ((float)nxx / nx);
    const float factory = ((float)nyy / ny);
    const float val1 = (0.5-0.5/factory);
    const float val2 = (0.5-0.5/factorx);
    long cnumt = cnum*(nxx*nyy);
    int j3;
    long nxtj3;


    // re-sample the image using bicubic interpolation
    //Option for parallel processing here
    for (int jj1 = 0; jj1 < nyy; jj1++)
    {
        //double j2 =  (double) jj1 / factory;
        float j2 =  (float) ((jj1 / factory)-val1);
        if(interp < 1)
        {
            j3 = int(j2+0.5);
            nxtj3 = nx*j3;
        }
        int nxxtjj1 = nxx*jj1;
        for (int i1 = 0; i1 < nxx; i1++)
        {
            float i2 =  (float) ((i1 / factorx)-val2);

            if(interp == 1)
            {
                g = oct_bicubic_float(flow, i2, j2, nx, ny,inout);
            } else
            {
                int i3 = int(i2+0.5);
                g = flow[i3+nxtj3];
            }
            flowout[i1+nxxtjj1+cnumt] = g;
        }
    }

}
