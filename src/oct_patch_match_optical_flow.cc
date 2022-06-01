#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <cstdio>
#include "image.h"
#include "goesread.h"
#include "util.h"
#include "oct_bc.h"
#include "offlags.h"
using namespace std;
double jsose(double **geo1,double **geo2,int i, int j, int n,int m, int nx, int ny, int rad)
{
    bool bc;
    double sose=0;
    for(int k = 0; k < 2*rad+1; k++)
    {
        for(int l = 0; l < 2*rad+1; l++)
        {
            //i+k-rad will need boundary conditions if they are out of bounds
            //note oct_bc is a boundary condition check, so ic1/jc1,ic2/jc2 will always be in bounds
            int ic1 = (int) oct_bc<int>(i+k-rad,nx,bc);
            int jc1 = (int) oct_bc<int>(j+l-rad,ny,bc);
            int ic2 = (int) oct_bc<int>(i+k+n-rad,nx,bc);
            int jc2 = (int) oct_bc<int>(j+l+m-rad,ny,bc);
            double sos1 = geo2[ic2][jc2]-geo1[ic1][jc1];

            sose += sos1*sos1;
        }
    }

    return sose;
}

double jquad_interp(double y2,double y1,double y3,double x2,double x1, double x3)
{
    double result;
    //Quadratic interpolation function, we are trying to find the x location of the minimum where
    // y = ax ^2 + bx + c
    // solving for where the derivative is 0, or  when
    // 0 = 2 a x + b or
    // -b/2a = x
    double C1 = (y2-y1)/(x2-x1);
    double C2 = (x2*x2-x1*x1)/(x2-x1);
    double a = (y3-C1*x3-y1+C1*x1)/(x3*x3-C2*x3-x1*x1+C2*x1);
    double b = C1 - a * C2;
    //double c = y1-a*pow(x1,2) - b*x1;
    if(a == 0)
    {
        return x2;
    } else
    {
        return -b/(2.*a);
    }
}
void oct_patch_match_optical_flow(float *geo1i,float *geo2i,float *uarr,float *varr, int nx, int ny,OFFlags args)
{
    //inputs: geo1i/2i = image data from the first and second scan
    //        defarr2: Initial guess for where the searcher needs to center the search region
    //        nx: size of x dimension of the image
    //        ny: size of y dimension of the image
    //        args: an object containing the arguments input by the user
    //outputs: uarr/varr, pixel displacements in geo1i dimensions

    double **geo1,**geo2;
    geo1 = dMatrix(nx,ny);
    geo2 = dMatrix(nx,ny);
    int rad = args.rad; //important, box is 2*rad+1 x 2*rad+1
    int srad = args.srad; //Search radius around your initial guess from defarr2
    int SX = 2*srad+1;
    int SY = 2*srad+1;
    int SXD2 = SX/2;
    int SYD2 = SY/2;
    double summin,sumv,sumv1,sumv2;
    int nmin,mmin;
    //Un-raveling for code simplicity
    for(int j = 0; j < ny; j++){
        long jtnx = j*nx;
        for(int i = 0; i < nx; i++)
        {
            long lxyz = i+jtnx;
            geo1[i][j] = geo1i[lxyz];
            geo2[i][j] = geo2i[lxyz];
        }
    }

    for(int j = 0; j < ny; j++)
    {
        long jtnx = j*nx;
        for(int i = 0; i < nx; i++)
        {
            long lxyz = i+jtnx;
            int n = 0;
            int m = 0;
            int dn = 0;
            int dm = -1;
            bool bc;
            int ibc = (int) oct_bc<int>(i+uarr[lxyz],nx,bc);
            int jbc = (int) oct_bc<int>(j+varr[lxyz],ny,bc);

            bool sumcheck = false;
            for(int ic = 0; ic< pow(max(SX,SY),2); ic++)
            {
                if( (-SXD2 < n <= SXD2) && (-SYD2 < m <= SYD2))
                {
                    sumv = jsose(geo1,geo2,ibc,jbc,n,m,nx,ny,rad);
                    if(sumcheck)
                    {
                        if(sumv < summin)
                        {
                            nmin = n; 
                            mmin = m; 
                            summin = sumv;
                        }
                    } else
                    {
                        summin = sumv;
                        nmin = n; 
                        mmin = m; 
                        sumcheck = true;
                    }
                }
                if( (n == m) || ((n < 0) && (n == -m)) || ((n > 0) && (n == 1-m)))
                {
                    int odn = dn;
                    dn = -dm;
                    dm = odn;
                }
                n +=dn;
                m +=dm;
            }

            sumv1 = jsose(geo1,geo2,ibc,jbc,nmin+1,mmin,nx,ny,rad);
            sumv2 = jsose(geo1,geo2,ibc,jbc,nmin-1,mmin,nx,ny,rad);

            if((summin < sumv1) && (summin < sumv2))
            {
                uarr[lxyz] = jquad_interp(summin,sumv1,sumv2,(double) (i+nmin),(double) (i+nmin+1),(double)(i+nmin-1))-(double) i;
            } else {
                uarr[lxyz] = nmin;
            }
            sumv1 = jsose(geo1,geo2,ibc,jbc,nmin,mmin+1,nx,ny,rad);
            sumv2 = jsose(geo1,geo2,ibc,jbc,nmin,mmin-1,nx,ny,rad);
            if((summin < sumv1) && (summin < sumv2))
            {
                varr[lxyz] = jquad_interp(summin,sumv1,sumv2,(double) (j+mmin),(double) (j+mmin+1),(double)(j+mmin-1))- (double) j;
            } else{
                varr[lxyz] = mmin;
            }

            
        }
    }
    free_dMatrix(geo1,nx);
    free_dMatrix(geo2,nx);
}
