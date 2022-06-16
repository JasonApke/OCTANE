#include <iostream>
#include <stdio.h>
#include <math.h>
#include "image.h"
#include "goesread.h"
#include "util.h"
#include "offlags.h"
#include "oct_bc.h"
using namespace std;
double oct_binterp(double,double,double,double,double,double,double,double,double,double);

//A simple optical flow interpolation function following BAKER ET AL. 2011 approach
//Citation: 
//Baker, S., Scharstein, D., Lewis, J.P. et al. A Database and Evaluation Methodology for Optical Flow. 
//Int J Comput Vis 92, 1–31 (2011). https://doi.org/10.1007/s11263-010-0390-2
//Still under development, use with caution J. Apke 2/23/2022
void oct_warpflow(double **u1, double **v1, double **sosarr, double **im1, double **im2,float time,long &holecount, int nx, int ny, double **ut, double **vt)
{
    bool bc, bc2,bc3,bc4;

    for(int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            //use u to seek u
            double iv = round(oct_bc<double>((double)(i+time*u1[i][j]),nx-1,bc));
            double jv = round(oct_bc<double>((double)(i+time*v1[i][j]),ny-1,bc2));
            double iv2 = round(oct_bc<double>((double)(i+u1[i][j]),nx-1,bc3));
            double jv2 = round(oct_bc<double>((double)(i+v1[i][j]),ny-1,bc4));

            for(int k = 0; k < 2; k++)
            {
                for(int l = 0; l < 2; l++)
                {
                    int posi = (int) iv+k;
                    int posj = (int) jv+l;
                    int posi2 = (int) iv2;
                    int posj2 = (int) jv2;
                    double imgdiff = (im1[i][j]-im2[posi2][posj2]); 
                    double imgdiff2 = imgdiff*imgdiff;
                    if((ut[posi][posj] < -998) || (sosarr[posi][posj] > imgdiff2))
                    {
                        if(ut[posi][posj] < -998) holecount -= 1; //reduce the hole count when a point is filled
                        ut[posi][posj] = u1[i][j];
                        vt[posi][posj] = v1[i][j];
                        sosarr[posi][posj] = imgdiff2;
                        //Here is where you splat if needed
                    }

                }
            }
        }
    }
}
int oct_interp (GOESVar &geo1,GOESVar &geo2, float fr, OFFlags args)
{
    int nx = geo1.nav.nx;
    int ny = geo1.nav.ny;
    int ival,jval;
    bool bc;
    double **im1, **im2,**u1,**v1,**ut, **vt, **ut2,**vt2,**sosarr,**sosarr2;
    double **im12,**im22, **im13,**im23;
    double imgnew,imgnew2,imgnew3;
    short * occ;
    long nxtny = nx*ny;
    occ = new short [nxtny];

    im1 = dMatrix(nx,ny);
    im2 = dMatrix(nx,ny);
    if(args.doc2 == 1)
    {
        im12 = dMatrix(nx,ny);
        im22 = dMatrix(nx,ny);
    }
    if(args.doc3 == 1)
    {
        im13 = dMatrix(nx,ny);
        im23 = dMatrix(nx,ny);
    }
    u1 = dMatrix(nx,ny);
    v1 = dMatrix(nx,ny);
    ut = dMatrix(nx,ny);
    vt = dMatrix(nx,ny);
    ut2 = dMatrix(nx,ny);
    vt2 = dMatrix(nx,ny);
    sosarr = dMatrix(nx,ny);
    sosarr2 = dMatrix(nx,ny);


    long holecount = nx*ny, holecount2 = nx*ny;
    for(int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            long lxyz = i+nx*j;
            int i2 = i-20;
            if(i2 < 0) i2 = 0;
            long lxyz2 = i2+nx*j;
            im1[i][j] = geo1.data.data[lxyz];
            im2[i][j] = geo2.data.data[lxyz];
            if(args.doc2 == 1)
            {
                im12[i][j] = geo1.data.data[lxyz+nxtny];
                im22[i][j] = geo2.data.data[lxyz+nxtny];
            }
            if(args.doc3 == 1)
            {
                im13[i][j] = geo1.data.data[lxyz+nxtny+nxtny];
                im23[i][j] = geo2.data.data[lxyz+nxtny+nxtny];
            }
            //Forward flow
            //u1[i][j] = 20.; //geo1.u1[lxyz];
            //v1[i][j] = 0.; //geo1.v1[lxyz];
            //BELOW IS LINEAR INTERPOLATION
            u1[i][j] = 0.; //geo1.u1[lxyz];
            v1[i][j] = 0.; //geo1.v1[lxyz];
            //BELOW IS THE ACTUAL FLOW, USE WHEN READY!!!!
            //u1[i][j] = geo1.u1[lxyz];
            //v1[i][j] = geo1.v1[lxyz];
            ut[i][j] = -999.;
            vt[i][j] = -999.;

            ut2[i][j] = -999.;
            vt2[i][j] = -999.;
            sosarr[i][j] = 999999.;
            sosarr2[i][j] = 999999.;


            
        }
    }

    float frinv = fr;
    geo1.frdt = (float) frinv;
    geo1.tint = geo1.t+(double) geo1.dT*fr;
    float time = frinv;
    //The color constancy test only uses channel1 right now, it will soon involve channels 2/3 as well

    oct_warpflow(u1, v1, sosarr,im1,im2, time,holecount, nx, ny, ut, vt); //warps the flow up to the value of time

    for(int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            //use u to seek u
            double iv = round(i+time*u1[i][j]);
            double jv = round(j+time*v1[i][j]);
            if((iv >= 0) && (iv < nx) && (jv >= 0) && (jv < ny))
            {
                int posi = (int) iv;
                int posj = (int) jv;
                if(ut[posi][posj] < -998)
                {
                    ut[posi][posj] = u1[i][j];
                    vt[posi][posj] = v1[i][j];
                    holecount -= 1; //found one, reduce the holecount
                } else
                {
                    //this is the case of multiple motions for the same pixel
                    double iv2 = round(i + u1[i][j]);
                    double jv2 = round(j + v1[i][j]);
                    int posi2 = (int) iv2;
                    int posj2 = (int) jv2;
                    if((iv2 >= 0) && (iv2 < nx) && (jv2 >= 0) && (jv2 < ny))
                    {
                        double imgdiff = (im1[i][j]-im2[posi2][posj2]);
                        double imgdiff2 = imgdiff*imgdiff;
                        if(sosarr[posi][posj] > (imgdiff*imgdiff))
                        {
                            //passed the color constancy test
                            ut[posi][posj] = u1[i][j];
                            vt[posi][posj] = v1[i][j];
                            sosarr[posi][posj] = imgdiff2;
                            //Baker used splatting here to reduce interpolated flow holes
                        }
                    }

                    
                } 
            }
        }
    }
    //Now that ut and vt have their initial fill, we need to fill in holes
    //I will use an outside-in filling strategy, with forward and backward looping
    int rev = 0;
    while(holecount > 0)
    {
        for(int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                if(rev == 0){
                    ival = i;
                    jval = j;
                } else
                {
                    ival = nx-1-i;
                    jval = ny-1-j;
                }
                if(ut[ival][jval] < -998)
                {
                    //do an average to fill the hole
                    double num1 = 0;
                    double sum1 = 0;
                    double sum2 = 0;
                    int iv1 = ival+1;
                    int jv1 = jval+1;
                    if((iv1 < nx) && (jv1 < ny))
                    {
                        if(ut[iv1][jv1] > -998) 
                        {
                            sum1 += ut[iv1][jv1];
                            sum2 += vt[iv1][jv1];
                            num1 += 1;
                        }
                    }
                    iv1 = ival-1;
                    jv1 = jval+1;
                    if((iv1 >= 0) && (jv1 < ny))
                    {
                        if(ut[iv1][jv1] > -998) 
                        {
                            sum1 += ut[iv1][jv1];
                            sum2 += vt[iv1][jv1];
                            num1 += 1;
                        }
                    }
                    iv1 = ival-1;
                    jv1 = jval-1;
                    if((iv1 >= 0) && (jv1 >= 0))
                    {
                        if(ut[iv1][jv1] > -998) 
                        {
                            sum1 += ut[iv1][jv1];
                            sum2 += vt[iv1][jv1];
                            num1 += 1;
                        }
                    }
                    iv1 = ival+1;
                    jv1 = jval-1;
                    if((iv1 < nx) && (jv1 >= 0))
                    {
                        if(ut[iv1][jv1] > -998) 
                        {
                            sum1 += ut[iv1][jv1];
                            sum2 += vt[iv1][jv1];
                            num1 += 1;
                        }
                    }
                    if(num1 > 0)
                    {
                        ut[ival][jval] = sum1/num1;
                        vt[ival][jval] = sum2/num1;
                        holecount -= 1;
                    }
                }
            }
        } //end i j for loops
        if(rev == 0){
            rev = 1;
        } else
        {
            rev= 0; //recent bug fix here J. Apke 2/23/2022
        }
    } //end while
    //holes should be filled now, hopefully on no more than two iterations of above
    //Now it is time for occlusion reasoning.
    // again, following Baker 2011
    //now check to ensure we have flow consistency
    oct_warpflow(u1, v1, sosarr2,im1,im2, 1.,holecount2, nx, ny, ut2, vt2); //warps the flow up to the value of time
    for(int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            int o1 = 0;
            int o0 = 0;
            //set the occlusion masks, note, this is still under development, masks may be bugged -J. Apke 2/24/2022
            if(ut2[i][j] < -998) o1 = 1; //this means the second image is occluded here
            //if(ut2[i][j] < -998) o0 = 1; //this means the second image is occluded here
            double iv = round(i+u1[i][j]);
            double jv = round(j+v1[i][j]);
            //if((iv >= 0) && (iv < nx) && (jv >=0) && (jv < ny) && (o0==0))
            if((iv >= 0) && (iv < nx) && (jv >=0) && (jv < ny) && o1==0)
            {
                int posi = (int) iv;
                int posj = (int) jv;
                double sqrval1 = u1[i][j] - ut2[posi][posj];
                double sqrval2 = v1[i][j] - vt2[posi][posj];
                if(sqrval1*sqrval1+sqrval2*sqrval2 > 0.25) //note 0.5^2 = 0.25
                {
                    o0 = 1; //this means the first image is occluded here
                    //o1 = 1; //this means the first image is occluded here
                    //printf("Ok heres the Os %d %d %f %f \n",o0,o1, u1[i][j],ut2[posi][posj]);
                }
            }
            //We now have enough information to interpolate, lets do it!
            double x00 = oct_bc<double>((double)(i-time*ut[i][j]),nx-1,bc);
            double y00 = oct_bc<double>((double)(j-time*vt[i][j]),ny-1,bc);
            double x10 = oct_bc<double>((double)(i+(1-time)*ut[i][j]),nx-1,bc);
            double y10 = oct_bc<double>((double)(j+(1-time)*vt[i][j]),ny-1,bc);
            double x1 = (double) ((int) x00);
            double x2 = x1+1;
            double y1 = (double) ((int) y00);
            double y2 = y1+1;
            double f11 = im1[(int) x1][(int) y1];
            double f21 = im1[(int) x2][(int) y1];
            double f12 = im1[(int) x1][(int) y2];
            double f22 = im1[(int) x2][(int) y2];
            //bilinear interpolation of boundary condition corrected points
            double I0X0=oct_binterp (x00, y00,x1, x2, y1, y2, f11, f21, f12, f22);
            double I0X02, I0X03;
            if(args.doc2 == 1)
            {
                f11 = im12[(int) x1][(int) y1];
                f21 = im12[(int) x2][(int) y1];
                f12 = im12[(int) x1][(int) y2];
                f22 = im12[(int) x2][(int) y2];
                I0X02=oct_binterp (x00, y00,x1, x2, y1, y2, f11, f21, f12, f22);
            }
            if(args.doc3 == 1)
            {
                f11 = im13[(int) x1][(int) y1];
                f21 = im13[(int) x2][(int) y1];
                f12 = im13[(int) x1][(int) y2];
                f22 = im13[(int) x2][(int) y2];
                I0X03=oct_binterp (x00, y00,x1, x2, y1, y2, f11, f21, f12, f22);
            }

            
            x1 = (double) ((int) x10);
            x2 = x1+1;
            y1 = (double) ((int) y10);
            y2 = y1+1;
            f11 = im2[(int) x1][(int) y1];
            f21 = im2[(int) x2][(int) y1];
            f12 = im2[(int) x1][(int) y2];
            f22 = im2[(int) x2][(int) y2];


            double I1X1=oct_binterp (x10, y10,x1, x2, y1, y2, f11, f21, f12, f22);
            double I1X12, I1X13;
            if(args.doc2 == 1)
            {
                f11 = im22[(int) x1][(int) y1];
                f21 = im22[(int) x2][(int) y1];
                f12 = im22[(int) x1][(int) y2];
                f22 = im22[(int) x2][(int) y2];
                I1X12=oct_binterp (x10, y10,x1, x2, y1, y2, f11, f21, f12, f22);
            }
            if(args.doc3 == 1)
            {
                f11 = im23[(int) x1][(int) y1];
                f21 = im23[(int) x2][(int) y1];
                f12 = im23[(int) x1][(int) y2];
                f22 = im23[(int) x2][(int) y2];
                I1X13=oct_binterp (x10, y10,x1, x2, y1, y2, f11, f21, f12, f22);
            }


            //Uncomment below to turn off occlusion masks
            o0 = 0; o1 = 0;

            if((o0 == 0) && (o1 == 0))
            {
                imgnew = (1.-time)*I0X0 + time*(I1X1);
               if(args.doc2 == 1)
               {
                   imgnew2 = (1.-time)*I0X02 + time*(I1X12);
               }
               if(args.doc3 == 1)
               {
                   imgnew3 = (1.-time)*I0X03 + time*(I1X13);
               } 
            }
            if(o0 == 1)
            {
                imgnew = I1X1;
                if(args.doc2 == 1) imgnew2 = I1X12;
                if(args.doc3 == 1) imgnew3 = I1X13;
            }
            if(o1 == 1)
            {
                imgnew = I0X0;
                if(args.doc2 == 1) imgnew2 = I0X02;
                if(args.doc3 == 1) imgnew3 = I0X03;
            }
            if((o0 == 1) && (o1 == 1))
            {
                //Both occluded seems to happen which is odd, but I am just going to blend when this happens
                imgnew = (1.-time)*I0X0 + time*(I1X1);   
               if(args.doc2 == 1)
               {
                   imgnew2 = (1.-time)*I0X02 + time*(I1X12);
               }
               if(args.doc3 == 1)
               {
                   imgnew3 = (1.-time)*I0X03 + time*(I1X13);
               } 
                printf("Something failed %d %d \n", i,j);
                exit(1);
            }

            //set the datasval now to the interpolated value, FOR NOW THIS IS HOW WE WILL PASS IT BACK
            //AN ALTERNATIVE IS TO CALL GOESWRITE HERE AND OUTPUT SEVERAL NETCDF FILES
            //goesData.dataSVal[lxyz] = (short)((geosub11[lxyz]-goesData.radOffset)/goesData.radScale);
            //
            long lxyz = i+nx*j;
            if(args.dopolar == 0)
            {
                geo1.dataSVal[lxyz] = (short)((imgnew - geo1.nav.radOffset)/(geo1.nav.radScale));
                if(args.doc2 == 1) geo1.dataSVal2[lxyz] = (short) ((imgnew2 - geo1.nav.radOffset2)/(geo1.nav.radScale2));
                if(args.doc3 == 1) geo1.dataSVal3[lxyz] = (short) ((imgnew3 - geo1.nav.radOffset3)/(geo1.nav.radScale3));
            } else {
                geo1.dataSValfloat[lxyz] = (float) imgnew;
                if(args.doc2 == 1) geo1.dataSValfloat2[lxyz] = (float) imgnew2;
                if(args.doc3 == 1) geo1.dataSValfloat3[lxyz] = (float) imgnew3;

            }
            occ[lxyz] = 0;
            if(o0 == 1)
            {
                occ[lxyz] = 1;
            }
            if(o1 == 1)
            {
                occ[lxyz] = 2;
            }


        }
    }

        

    free_dMatrix(im1,nx);
    free_dMatrix(im2,nx);
    free_dMatrix(u1,nx);
    free_dMatrix(v1,nx);
    free_dMatrix(ut,nx);
    free_dMatrix(vt,nx);
    free_dMatrix(ut2,nx);
    free_dMatrix(vt2,nx);
    free_dMatrix(sosarr,nx);
    free_dMatrix(sosarr2,nx);
    geo1.occlusion = occ;

    return 1;
}
