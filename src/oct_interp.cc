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
void oct_warpflow(float *u1, float *v1, float *sosarr, float *im1, float *im2,float time,long &holecount, int nx, int ny, float *ut, float *vt)
{
    bool bc, bc2,bc3,bc4;
    long nxtny = nx*ny;

    for (int j = 0; j < ny; j++)
    {
        long nxtj = nx*j;
        for(int i = 0; i < nx; i++)
        {
            long lxyz = i+nxtj; // 
            //use u to seek u
            int iv = (int)  oct_bc<double>((double)(round(i+time*u1[lxyz])),nx-1,bc);
            int jv = (int)  oct_bc<double>((double)(round(j+time*v1[lxyz])),ny-1,bc2);
            int iv2 = (int) oct_bc<double>((double)(round(i+u1[lxyz])),nx-1,bc3);
            int jv2 = (int) oct_bc<double>((double)(round(j+v1[lxyz])),ny-1,bc4);

            for(int l = 0; l < 2; l++)
            {
                int posj = jv+l;
                int posj2 = jv2 + l;
                long nxtposj = nx*posj;
                long nxtposj2 = nx*posj2;
                for(int k = 0; k < 2; k++)
                {
                    int posi = (int) iv+k;
                    long lxyz2 = posi+nxtposj; //lxyz at posi posj, or iv+k, jv+l
                    long lxyz3 = iv2+k + nxtposj2; //lxyz at posi2, posj2

                    //int posi2 = (int) iv2;
                    //int posj2 = (int) jv2;
                    double imgdiff = (im1[lxyz]-im2[lxyz3]); 
                    double imgdiff2 = imgdiff*imgdiff;
                    if((ut[lxyz2] < -998) || (sosarr[lxyz2] > imgdiff2))
                    {
                        if(ut[lxyz2] < -998) holecount -= 1; //reduce the hole count when a point is filled
                        ut[lxyz2] = u1[lxyz];
                        vt[lxyz2] = v1[lxyz];
                        sosarr[lxyz2] = imgdiff2;
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
    bool bc,bc3,bc4;
    float *u1,*v1,*ut, *vt, *ut2,*vt2,*sosarr,*sosarr2;
    double imgnew,imgnew2,imgnew3;
    short * occ, *o1a, *o0a;
    long nxtny = nx*ny;
    occ = new short [nxtny];

    //note, I changed 2d arrays to 1d as C++ is slower w/ 2d

    //im1 = dMatrix(nx,ny);
    //im2 = dMatrix(nx,ny);
    //if(args.doc2 == 1)
    //{
    //    im12 = dMatrix(nx,ny);
    //    im22 = dMatrix(nx,ny);
    //}
    //if(args.doc3 == 1)
    //{
    //    im13 = dMatrix(nx,ny);
    //    im23 = dMatrix(nx,ny);
    //}
    //u1 = dMatrix(nx,ny);
    //v1 = dMatrix(nx,ny);
    ut = new float [nxtny]; //dMatrix(nx,ny);
    vt = new float [nxtny]; //dMatrix(nx,ny);
    ut2 = new float [nxtny]; //dMatrix(nx,ny);
    vt2 = new float [nxtny]; //dMatrix(nx,ny);
    sosarr = new float [nxtny]; //dMatrix(nx,ny);
    sosarr2 = new float [nxtny]; //dMatrix(nx,ny);
    o1a = new short [nxtny];
    o0a = new short [nxtny];


    long holecount = nx*ny, holecount2 = nx*ny;
    for (int j = 0; j < ny; j++)
    {
        long nxtj = nx*j;
        for(int i = 0; i < nx; i++)
        {
            long lxyz = i+nx*j;
            //Forward flow
            //u1[i][j] = 20.; //geo1.u1[lxyz];
            //v1[i][j] = 0.; //geo1.v1[lxyz];
            //BELOW IS LINEAR INTERPOLATION
            //u1[i][j] = 0.; //geo1.u1[lxyz];
            //v1[i][j] = 0.; //geo1.v1[lxyz];
            //BELOW IS THE ACTUAL FLOW, USE WHEN READY!!!!
            //u1[i][j] = geo1.u1[lxyz];
            //v1[i][j] = geo1.v1[lxyz];
            ut[lxyz] = -999.;
            vt[lxyz] = -999.;

            ut2[lxyz] = -999.;
            vt2[lxyz] = -999.;
            sosarr[lxyz] = 999999.;
            sosarr2[lxyz] = 999999.;


            
        }
    }

    float frinv = fr;
    geo1.frdt = (float) frinv;
    geo1.tint = geo1.t+(double) geo1.dT*fr;
    float time = frinv;
    //The color constancy test only uses channel1 right now, it will soon involve channels 2/3 as well
    oct_warpflow(geo1.uPix, geo1.vPix, sosarr,geo1.data.data,geo2.data.data, time,holecount, nx, ny, ut, vt); //warps the flow up to the value of time at interpolate
    //This is superfluous to above...
    //for(int i = 0; i < nx; i++)
    //{
    //    for (int j = 0; j < ny; j++)
    //    {
    //        //use u to seek u
    //        double iv = round(i+time*u1[i][j]);
    //        double jv = round(j+time*v1[i][j]);
    //        if((iv >= 0) && (iv < nx) && (jv >= 0) && (jv < ny))
    //        {
    //            int posi = (int) iv;
    //            int posj = (int) jv;
    //            if(ut[posi][posj] < -998)
    //            {
    //                ut[posi][posj] = u1[i][j];
    //                vt[posi][posj] = v1[i][j];
    //                holecount -= 1; //found one, reduce the holecount
    //            } else
    //            {
    //                //this is the case of multiple motions for the same pixel
    //                double iv2 = round(i + u1[i][j]);
    //                double jv2 = round(j + v1[i][j]);
    //                int posi2 = (int) iv2;
    //                int posj2 = (int) jv2;
    //                if((iv2 >= 0) && (iv2 < nx) && (jv2 >= 0) && (jv2 < ny))
    //                {
    //                    double imgdiff = (im1[i][j]-im2[posi2][posj2]);
    //                    double imgdiff2 = imgdiff*imgdiff;
    //                    if(sosarr[posi][posj] > (imgdiff*imgdiff))
    //                    {
    //                        //passed the color constancy test
    //                        ut[posi][posj] = u1[i][j];
    //                        vt[posi][posj] = v1[i][j];
    //                        sosarr[posi][posj] = imgdiff2;
    //                        //Baker used splatting here to reduce interpolated flow holes
    //                    }
    //                }

    //                
    //            } 
    //        }
    //    }
    //}
    //Now that ut and vt have their initial fill, we need to fill in holes
    //I will use an outside-in filling strategy, with forward and backward looping
    int rev = 0;
    while(holecount > 0)
    {
        for (int j = 0; j < ny; j++)
        {
            if(rev == 0) {
                jval = j;
            } else
            {
                jval = ny-1-j;
            }
            int kmax = nx+nx;
            int kmin = -nx;
            int lmin = -1;
            int lmax = 2;
            if(j == 0) kmin = 0;
            if(j == ny-1) kmax = nx;
            long nxtjval = nx*jval;
            for(int i = 0; i < nx; i++)
            {
                if(i == 0) lmin = 0;
                if(i == nx-1) lmax = 1;
                if(rev == 0){
                    ival = i;
                } else
                {
                    ival = nx-1-i;
                }
                long lxyzval = ival + nxtjval;
                if(ut[lxyzval] < -998)
                {
                    //do an average to fill the hole
                    double num1 = 0;
                    double sum1 = 0;
                    double sum2 = 0;
                    //streamline this to for loop
                    for(int k = kmin; k < kmax; k=k+nx)
                    {
                        for(int l = lmin; l < lmax; l++)
                        {
                            int iv1 = ival + k;
                            int jv1 = jval + l;
                            long lxyzval1 = lxyzval + k + l;
                            if(ut[lxyzval1] > -998) 
                            {
                                sum1 += ut[lxyzval1];
                                sum2 += vt[lxyzval1];
                                num1 += 1;
                            }
                        }

                    }
                    
                    if(num1 > 0)
                    {
                        ut[lxyzval] = sum1/num1;
                        vt[lxyzval] = sum2/num1;
                        holecount -= 1;
                    }
                }
            }
        } //end i j for loops
        if(rev == 0){
            rev = 1;
        } else
        {
            rev= 0; 
        }
    } //end while
    //holes will be filled now
    //Now it is time for occlusion reasoning.
    // again, following Baker 2011
    //now check to ensure we have flow consistency
    oct_warpflow(geo1.uPix, geo1.vPix, sosarr2,geo1.data.data,geo2.data.data, 1.,holecount2, nx, ny, ut2, vt2); //warps the flow up to the value of time at image 2
    //occlusion masks are set here
    for(int j = 0; j < ny; j++)
    {
        long nxtj = nx*j;
        for(int i = 0; i < nx; i++)
        {
            long lxyz = i+nxtj;
            o1a[lxyz] = 0;
            o0a[lxyz] = 0;
            if(ut2[lxyz] < -998)
            {
                o1a[lxyz] = 1;
            } else
            {
                int iv = (int) oct_bc<double>((double)(round(i+geo1.uPix[lxyz])),nx-1,bc3);
                int jv = (int) oct_bc<double>((double)(round(j+geo1.vPix[lxyz])),ny-1,bc4);
                long lxyz2 = iv+jv*nx;

                double sqrval1 = geo1.uPix[lxyz] - ut2[lxyz2];
                double sqrval2 = geo1.vPix[lxyz] - vt2[lxyz2];
                if(sqrval1*sqrval1+sqrval2*sqrval2 > 0.25) //note 0.5^2 = 0.25
                {
                    o0a[lxyz] = 1;
                }
            }
            
            
        }
    }

    for (int j = 0; j < ny; j++)
    {
        long nxtj = j*nx;
        for(int i = 0; i < nx; i++)
        {
            long lxyz = i + nxtj;
            //Below was incorrect, I have fixed with the nested for loop above filling o0 and o1
            //int o1 = 0;
            //int o0 = 0;
            ////set the occlusion masks, note, this is still under development, masks may be bugged -J. Apke 2/24/2022
            //if(ut2[i][j] < -998) o1 = 1; //this means the second image is occluded here
            ////if(ut2[i][j] < -998) o0 = 1; //this means the second image is occluded here
            //double iv = round(i+u1[i][j]);
            //double jv = round(j+v1[i][j]);
            ////if((iv >= 0) && (iv < nx) && (jv >=0) && (jv < ny) && (o0==0))
            //if((iv >= 0) && (iv < nx) && (jv >=0) && (jv < ny) && o1==0)
            //{
            //    int posi = (int) iv;
            //    int posj = (int) jv;
            //    double sqrval1 = u1[i][j] - ut2[posi][posj];
            //    double sqrval2 = v1[i][j] - vt2[posi][posj];
            //    if(sqrval1*sqrval1+sqrval2*sqrval2 > 0.25) //note 0.5^2 = 0.25
            //    {
            //        o0 = 1; //this means the first image is occluded here
            //        //o1 = 1; //this means the first image is occluded here
            //        //printf("Ok heres the Os %d %d %f %f \n",o0,o1, u1[i][j],ut2[posi][posj]);
            //    }
            //}

            //We now have enough information to interpolate, lets do it!
            double x00 = oct_bc<double>((double)(i-time*ut[lxyz]),nx-1,bc);
            double y00 = oct_bc<double>((double)(j-time*vt[lxyz]),ny-1,bc);
            double x10 = oct_bc<double>((double)(i+(1-time)*ut[lxyz]),nx-1,bc);
            double y10 = oct_bc<double>((double)(j+(1-time)*vt[lxyz]),ny-1,bc);
            int x0i = (int) (x00+0.5);
            int y0i = (int) (y00+0.5); //location of nearest pix at x0, y0
            long lxyz00 = x0i + nx*y0i;
            int x1i = (int) (x10+0.5);
            int y1i = (int) (y10+0.5); //location of nearest pix at x1, y1
            long lxyz11 = x1i + nx*y1i;

            double x1 = (double) ((int) x00);
            double x2 = x1+1;
            double y1 = (double) ((int) y00);
            double y2 = y1+1;
            long lxyz1 = ((int) x1) + nx*((int) y1);
            long lxyz2 = ((int) x2) + nx*((int) y1);
            long lxyz3 = ((int) x1) + nx*((int) y2);
            long lxyz4 = ((int) x2) + nx*((int) y2);
            double f11 = geo1.data.data[lxyz1]; //im1[(int) x1][(int) y1];
            double f21 = geo1.data.data[lxyz2]; //im1[(int) x2][(int) y1];
            double f12 = geo1.data.data[lxyz3]; //im1[(int) x1][(int) y2];
            double f22 = geo1.data.data[lxyz4]; //im1[(int) x2][(int) y2];
            //bilinear interpolation of boundary condition corrected points
            double I0X0=oct_binterp (x00, y00,x1, x2, y1, y2, f11, f21, f12, f22);
            double I0X02, I0X03;
            if(args.doc2 == 1)
            {
                f11 = geo1.data.data[lxyz1+nxtny]; //im12[(int) x1][(int) y1];
                f21 = geo1.data.data[lxyz2+nxtny]; //im12[(int) x2][(int) y1];
                f12 = geo1.data.data[lxyz3+nxtny]; //im12[(int) x1][(int) y2];
                f22 = geo1.data.data[lxyz4+nxtny]; //im12[(int) x2][(int) y2];
                I0X02=oct_binterp (x00, y00,x1, x2, y1, y2, f11, f21, f12, f22);
            }
            if(args.doc3 == 1)
            {
                f11 = geo1.data.data[lxyz1+nxtny+nxtny]; //im13[(int) x1][(int) y1];
                f21 = geo1.data.data[lxyz2+nxtny+nxtny]; //im13[(int) x2][(int) y1];
                f12 = geo1.data.data[lxyz3+nxtny+nxtny]; //im13[(int) x1][(int) y2];
                f22 = geo1.data.data[lxyz4+nxtny+nxtny]; //im13[(int) x2][(int) y2];
                I0X03=oct_binterp (x00, y00,x1, x2, y1, y2, f11, f21, f12, f22);
            }

            
            x1 = (double) ((int) x10);
            x2 = x1+1;
            y1 = (double) ((int) y10);
            y2 = y1+1;
            lxyz1 = ((int) x1) + nx*((int) y1);
            lxyz2 = ((int) x2) + nx*((int) y1);
            lxyz3 = ((int) x1) + nx*((int) y2);
            lxyz4 = ((int) x2) + nx*((int) y2);
            f11 = geo2.data.data[lxyz1]; //im1[(int) x1][(int) y1];
            f21 = geo2.data.data[lxyz2]; //im1[(int) x2][(int) y1];
            f12 = geo2.data.data[lxyz3]; //im1[(int) x1][(int) y2];
            f22 = geo2.data.data[lxyz4]; //im1[(int) x2][(int) y2];


            double I1X1=oct_binterp (x10, y10,x1, x2, y1, y2, f11, f21, f12, f22);
            double I1X12, I1X13;
            if(args.doc2 == 1)
            {
                f11 =  geo2.data.data[lxyz1+nxtny]; //im22[(int) x1][(int) y1];
                f21 =  geo2.data.data[lxyz2+nxtny]; //im22[(int) x2][(int) y1];
                f12 =  geo2.data.data[lxyz3+nxtny]; //im22[(int) x1][(int) y2];
                f22 =  geo2.data.data[lxyz4+nxtny]; //im22[(int) x2][(int) y2];
                I1X12=oct_binterp (x10, y10,x1, x2, y1, y2, f11, f21, f12, f22);
            }
            if(args.doc3 == 1)
            {
                f11 =  geo2.data.data[lxyz1+nxtny+nxtny]; //im23[(int) x1][(int) y1];
                f21 =  geo2.data.data[lxyz2+nxtny+nxtny]; //im23[(int) x2][(int) y1];
                f12 =  geo2.data.data[lxyz3+nxtny+nxtny]; //im23[(int) x1][(int) y2];
                f22 =  geo2.data.data[lxyz4+nxtny+nxtny]; //im23[(int) x2][(int) y2];
                I1X13=oct_binterp (x10, y10,x1, x2, y1, y2, f11, f21, f12, f22);
            }


            //Uncomment below to turn off occlusion masks
            //o0 = 0; o1 = 0;
            short o0 = o0a[lxyz00];
            short o1 = o1a[lxyz11];
            occ[lxyz] = 0;

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
            } else if(o1 == 1)
            {
                occ[lxyz] = 2;
                imgnew = I0X0;
                if(args.doc2 == 1) imgnew2 = I0X02;
                if(args.doc3 == 1) imgnew3 = I0X03;
            } else{
                imgnew = I1X1;
                occ[lxyz] = 1;
                if(args.doc2 == 1) imgnew2 = I1X12;
                if(args.doc3 == 1) imgnew3 = I1X13;         
            }

            //set the datasval now to the interpolated value
            //long lxyz = i+nx*j;
            if(args.dopolar == 0)
            {
                //geo1.dataSVal[lxyz] = (short)((imgnew));
                //Scale the image back to native values
                //Note this currently assumes minout and maxout from fileread are 0 - 255, must add to args
                float imgscale = (imgnew/255.) * (args.NormMax-args.NormMin)+args.NormMin;
                geo1.dataSVal[lxyz] = (short)((imgscale - geo1.nav.radOffset)/(geo1.nav.radScale));
                if(args.doc2 == 1)
                {
                    imgscale = (imgnew2/255.) * (args.NormMax2-args.NormMin2)+args.NormMin2;
                    geo1.dataSVal2[lxyz] = (short) ((imgscale - geo1.nav.radOffset2)/(geo1.nav.radScale2));
                }
                if(args.doc3 == 1)
                {
                    imgscale = (imgnew3/255.) * (args.NormMax3-args.NormMin3)+args.NormMin3;
                    geo1.dataSVal3[lxyz] = (short) ((imgnew3 - geo1.nav.radOffset3)/(geo1.nav.radScale3));
                }
            } else {
                float imgscale = (imgnew/255.) * (args.NormMax-args.NormMin)+args.NormMin;
                geo1.dataSValfloat[lxyz] = (float) imgscale;
                if(args.doc2 == 1)
                {
                    imgscale = (imgnew2/255.) * (args.NormMax2-args.NormMin2)+args.NormMin2;
                    geo1.dataSValfloat2[lxyz] = (float) imgscale;
                }
                if(args.doc3 == 1)
                {
                    imgscale = (imgnew3/255.) * (args.NormMax3-args.NormMin3)+args.NormMin3;
                    geo1.dataSValfloat3[lxyz] = (float) imgscale;
                }

            }
        }
    }

        

    //free_dMatrix(im1,nx);
    //free_dMatrix(im2,nx);
    //free_dMatrix(u1,nx);
    //free_dMatrix(v1,nx);
    //free_dMatrix(ut,nx);
    //free_dMatrix(vt,nx);
    //free_dMatrix(ut2,nx);
    //free_dMatrix(vt2,nx);
    //free_dMatrix(sosarr,nx);
    //free_dMatrix(sosarr2,nx);
    delete [] ut;
    delete [] vt;
    delete [] ut2;
    delete [] vt2;
    delete [] sosarr;
    delete [] sosarr2;
    delete [] o1a;
    delete [] o0a;
    geo1.occlusion = occ;

    return 1;
}
