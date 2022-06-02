#include <iostream>
#include <netcdf>
#include <math.h>
#include "image.h"
#include "goesread.h"
#include "offlags.h"

using namespace std;

//Optical flow approach options
void oct_patch_match_optical_flow (float *, float *, float *, float *, int, int,OFFlags);
void oct_variational_optical_flow (Image, Image, float *, float *,float *, int, int,int,OFFlags);

//Post-processing functions
void oct_pix2uv_cuda(GOESVar&,double,float *,float *,short *, short *,short *, short *,OFFlags);
void oct_uv2pix(GOESVar&,float *, float *,double,OFFlags);
void oct_srsal_cu (float *, float *, float *, int, int,OFFlags);


//This is the optical flow code wrapper, reads in just the two GOESVar image files and the arguments
int oct_optical_flow (GOESVar &goesData,GOESVar &goesData2,OFFlags &args)
{
    int nx, ny,i,j;
    long lxyz;
    //Below are the short arrays to store the datasets on the output netcdf file
    short *ur, *vr,*ur2,*vr2,*CTP;
    //Below is the float-precision return from the optical flow algorithms
    nx = goesData.nav.nx;
    ny = goesData.nav.ny;
    int nxtny = nx*ny;
    // array allocations
    ur = new short [nxtny];
    vr = new short [nxtny];
    ur2 = new short [nxtny];
    vr2 = new short [nxtny];

    //Preprocessing steps here, determine the value of the first guess fed into u/vPix
    if(args.dofirstguess == 0)
    {
        //this is otherwise delcared in firstguess read where it is filled
        goesData.uPix = new float [nxtny]; //float precision pixel displacements
        goesData.vPix = new float [nxtny];
        // fill it with 0s
        for(int fi = 0; fi<nxtny; fi++)
        {
                goesData.uPix[fi] = 0.;
                goesData.vPix[fi] = 0.;
        }
    } else
    {
        //The first guess files are assumed to be navigated winds, this needs to be shut off for raw arrays
        oct_uv2pix(goesData,goesData.uPix,goesData.vPix,goesData2.t,args);
    }
    //Run the optical flow algorithms now
    int nc = 1+args.doc2+args.doc3;
    //Initial octane release has two primary optical flow algorithms, patch matching and variational
    //with more planned in the future -J. Apke 2/23/2022
    if(args.dososm == 1)
    {
        if(args.doc2 == 1 || args.doc3 == 1)
        {
            printf("Multichannel not yet supported on Patch Matching/Sum-of-Squared-error minimization, exiting\n");
            exit(0);
        }
        oct_patch_match_optical_flow(goesData.data.data,goesData2.data.data,goesData.uPix,goesData.vPix,nx,ny,args);
    } else{
        oct_variational_optical_flow(goesData.data,goesData2.data,goesData.CTHVal,goesData.uPix,goesData.vPix,nx,ny,nc,args);
    }
    //I do below after the optical flow method in case those methods change the height assignment value, or if there is
    //post processing height assignment procedures added here -J. Apke 2/23/2022
    if(args.doCTH == 1)
    {
        CTP = new short [nxtny];
        for (int jj = 0; jj<ny; ++jj)
        {
            long nxtjj = nx*jj;
            for(int ii = 0; ii<nx; ++ii)
            {
                lxyz = ii+nxtjj;
                if(args.ir==1)
                {
                    CTP[lxyz] = (short) ((goesData.CTHVal[lxyz]-300)*100); 
                } else{
                    CTP[lxyz] = (short) goesData.CTHVal[lxyz]; 
                }
            }
        }
    }

    //This segment navigates pixel displacements to u/v in m/s    
    oct_pix2uv_cuda(goesData,goesData2.t,goesData.uPix,goesData.vPix,ur, vr,ur2,vr2,args);
    


    goesData.uVal = ur; //Navigated short-precision flow
    goesData.vVal = vr;
    goesData.uVal2 = ur2; //Un-navigated pixel displacements
    goesData.vVal2 = vr2;
    //Possible flow-smoothing post processing (used for 2021 paper)
    if(args.dosrsal == 1)
    {
        cout << "Beginning anisotropic smoothing\n";
        oct_srsal_cu(goesData.uPix,goesData.vPix,goesData.CTHVal,nx,ny,args);
        cout << "Finished\n";
    }

    if(args.doCTH==1) goesData.CTP = CTP; //short storage for the motion vector height

    return 1;

}
