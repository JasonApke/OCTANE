// File Name: main.cc
// Purpose: Wrapper for OCTANE functions, reads in command line args and launches optical flow code
// Inputs: Command Line Arguements (for a list, simply execute octane with no arguments)
// Outputs: Output is a netcdf file called "outfile.nc", placed in working directory
// Author: Jason Apke
// Contact: jason.apke@colostate.edu

#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <string>
#include "image.h"
#include "goesread.h"
#include "offlags.h"
using namespace std;

int oct_fileread(string,string,string,int,int,GOESVar &,OFFlags &);
int oct_filewrite(string,string,GOESVar &,OFFlags);
int oct_interp (GOESVar &,GOESVar &, float, OFFlags);
int oct_optical_flow(GOESVar &, GOESVar &,OFFlags &);



int main(int argc, char *argv[])
{
    //A few integers to check if things run smoothly
    int z;
    int t;
    OFFlags args; //a structure holding pertinent arguments
    int t3;
    string interpoutloc;
    //File name strings
    string f1, f2;
    string fc21, fc22;
    string fc31, fc32;
    string f1c, f2c;
    string f1fg;
    string interploc,outdir;
    //Argument definitions, used to be in order, though I removed a few with time
    string arg1="-i1",arg2="-i2",arg4="-i1cth",arg5="-i2cth",arg7="-farn",arg8="-pd",arg9="-srsal";
    string arg10="-Polar",arg11="-Merc",arg12="-ahi",arg13="-ir",arg14="-sosm",arg15="-interp";
    string arg16="-ic21",arg17="-ic22";
    string arg19="-ic31",arg20="-ic32";
    string arg22="-alpha",arg23="-lambda",arg24="-scsig",arg25="-alpha2",arg26="-lambdac";
    string arg27="-fwinsize",arg28="-polyn",arg29="-nncth",arg30="-inv",arg31="-ctt",arg32="-kiters",arg33="-brox",arg34="-corn";
    string arg35="-firstguess",arg36="-rad",arg37="-srad",arg38="-liters",arg39="-deltat",arg40="-interploc";
    string arg41="-no_outnav", arg42="-no_outraw", arg43="-no_outrad",arg44="-no_outctp",arg45="-set_device";
    string arg46="-normmax", arg47="-normmin", arg48="-normmax2", arg49="-normmin2", arg50="-normmax3", arg51="-normmin3",arg52="-o";
    GOESVar goesData,goesData2; //,goesData3;
    
    args.farn =0; //farneback set off, use -farn to turn on
    args.pixuv = 0; //output set to wind, use -pd to set to pixel displacement instead
    args.dosrsal= 0; //output not smoothed by default for SRSAL, set -srsal to perform this
    args.dopolar= 0; //input has a polar orthonormal grid 
    args.domerc= 0; //input has a mercator grid
    args.ftype="GOES";
    //Farneback defaults, eventually I will move these out of code so I don't have to keep updating git to change them
    args.fpyr_scale=0.5;
    args.flevels=2;
    args.fwinsize=20;
    args.fiterations=5;
    args.poly_n=10;
    args.poly_sigma=0.5;
    args.uif = 0;
    args.fg = 1;
    args.dofirstguess = 0;
    args.ir = 0;
    args.dososm = 0;
    args.dointerp=0;
    interploc="./interpolation";
    outdir = "./";
    args.docorn = 0;
    args.rad = 2;
    args.srad = 2;
    args.lambda=1.; //I found this too large, it introduced noise every once in a while, especially on boundary pixels
    args.alpha=5.;
    //I use this below when I have no median smoothing
    args.filtsigma=3.; //deprecated
    args.scaleF=0.5;
    args.kiters=4;
    args.alpha2=20.; //relevant for div-curl expansion, not for what we do yet
    args.lambdac=0.;
    args.liters=3; //3;
    args.cgiters=30;
    args.miters=5;
    args.scsig=400.; //deprecated
    args.interpcth = 1;
    args.deltat = 60.;
    args.doc2 = 0;
    args.doahi = 0;
    args.doc3 = 0;
    args.doinv = 0;
    args.doctt = 0; //deprecated
    args.doCTH = 0;
    args.dozim = 1;
    args.outraw= true;
    args.outctp= true;
    args.outrad= true;
    args.outnav= true;
    args.setdevice = 0;
    args.setNormMax=true;
    args.setNormMin=true;
    args.setNormMax2=true;
    args.setNormMin2=true;
    args.setNormMax3=true;
    args.setNormMin3=true;
    ////////////////////////////////End setting the defaults

    cout<< "Beginning variational dense optical flow..." << endl;
    if(argc < 4)
    {
        cout << "Optical Flow Toolkit for Atmospheric aNd Earth sciences (OCTANE)\n";
        cout << "Author: Jason Apke\n";
        cout << "Contact: jason.apke@colostate.edu\n";
        cout << "input flags:\n\n"; 
        cout << "-i1 <filename>, -i2 <filename> are the GOES-R file netcdf full paths, i1 is the first image, i2 is the second\n\n";
        cout << "-i1cth <filename>, -i2cth <filename> are optional paths to cloud top height netcdfs \n\n";
        cout << "-nncth instead of default bilinear interpolation, remap the CTH grids with nearest neighbor \n\n";
        cout << "-o <directory> writes the file to the designated directory, include slash at the end (default is ./) \n\n";
        cout << "-pd forces OCTANE to return unnavigated pixel displacements \n\n";
        cout << "-srsal has OCTANE return bilinearly smoothed optical flow output (useful for computing cloud-top divergence) \n\n";
        cout << "-Polar use this flag to ingest polar-orthonormal grid images instead of GOES projections (used for sea-ice tracking)\n\n";
        cout << "-Merc use this flag to injest mercator grid images instead of GOES projections (definition) \n\n";
        cout << "-ahi (deprecated) use when reading netcdfs converted from AHI binaries \n\n";
        cout << "-ir use this flag to output ir temperatures instead of cloud-top height (changes the scaling of the short variable ctp)\n\n";
        cout << "-sosm use this flag to perform least-squares minimization or patch-match tracking instead of zimmer optical flow \n\n";
        cout << "-rad <int value> set the target radius (in x- and y-pixels) for sosm tracking \n\n";
        cout << "-srad <int value> set the search radius (in x- and y-pixels) for sosm tracking \n\n";
        cout << "-interp use this flag to perform optical flow image interpolation (temporal for image sequences) \n\n";
        cout << "-normmin(2|3) <value> sets the image brighness minimum for normalization (defaults for each band from GOES determined in OCT bandminmax) \n\n";
        cout << "-normmax(2|3) <value> sets the image brighness maxiumum for normalization (defaults for each band from GOES determined in OCT bandminmax) \n\n";
        cout << "-deltat <value> use this to set the framerate for optical flow interpolation (in seconds) \n\n";
        cout << "-interpout <directory> use this to set the output for the interp directory \n\n";
        //Need to add flags for interpolation -J. Apke 2/10/2022
        cout << "-ic21 <filename> -ic22 <filename> are flags to input another channel (ch2) for files 1 and 2 (useful for RGB tracking)\n\n";
        cout << "-ic31 <filename> -ic32 <filename> are flags to input another channel (ch3) for files 1 and 2 (useful for RGB tracking)\n\n";
        cout << "-alpha <value> is a flag to set the smoothness constraint constant for Brox/Zimmer-based approaches \n\n";
        cout << "-lambda <value> is a flag to set the gradient constraint constant for Brox/Zimmer-based approaches \n\n";
        cout << "-lambdac <value> this is to set the weight of a hinting term, only used when -firstguess is active \n\n";
        cout << "-kiters <int value> number of outer iterations/pyramid levels in Brox/Zimmer-based approaches, default is 4 \n\n";
        cout << "-liters <int value> number of inner iterations/pyramid levels in Brox/Zimmer-based approaches, default is 3 (use more if OF struggles to converge)\n\n";
        cout << "-cgiters <int value> maximum number of preconditioned conjugate gradient iterations in Brox/Zimmer-based approaches, default is 30 (use more if OF struggles to converge)\n\n";
        cout << "-scsig (deprecated) this was the sigma value for experimental smoothness constraint robust functions which are no longer used \n\n";
        cout << "-alpha2 (deprecated) This is a smoothness constraint constant term for a div-curl approach which is not yet added (but planned) within the Conjugate Gradient solver \n\n";
        cout << "Farneback Definitions (All Deprecated) \n\n";
        cout << "-fwinsize -polyn both are inputs to Farneback opencv algorithm (see documentation here https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html \n\n";
        cout << "-inv (deprecated) flag to pass through inversion check from clavrx files \n\n";
        cout << "-ctt (deprecated) flag to pass through estimated temperatures from clavrx files \n\n";
        cout << "-brox set to perform pure Brox approach (default is modified zimmer), this flag turns off gradient (and lapacian) scaling of brightness (and gradient) constraints \n\n";
        cout << "-corn set to output a Shi/Tomasi 1996 corner detection algorithm, and output corner locations \n\n";
        cout << "-firstguess <filename> set to input a first guess file (Only for GOES files, motions must be navigated) \n\n";
        cout << "-no_outnav turns off output of navigated u/v optical flow motions\n\n";
        cout << "-no_outraw turns off output of raw (pixel) u/v optical flow displacements\n\n";
        cout << "-no_outrad turns off output of imagery used to derive optical flow\n\n";
        cout << "-no_outctp turns off output of cloud-top height (or infrared) data used with the imagery\n\n";
        cout << "-set_device <int> sets which gpu to run on, 1-based (default is 1), must be less than # of gpus \n\n";
        cout << "Below is an example running octane on a sequence of GOES image files: \n\n";

        cout << "./octane \n";
        
        return 0;
    }
    //may want to switch these int switches to boolean flags -J. Apke 2/10/2022
    for (int i = 0; i < argc; ++i)
    {
        if(!arg1.compare(argv[i]))
            f1=argv[i+1];
        if(!arg2.compare(argv[i]))
            f2=argv[i+1];
        if(!arg4.compare(argv[i]))
        {
            f1c=argv[i+1];
            args.doCTH = 1;
        }
        if(!arg5.compare(argv[i]))
            f2c=argv[i+1];
        if(!arg7.compare(argv[i]))
        {
            args.farn=1; 
            printf("Farneback disabled for this version of OCTANE, run without -farn, exiting...");
            exit(0);
        }
        if(!arg8.compare(argv[i]))
            args.pixuv=1; 
        if(!arg9.compare(argv[i]))
            args.dosrsal=1;
        if(!arg10.compare(argv[i]))
        {
            args.dopolar=1;
            args.ftype="POLAR";
        }
        if(!arg11.compare(argv[i]))
        {
            args.domerc=1;
            args.ftype="MERC";
        }
        if(!arg12.compare(argv[i]))
            args.doahi=1;
        if(!arg13.compare(argv[i]))
            args.ir=1; 
        if(!arg14.compare(argv[i]))
            args.dososm=1; 
        if(!arg15.compare(argv[i]))
            args.dointerp=1; 
        if(!arg16.compare(argv[i]))
        {
            args.doc2=1;
            fc21=argv[i+1];
        }
        if(!arg17.compare(argv[i]))
            fc22=argv[i+1];
        //Third channel option
        if(!arg19.compare(argv[i]))
        {
            args.doc3=1;
            fc31=argv[i+1];
        }
        if(!arg20.compare(argv[i]))
            fc32=argv[i+1];
        if(!arg22.compare(argv[i]))
        {
            args.alpha=atof(argv[i+1]);
        }
        if(!arg23.compare(argv[i]))
            args.lambda=atof(argv[i+1]);
        if(!arg24.compare(argv[i]))
            args.scsig=atof(argv[i+1])*atof(argv[i+1]);
        if(!arg25.compare(argv[i]))
        {
            args.alpha2=atof(argv[i+1]);
        }
        if(!arg26.compare(argv[i]))
        {
            args.lambdac=atof(argv[i+1]);
        }
        if(!arg27.compare(argv[i]))
        {
            args.fwinsize=atoi(argv[i+1]);
        }
        if(!arg28.compare(argv[i]))
        {
            args.poly_n=atoi(argv[i+1]);
        }
        if(!arg29.compare(argv[i]))
        {
            args.interpcth=0;
        }
        if(!arg30.compare(argv[i]))
        {
            args.doinv=1;
        }
        if(!arg31.compare(argv[i]))
        {
            args.doctt=1;
        }
        if(!arg32.compare(argv[i]))
        {
            args.kiters=atoi(argv[i+1]);
        }
        if(!arg38.compare(argv[i]))
        {
            args.liters=atoi(argv[i+1]);
        }
        if(!arg33.compare(argv[i]))
        {
            args.dozim=0;
        }
        if(!arg34.compare(argv[i]))
        {
            args.docorn=0;
        }
        if(!arg35.compare(argv[i]))
        {
            args.dofirstguess =1;
            f1fg=argv[i+1];
        }
        if(!arg36.compare(argv[i]))
        {
            args.rad=atoi(argv[i+1]);
        }
        if(!arg37.compare(argv[i]))
        {
            args.srad=atoi(argv[i+1]);
        }
        if(!arg39.compare(argv[i]))
        {
            args.deltat=atof(argv[i+1]);
        }
        if(!arg40.compare(argv[i]))
        {
            interploc=argv[i+1];
        }
        if(!arg41.compare(argv[i]))
        {
            args.outnav=false;
        }
        if(!arg42.compare(argv[i]))
        {
            args.outraw=false;
        }
        if(!arg43.compare(argv[i]))
        {
            args.outrad=false;
        }
        if(!arg44.compare(argv[i]))
        {
            args.outctp=false;
        }
        if(!arg45.compare(argv[i]))
        {
            args.setdevice=atoi(argv[i+1])-1;
        }
        if(!arg46.compare(argv[i]))
        {
            args.NormMax=atof(argv[i+1]);
            args.setNormMax=false;
        }
        if(!arg47.compare(argv[i]))
        {
            args.NormMin=atof(argv[i+1]);
            args.setNormMin=false;
        }
        if(!arg48.compare(argv[i]))
        {
            args.NormMax2=atof(argv[i+1]);
            args.setNormMax2=false;
        }
        if(!arg49.compare(argv[i]))
        {
            args.NormMin2=atof(argv[i+1]);
            args.setNormMin2=false;
        }
        if(!arg50.compare(argv[i]))
        {
            args.NormMax3=atof(argv[i+1]);
            args.setNormMax3=false;
        }
        if(!arg51.compare(argv[i]))
        {
            args.NormMin3=atof(argv[i+1]);
            args.setNormMin3=false;
        }
        if(!arg52.compare(argv[i]))
        {
            outdir=argv[i+1];
        }
        
    }
    //quick check for multi-channel files
    if((args.doc2 == 1) && ((fc22 == "none")))
    {
        printf("Missing files for second channel...stopping \n");
        exit(0);
    }
    if((args.doc3 == 1) && ((fc32 == "none")))
    {
        printf("Missing files for third channel...stopping \n");
        exit(0);
    }
    //Set the optical flow type for the output file based on input command line arguments
    if(args.farn == 1)
    {
        // 2 for farneback, which will be removed from OCTANE to remove dependencies on OPENCV
        args.oftype=2;
    } else{
        //1 for Zimmer
        args.oftype=1;
        if(args.dozim == 0)
        {
            //3 for Brox
            args.oftype=3;
        }
    }
    if(args.dososm == 1)
    {
        args.oftype=4;
    }
    if(args.dopolar == 1)
    {
        args.doCTH = 0;
    }
    if(args.domerc == 1)
    {
        args.doCTH = 0;
    }
    if(args.doahi == 1)
    {
        args.doCTH = 0;
    }

    cout <<"Here are the file names being used: \n";
    cout <<"File 1 : " << f1 << endl;
    cout <<"File 2 : " << f2 << endl;
    //First step is to read the GOES data, jma_goesread/polarread/mercread fills the GOESVar variables
    
    t = oct_fileread(f1,args.ftype,"RAW",1,1,goesData,args);
    t = oct_fileread(f2,args.ftype,"RAW",0,1,goesData2,args);
    
    if((args.dopolar==0) && (args.domerc==0))
    {
        goesData.nav.g2xOffset = goesData2.nav.xOffset;
        goesData.nav.g2yOffset = goesData2.nav.yOffset;
    }
    //This is a reader for ancilliary cloud-top height information, which gets stored in the goesData objects
    if(args.doCTH == 1){
        t = oct_fileread(f1c,"CLAVRX","RAW",0,0,goesData,args);
    }
    if(args.dofirstguess == 1){
        t = oct_fileread(f1fg,"FIRSTGUESS","RAW",0,0,goesData,args);
    }

    //Below are readers for multi-channel inputs. Currently, 3 channels are the most allowed
    if(args.doc2 == 1)
    {
        //I will need to add this support later -J. Apke 2/11/2022
        if((args.domerc == 1) )
        {
            printf("Mercator multi-channel not compatable with this version, use single channel only\n");
            exit(0);
        }
        t = oct_fileread(fc21,args.ftype,"RAW",1,2,goesData,args);
        t = oct_fileread(fc22,args.ftype,"RAW",0,2,goesData2,args); 
    }
    if(args.doc3 == 1)
    {
        if((args.domerc == 1) )
        {
            printf("Mercator multi-channel not compatable with this version, use single channel only\n");
            exit(0);
        }
        t = oct_fileread(fc31,args.ftype,"RAW",1,3,goesData,args);
        t = oct_fileread(fc32,args.ftype,"RAW",0,3,goesData2,args); 
    }
    
    //The function below is the primary optical flow computation algorithm.  It reads in the goesData objects for each image (goesData and
    //goesData2), and the arguments provided for the optical flow settings, and fills the uval and vval arrays in the goesData objects
    t3 = oct_optical_flow(goesData,goesData2,args);

    //The interpolated files have slightly different output in filewrite, hence the setting below
    args.putinterp = 0;
    string outname = outdir+"outfile.nc";
    if(args.ftype=="POLAR") outname=outdir+"outfile_polar.nc";
    if(args.ftype=="MERC") outname=outdir+"outfile_merc.nc";
    //Writes the output file
    t = oct_filewrite(outname,args.ftype,goesData,args);
    cout << outname << " written\n";

    if(args.dointerp == 1)
    {
        //loop through each individual interp file requested, and write it out
        cout << "Interpolation flag on, be warned, this part is still under development!!!\n";
        int fwriteint = 1;
        float deltat = args.deltat; //deltat in seconds that we want to change
        float frt = deltat/goesData.dT;
        cout << "Outputting files every " << deltat << " seconds\n";
        
        args.putinterp = 1;
        while((frt < 1.) && ((1.-frt) >= ((deltat/goesData.dT)/2.)))
        {
            std::stringstream interpoutlocstr;
            interpoutlocstr << fwriteint;
            string interpoutloc;
            t3 = oct_interp(goesData,goesData2,frt,args);
            if(args.dopolar == 1)
            {
                interpoutloc = interploc+"/outfile_interp_polar"+interpoutlocstr.str()+".nc";
            } else
            {
                interpoutloc = interploc+"/outfile_interp"+interpoutlocstr.str()+".nc";
            }
            t = oct_filewrite(interpoutloc,args.ftype,goesData,args);
            fwriteint++;
            frt += deltat/goesData.dT;
            cout << interpoutloc << " written" << endl;
            cout << "FRT " << frt << " out of 1 " << endl;
        }

    }
    cout << "OCTANE completed, exiting\n";

    return 0;
}
