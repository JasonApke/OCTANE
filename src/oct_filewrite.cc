#include <iostream>
#include <netcdf>
#include "image.h"
#include "goesread.h"
#include "offlags.h"
using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

//Function: oct_filewrite- a function to write the output from OCTANE
//Requires: The netcdf C++ library
//Author: Jason Apke, Updated 2/21/2022
static const int NC_ERR = 2;
//There are a few functions designed for outputting certian types of files, the primary is goeswrite
//Additions will be made for a simple image output as well -J. Apke 2/22/2022

int oct_goeswrite (string fpath,GOESVar &resVar,OFFlags args)
{
    //This is a function designed for reading GOES data files
    int r = 1;
    try
    {
        NcFile ncf(fpath, NcFile::replace);
        NcVar dataVar22, dataVar23,dataVar32,dataVar33,dataVar2Occlusion;
        NcVar cnrVar,dataVarraw,dataVarraw2,dataVar,dataVar2,dataVarCTP,dataVar3;
        NcVar fk1Var,fk2Var,bc1Var,bc2Var,kapVar;
        NcVar fk1Var2,fk2Var2,bc1Var2,bc2Var2,kapVar2;
        NcVar fk1Var3,fk2Var3,bc1Var3,bc2Var3,kapVar3;
        NcDim xDim = ncf.addDim("x",resVar.nav.nx);
        NcDim yDim = ncf.addDim("y",resVar.nav.ny);
        
        //now create the variable
        NcVar xVar = ncf.addVar("x",ncShort, xDim);
        NcVar yVar = ncf.addVar("y",ncShort, yDim);
        xVar.putAtt("scale_factor",NC_FLOAT,resVar.nav.xScale);
        xVar.putAtt("add_offset",NC_FLOAT,resVar.nav.xOffset);
        yVar.putAtt("scale_factor",NC_FLOAT,resVar.nav.yScale);
        yVar.putAtt("add_offset",NC_FLOAT,resVar.nav.yOffset);
        
        xVar.putVar(resVar.x);
        yVar.putVar(resVar.y);
        
        NcVar tVar = ncf.addVar("t",ncDouble);
        tVar.putAtt("standard_name","time");
        tVar.putAtt("units",resVar.tUnits);
        tVar.putAtt("axis","T");
        tVar.putAtt("bounds","time_bounds");
        tVar.putAtt("long_name" , "J2000 epoch mid-point between the start and end image scan in seconds");
        if(args.putinterp == 1){
            tVar.putAtt("frdt",NC_FLOAT,resVar.frdt);
        }

        if(args.putinterp == 0)
        {

            tVar.putVar(&resVar.t);
        } else
        {
            tVar.putVar(&resVar.tint);
        }


        vector<NcDim> dims;
        dims.push_back(yDim);
        dims.push_back(xDim);

        if(args.outnav){
            dataVar = ncf.addVar("U",ncShort,dims);
            dataVar2 = ncf.addVar("V",ncShort,dims);
        }
        if(args.outraw){
            dataVarraw = ncf.addVar("U_raw",ncShort,dims);
            dataVarraw2 = ncf.addVar("V_raw",ncShort,dims);
        }
        if(args.docorn == 1)
            cnrVar = ncf.addVar("cnr",ncShort,dims);
        //This is to output the full float resolution pixel displacements if needed
        //Though it should not be predicated on dosrsal...
        if(args.pixuv==1)
        {
            dataVar22 = ncf.addVar("Upix",ncFloat,dims);
            dataVar23 = ncf.addVar("Vpix",ncFloat,dims);
        }
        if(args.putinterp==1)
        {
            //variable to store the occlusion masks
            dataVar2Occlusion = ncf.addVar("Occlusion",ncShort,dims);
        }
        if(args.outctp && (args.doCTH == 1))
        {
            dataVarCTP = ncf.addVar("CTP",ncShort,dims);
        }

        if(args.outrad)
        {
            dataVar3 = ncf.addVar("Rad",ncShort,dims);
            if(args.doc2 == 1) dataVar32 = ncf.addVar("Rad2",ncShort,dims);
            if(args.doc3 == 1) dataVar33 = ncf.addVar("Rad3",ncShort,dims);
        }
        //Below are not optional outputs, including the goes projection and the optical flow settings
        NcVar gipVar = ncf.addVar("goes_imager_projection",NC_INT);
        NcVar ofVar = ncf.addVar("optical_flow_settings",NC_INT);
        //no need for below if outrad is false
        if(args.outrad)
        {

            fk1Var = ncf.addVar("planck_fk1",NC_FLOAT);
            fk2Var = ncf.addVar("planck_fk2",NC_FLOAT);
            bc1Var = ncf.addVar("planck_bc1",NC_FLOAT);
            bc2Var = ncf.addVar("planck_bc2",NC_FLOAT);
            kapVar = ncf.addVar("kappa0",NC_FLOAT);
            if(args.doc2 == 1)
            {
                fk1Var2 = ncf.addVar("planck_fk1_2",NC_FLOAT);
                fk2Var2 = ncf.addVar("planck_fk2_2",NC_FLOAT);
                bc1Var2 = ncf.addVar("planck_bc1_2",NC_FLOAT);
                bc2Var2 = ncf.addVar("planck_bc2_2",NC_FLOAT);
                kapVar2 = ncf.addVar("kappa0_2",NC_FLOAT);
            }
            if(args.doc3 == 1)
            {
                fk1Var3 = ncf.addVar("planck_fk1_3",NC_FLOAT);
                fk2Var3 = ncf.addVar("planck_fk2_3",NC_FLOAT);
                bc1Var3 = ncf.addVar("planck_bc1_3",NC_FLOAT);
                bc2Var3 = ncf.addVar("planck_bc2_3",NC_FLOAT);
                kapVar3 = ncf.addVar("kappa0_3",NC_FLOAT);
            }
        }

        if(args.outnav){
            dataVar.putAtt("long_name","U");
            dataVar.putAtt("grid_mapping","goes_imager_projection");
            dataVar.putAtt("scale_factor",NC_FLOAT,0.01);
            dataVar.putAtt("grid_mapping","goes_imager_projection");
            if(args.pixuv == 0)
            {
                dataVar.putAtt("units","meters per second");
            }else{
                dataVar.putAtt("units","x-pixels");
            }
            dataVar2.putAtt("long_name","V");
            dataVar2.putAtt("grid_mapping","goes_imager_projection");
            dataVar2.putAtt("scale_factor",NC_FLOAT,0.01);
            
            dataVar2.putAtt("grid_mapping","goes_imager_projection");
            if(args.pixuv == 1)
            {
                dataVar2.putAtt("units","y-pixels");
                if(args.dosrsal == 1)
                {
                    dataVar22.putAtt("long_name","Upix");
                    dataVar22.putAtt("grid_mapping","goes_imager_projection");
                    dataVar22.putAtt("grid_mapping","goes_imager_projection");
                    
                    dataVar23.putAtt("long_name","Vpix");
                    dataVar23.putAtt("grid_mapping","goes_imager_projection");
                    dataVar23.putAtt("grid_mapping","goes_imager_projection");
                }
            }else{
                dataVar2.putAtt("units","meters per second");
            }
        }

        
        
        if(args.docorn == 1) cnrVar.putAtt("long_name","Corner Locations");
        if(args.outraw){
            dataVarraw.putAtt("long_name","U Raw");
            dataVarraw.putAtt("grid_mapping","goes_imager_projection");
            dataVarraw.putAtt("scale_factor",NC_FLOAT,0.01);
            
            dataVarraw.putAtt("grid_mapping","goes_imager_projection");
            dataVarraw.putAtt("units","x-pixels");

            dataVarraw2.putAtt("long_name","V Raw");
            dataVarraw2.putAtt("grid_mapping","goes_imager_projection");
            dataVarraw2.putAtt("scale_factor",NC_FLOAT,0.01);
            
            dataVarraw2.putAtt("grid_mapping","goes_imager_projection");
            dataVarraw2.putAtt("units","y-pixels");
        }
        if(args.putinterp==1)
        {
            dataVar2Occlusion.putAtt("long_name","Occlusion Masks");
            dataVar2Occlusion.putAtt("key","0 - both, 1 - only in image 1, 2 - only in image 2");
        }

        

        if(args.outctp & (args.doCTH == 1))
        {
            dataVarCTP.putAtt("long_name","CTP");
            dataVarCTP.putAtt("grid_mapping","goes_imager_projection");
            dataVarCTP.putAtt("interpcth",NC_FLOAT,args.interpcth); //1 if cth was interpolated w/ bicubic interpolation, 0 if nearest neighbor
        }


        if(args.outrad)
        {
            dataVar3.putAtt("long_name","Rad");
            dataVar3.putAtt("grid_mapping","goes_imager_projection");
            dataVar3.putAtt("scale_factor",NC_FLOAT,resVar.nav.radScale);
            dataVar3.putAtt("add_offset",NC_FLOAT,resVar.nav.radOffset);
            dataVar3.putAtt("grid_mapping","goes_imager_projection");
            if(args.doc2 == 1){
                dataVar32.putAtt("long_name","Rad2");
                dataVar32.putAtt("grid_mapping","goes_imager_projection");
                dataVar32.putAtt("scale_factor",NC_FLOAT,resVar.nav.radScale2);
                dataVar32.putAtt("add_offset",NC_FLOAT,resVar.nav.radOffset2);
                dataVar32.putAtt("grid_mapping","goes_imager_projection");
            }
            if(args.doc3 == 1){
                dataVar33.putAtt("long_name","Rad2");
                dataVar33.putAtt("grid_mapping","goes_imager_projection");
                dataVar33.putAtt("scale_factor",NC_FLOAT,resVar.nav.radScale3);
                dataVar33.putAtt("add_offset",NC_FLOAT,resVar.nav.radOffset3);
                dataVar33.putAtt("grid_mapping","goes_imager_projection");
            }
        }
        gipVar.putAtt("long_name" , "GOES-R ABI fixed grid projection");
        gipVar.putAtt("grid_mapping_name" , "geostationary");
        gipVar.putAtt("perspective_point_height",NC_DOUBLE,resVar.nav.pph);
        gipVar.putAtt("semi_major_axis",NC_DOUBLE,resVar.nav.req);
        gipVar.putAtt("semi_minor_axis",NC_DOUBLE,resVar.nav.rpol);
        gipVar.putAtt("inverse_flattening",NC_DOUBLE, resVar.nav.inverse_flattening) ;
        gipVar.putAtt("latitude_of_projection_origin",NC_DOUBLE,resVar.nav.lat0);
        gipVar.putAtt("longitude_of_projection_origin",NC_DOUBLE,resVar.nav.lpo);
        gipVar.putAtt("sweep_angle_axis","x");

        ofVar.putAtt("long_name" , "Optical Flow Settings");
        ofVar.putAtt("key" , "1 = Modified Zimmer et al. (2011), 2 = Farneback, 3 = Brox (2004), 4 = Least Squares");
        //An additional navigation variable to check is the nav for the other image used
        //Below changes when a mesosector moves, which can create undesirable optical flow results
        ofVar.putAtt("Image2_xOffset",NC_FLOAT,resVar.nav.g2xOffset);
        ofVar.putAtt("Image2_yOffset",NC_FLOAT,resVar.nav.g2yOffset);
        if((args.oftype==1) || (args.oftype==3))
        {
            //Make sure to add ALL arguments used for reproducing modified zimmer approach -J. Apke 2/10/2022
            ofVar.putAtt("lambda",NC_DOUBLE,args.lambda);
            ofVar.putAtt("lambdac",NC_DOUBLE,args.lambdac); //hinting term weight (0 if dofirstguess==0)
            ofVar.putAtt("alpha", NC_DOUBLE,args.alpha);
            ofVar.putAtt("filtsigma", NC_DOUBLE,args.filtsigma);
            ofVar.putAtt("ScaleF", NC_DOUBLE,args.scaleF);
            ofVar.putAtt("K_Iterations", NC_INT,args.kiters);
            ofVar.putAtt("L_Iterations", NC_INT,args.liters);
            ofVar.putAtt("M_Iterations", NC_INT,args.miters);
            ofVar.putAtt("CG_Iterations", NC_INT,args.cgiters);
            ofVar.putAtt("NormMax", NC_FLOAT,args.NormMax);
            ofVar.putAtt("NormMin", NC_FLOAT,args.NormMin);
            ofVar.putAtt("dofirstguess", NC_INT,args.dofirstguess);
        }
        //Below is farneback which has been removed from this iteration of OCTANE
        //to remove the dependencies on opencv
        if(args.oftype==2)
        {
            ofVar.putAtt("pyr_scale",NC_FLOAT,args.fpyr_scale);
            ofVar.putAtt("levels", NC_INT,args.flevels);
            ofVar.putAtt("winsize", NC_INT,args.fwinsize);
            ofVar.putAtt("iterations", NC_INT,args.fiterations);
            ofVar.putAtt("poly_n", NC_INT,args.poly_n);
            ofVar.putAtt("poly_sigma", NC_FLOAT,args.poly_sigma);
            ofVar.putAtt("use_initial_flow", NC_INT,args.uif);
            ofVar.putAtt("farneback_gaussian", NC_INT,args.fg);
            ofVar.putAtt("NormMax", NC_FLOAT,args.NormMax);
            ofVar.putAtt("NormMin", NC_FLOAT,args.NormMin);
        }
        //Patch Match optical flow settings
        if(args.oftype==4)
        {
            ofVar.putAtt("Rad", NC_INT,args.rad);
            ofVar.putAtt("SRad", NC_INT,args.srad);
            ofVar.putAtt("NormMax", NC_FLOAT,args.NormMax);
            ofVar.putAtt("NormMin", NC_FLOAT,args.NormMin);
        }
        //This is only important for image sequences, dT gives the time difference between images
        ofVar.putAtt("dt_seconds",NC_FLOAT,resVar.dT);


        if(args.outnav){
            dataVar.putVar(resVar.uVal);
            dataVar2.putVar(resVar.vVal);
        }
        if(args.outraw){
            dataVarraw.putVar(resVar.uVal2);
            dataVarraw2.putVar(resVar.vVal2);
        }
        
        if(args.pixuv == 1)
        {
            dataVar22.putVar(resVar.uPix);
            dataVar23.putVar(resVar.vPix);
        }
        if(args.putinterp ==1)
        {
            dataVar2Occlusion.putVar(resVar.occlusion); 
        }

        if(args.outctp && (args.doCTH == 1))
        {
            dataVarCTP.putVar(resVar.CTP);
        }

        if(args.outrad)
        {
            if(args.putinterp == 0)
            {
                dataVar3.putVar(resVar.dataSVal);
                if(args.doc2 == 1) dataVar32.putVar(resVar.dataSVal2);
                if(args.doc3 == 1) dataVar33.putVar(resVar.dataSVal3);
            } else
            {
                dataVar3.putVar(resVar.dataSVal);
                if(args.doc2 == 1) dataVar32.putVar(resVar.dataSVal2);
                if(args.doc3 == 1) dataVar33.putVar(resVar.dataSVal3);
            }
        }
        gipVar.putVar(&resVar.nav.gipVal);
        if(args.outrad){
            fk1Var.putVar(&resVar.nav.fk1);
            fk2Var.putVar(&resVar.nav.fk2);
            bc1Var.putVar(&resVar.nav.bc1);
            bc2Var.putVar(&resVar.nav.bc2);
            kapVar.putVar(&resVar.nav.kap1);
            if(args.doc2 == 1)
            {
                fk1Var2.putVar(&resVar.nav.fk12);
                fk2Var2.putVar(&resVar.nav.fk22);
                bc1Var2.putVar(&resVar.nav.bc12);
                bc2Var2.putVar(&resVar.nav.bc22);
                kapVar2.putVar(&resVar.nav.kap12);
            }
            if(args.doc3 == 1)
            {
                fk1Var3.putVar(&resVar.nav.fk13);
                fk2Var3.putVar(&resVar.nav.fk23);
                bc1Var3.putVar(&resVar.nav.bc13);
                bc2Var3.putVar(&resVar.nav.bc23);
                kapVar3.putVar(&resVar.nav.kap13);
            }
        }
        return 0;
    }
    catch(NcException& e)
    {
        e.what();
        cout << "GOESWRITE failure\n";
        return NC_ERR;
    }
}
//writes polar orthonormal grid files
//Still a bit more to add here, raw output settings, also full float output is
//important for slow motions -J. Apke 2/23/2022
int oct_polarwrite (string fpath,GOESVar &resVar,OFFlags args)
{
    int r = 1;
    float *dummy;
    try
    {
        NcFile ncf(fpath, NcFile::replace);
        NcVar dataVar22, dataVar23,dataVar2Occlusion,dataVar3;
        NcVar dataVar32, dataVar33;
        NcDim xDim = ncf.addDim("x",resVar.nav.nx);
        NcDim yDim = ncf.addDim("y",resVar.nav.ny);

        //now create the variable
        NcVar xVar = ncf.addVar("x",ncShort, xDim);
        NcVar yVar = ncf.addVar("y",ncShort, yDim);
        xVar.putAtt("scale_factor",NC_FLOAT,resVar.nav.xScale);
        xVar.putAtt("add_offset",NC_FLOAT,resVar.nav.xOffset);
        yVar.putAtt("scale_factor",NC_FLOAT,resVar.nav.yScale);
        yVar.putAtt("add_offset",NC_FLOAT,resVar.nav.yOffset);

        xVar.putVar(resVar.x);
        yVar.putVar(resVar.y);

        //add the time variables
        NcVar tVar = ncf.addVar("t",ncDouble);
        tVar.putAtt("standard_name","time");
        tVar.putAtt("units",resVar.tUnits);
        tVar.putAtt("axis","T");
        tVar.putAtt("bounds","time_bounds");
        tVar.putAtt("long_name" , "J2000 epoch mid-point between the start and end image scan in seconds");

        if(args.dointerp == 1){
            tVar.putAtt("frdt",NC_FLOAT,resVar.frdt);
        }
        if(args.putinterp == 0)
        {
            tVar.putVar(&resVar.t);
        } else
        {
            tVar.putVar(&resVar.tint);
        }

        //now add the data variable

        vector<NcDim> dims;
        dims.push_back(yDim);
        dims.push_back(xDim);
        NcVar dataVar = ncf.addVar("U",ncDouble,dims);
        NcVar dataVar2 = ncf.addVar("V",ncDouble,dims);
        if(args.pixuv==1)
        {
            dataVar22 = ncf.addVar("Upix",ncFloat,dims);
            dataVar23 = ncf.addVar("Vpix",ncFloat,dims);
        }
        if(args.outrad) 
        {
            dataVar3 = ncf.addVar("Rad",ncFloat,dims);
            if(args.doc2 == 1) dataVar32 = ncf.addVar("Rad2",ncFloat,dims);
            if(args.doc3 == 1) dataVar33 = ncf.addVar("Rad3",ncFloat,dims);
        }
        if(args.dointerp==1)
        {
            dataVar2Occlusion = ncf.addVar("Occlusion",ncShort,dims);
        }
        NcVar gipVar = ncf.addVar("polar_imager_projection",NC_INT);
        NcVar ofVar = ncf.addVar("optical_flow_settings",NC_INT);

        //define atts
        dataVar.putAtt("long_name","U");
        dataVar.putAtt("grid_mapping","polar_orthonormal");
        if(args.pixuv == 0)
        {
            dataVar.putAtt("units","meters per second");
        }else{
            dataVar.putAtt("units","x-pixels");
        }


        dataVar2.putAtt("long_name","V");
        dataVar2.putAtt("grid_mapping","polar_orthonormal");

        if(args.pixuv == 1)
        {
            dataVar2.putAtt("units","y-pixels");
            if(args.dosrsal == 1)
            {
                dataVar22.putAtt("long_name","Upix");

                dataVar23.putAtt("long_name","Vpix");
            }
        }else{
            dataVar2.putAtt("units","meters per second");
        }
        if(args.dointerp==1)
        {
            dataVar2Occlusion.putAtt("long_name","Occlusion Masks");
            dataVar2Occlusion.putAtt("key","0 - both, 1 - only in image 1, 2 - only in image 2");
        }
        if(args.outrad)
        {
            dataVar3.putAtt("long_name","Rad");
            dataVar3.putAtt("grid_mapping","polar_orthonormal");
            if(args.doc2 == 1)
            {
                dataVar32.putAtt("long_name","Rad2");
                dataVar32.putAtt("grid_mapping","polar_orthonormal");
            }
            if(args.doc3 == 1)
            {
                dataVar33.putAtt("long_name","Rad3");
                dataVar33.putAtt("grid_mapping","polar_orthonormal");
            }
        }
        gipVar.putAtt("long_name" , "Polar_Orthonormal_Grid");
        gipVar.putAtt("grid_mapping_name" , "polar");
        double lat1val = resVar.nav.lat1;
        double lon0val = resVar.nav.lon0;
        double Rval = resVar.nav.R;
        gipVar.putAtt("lat1",NC_DOUBLE,lat1val);
        gipVar.putAtt("lon0",NC_DOUBLE,lon0val);
        gipVar.putAtt("R",NC_DOUBLE,Rval);

        ofVar.putAtt("long_name" , "Optical Flow Settings");
        ofVar.putAtt("key" , "1 = Modified Sun (2014), 2 = Farneback, 3 = Brox (2004)");
        if(args.oftype==1 || args.oftype==3)
        {
            ofVar.putAtt("lambda",NC_DOUBLE,args.lambda);
            ofVar.putAtt("lambdac",NC_DOUBLE,args.lambdac); //hinting term weight (0 if dofirstguess==0)
            ofVar.putAtt("alpha", NC_DOUBLE,args.alpha);
            ofVar.putAtt("filtsigma", NC_DOUBLE,args.filtsigma);
            ofVar.putAtt("ScaleF", NC_DOUBLE,args.scaleF);
            ofVar.putAtt("K_Iterations", NC_INT,args.kiters);
            ofVar.putAtt("L_Iterations", NC_INT,args.liters);
            ofVar.putAtt("M_Iterations", NC_INT,args.miters);
            ofVar.putAtt("CG_Iterations", NC_INT,args.cgiters);
            ofVar.putAtt("NormMax", NC_FLOAT,args.NormMax);
            ofVar.putAtt("NormMin", NC_FLOAT,args.NormMin);
            ofVar.putAtt("dofirstguess", NC_INT,args.dofirstguess);
        }
        if(args.oftype==2)
        {
            ofVar.putAtt("pyr_scale",NC_FLOAT,args.fpyr_scale);
            ofVar.putAtt("levels", NC_INT,args.flevels);
            ofVar.putAtt("winsize", NC_INT,args.fwinsize);
            ofVar.putAtt("iterations", NC_INT,args.fiterations);
            ofVar.putAtt("poly_n", NC_INT,args.poly_n);
            ofVar.putAtt("poly_sigma", NC_FLOAT,args.poly_sigma);
            ofVar.putAtt("use_initial_flow", NC_INT,args.uif);
            ofVar.putAtt("farneback_gaussian", NC_INT,args.fg);
            ofVar.putAtt("NormMax", NC_FLOAT,args.NormMax);
            ofVar.putAtt("NormMin", NC_FLOAT,args.NormMin);
        }
        ofVar.putAtt("dt_seconds",NC_FLOAT,resVar.dT);
        
        dataVar.putVar(resVar.uPix);
        dataVar2.putVar(resVar.vPix);
        if(args.pixuv == 1)
        {
            dataVar22.putVar(resVar.uPix);
            dataVar23.putVar(resVar.vPix);
        }
        if(args.putinterp ==1)
        {
            dataVar2Occlusion.putVar(resVar.occlusion);
        }
        if(args.putinterp == 0)
        {
            long nxtny = resVar.nav.nx*resVar.nav.ny;
            dummy = new float[nxtny];
            for(int dum=0; dum < (resVar.nav.nx*resVar.nav.ny); dum++)
            {
                dummy[dum] = resVar.data.data[dum];
            }
            if(args.outrad)
            {
                dataVar3.putVar(dummy);
                if(args.doc2 == 1)
                {
                    for(int dum=0; dum < (resVar.nav.nx*resVar.nav.ny); dum++)
                    {
                        dummy[dum] = resVar.data.data[dum+nxtny];
                    }
                    dataVar32.putVar(dummy);
                }
                if(args.doc3 == 1)
                {
                    for(int dum=0; dum < (resVar.nav.nx*resVar.nav.ny); dum++)
                    {
                        dummy[dum] = resVar.data.data[dum+nxtny+nxtny];
                    }
                    dataVar33.putVar(dummy);
                }
            }
            delete [] dummy;
        } else
        {
            dataVar3.putVar(resVar.dataSValfloat);
            if(args.doc2 == 1) dataVar32.putVar(resVar.dataSValfloat2);
            if(args.doc3 == 1) dataVar33.putVar(resVar.dataSValfloat3);
        }


        gipVar.putVar(&resVar.nav.gipVal);
        return 0;
   }
   catch(NcException& e)
     {
      e.what();
      return NC_ERR;
   }
}
//writes mercator grid netcdf files
int oct_mercwrite (string fpath,GOESVar &resVar,OFFlags args)
{
    //This is a function designed for reading GOES data files
    //int r = 1;
    try
    {
        NcFile ncf(fpath, NcFile::replace);
        NcVar dataVar22, dataVar23,dataVar3;
        NcDim xDim = ncf.addDim("x",resVar.nav.nx);
        NcDim yDim = ncf.addDim("y",resVar.nav.ny);

        //now create the variable
        NcVar xVar = ncf.addVar("x",ncShort, xDim);
        NcVar yVar = ncf.addVar("y",ncShort, yDim);
        xVar.putAtt("scale_factor",NC_FLOAT,resVar.nav.xScale);
        xVar.putAtt("add_offset",NC_FLOAT,resVar.nav.xOffset);
        yVar.putAtt("scale_factor",NC_FLOAT,resVar.nav.yScale);
        yVar.putAtt("add_offset",NC_FLOAT,resVar.nav.yOffset);

        xVar.putVar(resVar.x);
        yVar.putVar(resVar.y);

        //add the time variables
        NcVar tVar = ncf.addVar("t",ncDouble);
        tVar.putAtt("standard_name","time");
        tVar.putAtt("units",resVar.tUnits);
        tVar.putAtt("axis","T");
        tVar.putAtt("bounds","time_bounds");
        tVar.putAtt("long_name" , "J2000 epoch mid-point between the start and end image scan in seconds");

        tVar.putVar(&resVar.t);


        vector<NcDim> dims;
        dims.push_back(yDim);
        dims.push_back(xDim);

        NcVar dataVar = ncf.addVar("U",ncDouble,dims);
        NcVar dataVar2 = ncf.addVar("V",ncDouble,dims);

       if(args.pixuv==1)
        {
            dataVar22 = ncf.addVar("Upix",ncFloat,dims);
            dataVar23 = ncf.addVar("Vpix",ncFloat,dims);
        }
        if(args.outrad) dataVar3 = ncf.addVar("Rad",ncFloat,dims);
        NcVar gipVar = ncf.addVar("merc_imager_projection",NC_INT);
        NcVar ofVar = ncf.addVar("optical_flow_settings",NC_INT);

        dataVar.putAtt("long_name","U");
        dataVar.putAtt("grid_mapping","Mercator Sphere");
        dataVar.putAtt("scale_factor",NC_FLOAT,0.01);
        if(args.pixuv == 0)
        {
            dataVar.putAtt("units","meters per second");
        }else{
            dataVar.putAtt("units","x-pixels");
        }


        dataVar2.putAtt("long_name","V");
        dataVar2.putAtt("grid_mapping","Mercator Sphere");
        dataVar2.putAtt("scale_factor",NC_FLOAT,0.01);

        if(args.pixuv == 1)
        {
            dataVar2.putAtt("units","y-pixels");
            if(args.dosrsal == 1)
            {
                dataVar22.putAtt("long_name","Upix");
                dataVar23.putAtt("long_name","Vpix");
            }
        }else{
            dataVar2.putAtt("units","meters per second");
        }




        if(args.outrad)
        {
            dataVar3.putAtt("long_name","Rad");
            dataVar3.putAtt("grid_mapping","Mercator Sphere");
        }
        gipVar.putAtt("long_name" , "Mercator_Grid");
        gipVar.putAtt("grid_mapping_name" , "Mercator");
        double lon1val = resVar.nav.lon1;
        double Rval = resVar.nav.R;
        gipVar.putAtt("lon1",NC_DOUBLE,lon1val);
        gipVar.putAtt("R",NC_DOUBLE,Rval);

        ofVar.putAtt("long_name" , "Optical Flow Settings");
        ofVar.putAtt("key" , "1 = Modified Sun (2014), 2 = Farneback, 3 = Brox (2004)");
        if(args.oftype==1)
        {
            ofVar.putAtt("lambda",NC_DOUBLE,args.lambda);
            ofVar.putAtt("lambdac",NC_DOUBLE,args.lambdac); //hinting term weight (0 if dofirstguess==0)
            ofVar.putAtt("alpha", NC_DOUBLE,args.alpha);
            ofVar.putAtt("filtsigma", NC_DOUBLE,args.filtsigma);
            ofVar.putAtt("ScaleF", NC_DOUBLE,args.scaleF);
            ofVar.putAtt("K_Iterations", NC_INT,args.kiters);
            ofVar.putAtt("L_Iterations", NC_INT,args.liters);
            ofVar.putAtt("M_Iterations", NC_INT,args.miters);
            ofVar.putAtt("CG_Iterations", NC_INT,args.cgiters);
            ofVar.putAtt("NormMax", NC_FLOAT,args.NormMax);
            ofVar.putAtt("NormMin", NC_FLOAT,args.NormMin);
            ofVar.putAtt("dofirstguess", NC_INT,args.dofirstguess);
        }
        if(args.oftype==2)
        {
            ofVar.putAtt("pyr_scale",NC_FLOAT,args.fpyr_scale);
            ofVar.putAtt("levels", NC_INT,args.flevels);
            ofVar.putAtt("winsize", NC_INT,args.fwinsize);
            ofVar.putAtt("iterations", NC_INT,args.fiterations);
            ofVar.putAtt("poly_n", NC_INT,args.poly_n);
            ofVar.putAtt("poly_sigma", NC_FLOAT,args.poly_sigma);
            ofVar.putAtt("use_initial_flow", NC_INT,args.uif);
            ofVar.putAtt("farneback_gaussian", NC_INT,args.fg);
            ofVar.putAtt("NormMax", NC_FLOAT,args.NormMax);
            ofVar.putAtt("NormMin", NC_FLOAT,args.NormMin);
        }
        ofVar.putAtt("dt_seconds",NC_FLOAT,resVar.dT);

        dataVar.putVar(resVar.uVal);
        dataVar2.putVar(resVar.vVal);
        if(args.pixuv == 1)
        {
            dataVar22.putVar(resVar.uPix);
            dataVar23.putVar(resVar.vPix);
        }
        if(args.outrad) dataVar3.putVar(resVar.data.data);
        gipVar.putVar(&resVar.nav.gipVal);
        return 0;
   }
   catch(NcException& e)
     {
      e.what();
      return NC_ERR;
   }
}


int oct_filewrite (string fpath,string ftype, GOESVar &resVar,OFFlags args)
{
    int t;
    if(ftype=="GOES") t = oct_goeswrite (fpath,resVar,args);
    if(ftype=="POLAR") t = oct_polarwrite (fpath,resVar,args);
    if(ftype=="MERC") t = oct_mercwrite (fpath,resVar,args);

    return 0;
}
