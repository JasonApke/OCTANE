#include <iostream>
#include <netcdf>
#include <math.h>
#include "image.h"
#include "goesread.h"
#include "offlags.h"
#include "zoom.h"
using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;

//Function: jma_goesread
//Purpose: This is a C++ function which reads, navigates, and calibrates GOES-R netcdf4 data
//         Now with added functions to read/navigate polar orthonormal/mercator
//Requires: The netcdf C++ library
//Inputs: fpath- a full path to the netcdf file containing GOES-R Imagery
//        cal- A setting for callibration, default is RAW, options include BRIT, TEMP, and RAW
//        donav- set to 1 to return latitude and longitude arrays with the object, otherwise they are set to 0
//        The reason I have this as an option is for speed purposes, navigation can slow down optical flow code if it is not needed
//Returns: A structure containing the calbrated GOES data, with Latitude and Longitude files from navigation
//
//Author: Jason Apke, Updated 9/10/2018
void oct_navcal_cuda(short *,short *, short *, short *, short *, short *, int,
                     int, int, int, int, int, float *, float *,
                     float *,string,int,float, float, float, 
                     float, float, float, float, float, 
                     float, float,float,float,float,float,
                     float,float,float,float,float,int,OFFlags);
void oct_polar_navcal_cuda(float *,short *, short *, short *, short *, short *, int,
                     int, int, int, int, int, float *, float *,
                     float *,float, float, float,
                     float, float, float,
                     float,int,int,OFFlags);
void oct_merc_navcal_cuda(float *,short *, short *, short *, short *, short *, int,
                     int, int, int, int, int, float *, float *,
                     float *,float, float, float,
                     float, float,
                     float,int,OFFlags);
void oct_bandminmax(int, float &, float &);


static const int NC_ERR = 2;
int oct_goesread (string fpath,string cal,int donav,int channelnum, GOESVar &resVar,OFFlags &args)
{
    //This is a function designed for reading GOES data files
    using namespace std;
    const double PI=3.14159265359;
    const double DTOR = PI/180.;

    long nv;
    long xdimsize,ydimsize;
    int datf=0;
    float *lat,*lon,*data3;
    short *data2;
    short *y;
    short *x;
    short *data2s;
    short *ys;
    short *xs;
    int band;
    float xScale,yScale,xOffset,yOffset,lpo;
    float radScale,radOffset;
    float req,rpol,pph,lam0,inverse,lat0;
    float fk1,fk2,bc1,bc2,kap1;
    float gipv;
    string tUnitString;
    NcVarAtt reqVar;
    try
    {
        //open the file
        NcFile dataFile(fpath, NcFile::read);
        NcVar xVar, yVar,dataVar,gipVar,tVar,bandVar;
        int xv = dataFile.getVarCount();
        multimap< string, NcDim > xxv=dataFile.getDims();
        NcDim ydim=xxv.find("y")->second;
        NcDim xdim=xxv.find("x")->second;

        ydimsize=ydim.getSize();
        xdimsize=xdim.getSize();
        nv = xdimsize*ydimsize;
        data2= new short[nv];
        if(!data2){
            cout << "Memory Allocation Failed\n";
            exit(0);
        }

        x = new short[xdimsize];
        if(!x){
            cout << "Memory Allocation Failed y\n";
            exit(0);
        }
        y = new short[ydimsize];
        if(!y){
            cout << "Memory Allocation Failed y\n";
            exit(0);
        }
        

        dataVar=dataFile.getVar("Rad");
        yVar = dataFile.getVar("y");
        xVar = dataFile.getVar("x");
        tVar = dataFile.getVar("t");
        bandVar = dataFile.getVar("band_id");
        reqVar=dataVar.getAtt("scale_factor");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(&radScale);
        if(channelnum == 1){
            resVar.nav.radScale = radScale;
        }
        if(channelnum == 2){
            resVar.nav.radScale2 = radScale;
        }
        if(channelnum == 3){
            resVar.nav.radScale3 = radScale;
        }
        reqVar=dataVar.getAtt("add_offset");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(&radOffset);
        if(channelnum == 1){
            resVar.nav.radOffset = radOffset;
        }
        if(channelnum == 2){
            resVar.nav.radOffset2 = radOffset;
        }
        if(channelnum == 3){
            resVar.nav.radOffset3 = radOffset;
        }
        if(channelnum == 1)
        {
            reqVar=yVar.getAtt("scale_factor");
            if (reqVar.isNull()) return NC_ERR;
            reqVar.getValues(&yScale);
            resVar.nav.yScale=yScale;
            reqVar=yVar.getAtt("add_offset");
            if (reqVar.isNull()) return NC_ERR;
            reqVar.getValues(&yOffset);
            resVar.nav.yOffset=yOffset;
            reqVar=xVar.getAtt("scale_factor");
            if (reqVar.isNull()) return NC_ERR;
            reqVar.getValues(&xScale);
            resVar.nav.xScale=xScale;
            reqVar=xVar.getAtt("add_offset");
            if (reqVar.isNull()) return NC_ERR;
            reqVar.getValues(&xOffset);
            resVar.nav.xOffset = xOffset;
            reqVar=tVar.getAtt("units");
            if (reqVar.isNull()) return NC_ERR;
            reqVar.getValues(tUnitString);
            resVar.tUnits=tUnitString;
            gipVar = dataFile.getVar("goes_imager_projection");
            gipVar.getVar(&gipv);
            resVar.nav.gipVal=gipv;
            
            reqVar=gipVar.getAtt("longitude_of_projection_origin");
            if (reqVar.isNull()) return NC_ERR;
            reqVar.getValues(&lpo);
            resVar.nav.lpo=lpo;

            reqVar=gipVar.getAtt("semi_major_axis");
            if (reqVar.isNull()) return NC_ERR;
            reqVar.getValues(&req);
            resVar.nav.req = req;
            reqVar=gipVar.getAtt("semi_minor_axis");
            if (reqVar.isNull()) return NC_ERR;
            reqVar.getValues(&rpol);
            resVar.nav.rpol=rpol;
            reqVar=gipVar.getAtt("inverse_flattening");
            if (reqVar.isNull()) return NC_ERR;
            reqVar.getValues(&inverse);
            resVar.nav.inverse_flattening=inverse;
            reqVar=gipVar.getAtt("latitude_of_projection_origin");
            if (reqVar.isNull()) return NC_ERR;
            reqVar.getValues(&lat0);
            resVar.nav.lat0=lat0;
            reqVar=gipVar.getAtt("perspective_point_height");
            if (reqVar.isNull()) return NC_ERR;
            reqVar.getValues(&pph);
            resVar.nav.pph=pph;
            reqVar=gipVar.getAtt("longitude_of_projection_origin");
            if (reqVar.isNull()) return NC_ERR;
            reqVar.getValues(&lam0);
            lam0 = lam0*DTOR;
            resVar.nav.lam0=lam0;
        }
        NcVar fk1Var=dataFile.getVar("planck_fk1");
        fk1Var.getVar(&fk1);
        if(fk1Var.isNull()) return NC_ERR;
        if(channelnum == 1)
        {
            resVar.nav.fk1=fk1;
        }
        if(channelnum == 2)
        {
            resVar.nav.fk12 = fk1;
        }
        if(channelnum == 3)
        {
            resVar.nav.fk13 = fk1;
        }

        NcVar fk2Var=dataFile.getVar("planck_fk2");
        fk2Var.getVar(&fk2);
        if(fk2Var.isNull()) return NC_ERR;
        if(channelnum == 1)
        {
            resVar.nav.fk2=fk2;
        }
        if(channelnum == 2)
        {
            resVar.nav.fk22 = fk2;
        }
        if(channelnum == 3)
        {
            resVar.nav.fk23 = fk2;
        }

        NcVar bc1Var=dataFile.getVar("planck_bc1");
        bc1Var.getVar(&bc1);
        if(bc1Var.isNull()) return NC_ERR;
        if(channelnum == 1)
        {
            resVar.nav.bc1=bc1;
        }
        if(channelnum == 2)
        {
            resVar.nav.bc12 = bc1;
        }
        if(channelnum == 3)
        {
            resVar.nav.bc13 = bc1;
        }

        NcVar bc2Var=dataFile.getVar("planck_bc2");
        bc2Var.getVar(&bc2);
        if(bc2Var.isNull()) return NC_ERR;
        if(channelnum == 1)
        {
            resVar.nav.bc2=bc2;
        }
        if(channelnum == 2)
        {
            resVar.nav.bc22 = bc2;
        }
        if(channelnum == 3)
        {
            resVar.nav.bc23 = bc2;
        }
        
        NcVar kap1Var=dataFile.getVar("kappa0");
        kap1Var.getVar(&kap1);
        if(kap1Var.isNull()) return NC_ERR;
        if(channelnum == 1)
        {
            resVar.nav.kap1=kap1;
        }
        if(channelnum == 2)
        {
            resVar.nav.kap12 = kap1;
        }
        if(channelnum == 3)
        {
            resVar.nav.kap13 = kap1;
        }
        float H = pph+req;
	    float x1v, y1v, x1v2,y1v2,x1v3,y1v3,x1v4,y1v4;
        //This looks like strange notation because there used to be a subsetter here
        //I am now moving that out of octane
        int minx,maxx,miny,maxy;
        minx = 0;
        maxx = xdimsize;
        miny = 0;
        maxy = ydimsize;
        //now get them in CLAVR-x coords

        if(channelnum == 1)
        {
			int nc = 1;
			if(args.doc2 == 1) nc++;
			if(args.doc3 == 1) nc++;

            resVar.data.setdims(maxx-minx,maxy-miny,nc);
            resVar.data.data = new float[(maxx-minx)*(maxy-miny)*nc];
        } else{
            data3= new float[nv];  // we need a dummy array for the zoom function
        }
        lat = new float[nv];
        lon = new float[nv];
        xs = new short[maxx-minx];
        ys = new short[maxy-miny];
        data2s = new short[nv];
        if(channelnum == 1)
        {
            resVar.nav.nx=(maxx-minx);
            resVar.nav.ny=(maxy-miny);
        }
        if(channelnum == 2)
        {
            resVar.nav.nx2=(maxx-minx);
            resVar.nav.ny2=(maxy-miny);
        }
        if(channelnum == 3)
        {
            resVar.nav.nx3=(maxx-minx);
            resVar.nav.ny3=(maxy-miny);
        }
        yVar.getVar(y);
        xVar.getVar(x);
        if(channelnum == 1) tVar.getVar(&resVar.t);
        dataVar.getVar(data2);
        bandVar.getVar(&band);
        if(dataVar.isNull()) return NC_ERR;
        if(bandVar.isNull()) return NC_ERR;
        if(yVar.isNull()) return NC_ERR;
        if(xVar.isNull()) return NC_ERR;
        if(channelnum ==1)
        {
            if(band == 2)
            {
                resVar.nav.minXc = minx/4;
                resVar.nav.minYc = miny/4;
                resVar.nav.maxXc = maxx/4;
                resVar.nav.maxYc = maxy/4;
            } else if(band == 1 || band == 3)
            {
                resVar.nav.minXc = minx/2;
                resVar.nav.minYc = miny/2;
                resVar.nav.maxXc = maxx/2;
                resVar.nav.maxYc = maxy/2;
            } else
            {
                resVar.nav.minXc = minx;
                resVar.nav.minYc = miny;
                resVar.nav.maxXc = maxx;
                resVar.nav.maxYc = maxy;
            }
        }
        resVar.nav.minX = minx;
        resVar.nav.minY = miny;
        resVar.nav.maxX = maxx;
        resVar.nav.maxY = maxy;
        float maxch, minch;
        float minout = 0.;
        float maxout = 255.;
        oct_bandminmax(band,maxch,minch);
        if(channelnum == 1)
        {
            if(args.setNormMax) args.NormMax = maxch;
            if(args.setNormMin) args.NormMin = minch;
        }
        if(channelnum == 2)
        {
            if(args.setNormMax2) args.NormMax2 = maxch;
            if(args.setNormMin2) args.NormMin2 = minch;
        }
        if(channelnum == 3)
        {
            if(args.setNormMax3) args.NormMax3 = maxch;
            if(args.setNormMin3) args.NormMin3 = minch;
        }

        if(channelnum > 1){
            oct_navcal_cuda(data2,data2s, x, y, xs, ys, xdimsize,
                            ydimsize, minx, maxx, miny, maxy, data3, lat,
                            lon,cal,datf,xScale,xOffset,yScale, 
                            yOffset,radScale,radOffset,rpol,req, 
                            H,lam0,fk1,fk2,bc1,bc2,
                            kap1,maxch,minch,maxout,minout,donav,args);
            if(resVar.nav.nx > (maxx-minx))
            {
                oct_zoom_in_float(data3, resVar.data.data,(maxx-minx),(maxy-miny),resVar.nav.nx,resVar.nav.ny,channelnum-1,1);
            } else{
                double factor = (double) resVar.nav.nx / ((double) (maxx-minx));
                double factor2 = (double)  resVar.nav.ny / ((double) (maxy-miny));
                if(pow(factor-factor2,2) > 0.000001)
                {
                    printf("Image x and y dimensions not compatable for scaling (factor not the same), exiting");
                    exit(0);
                }
                oct_zoom_out_float(data3,resVar.data.data,(maxx-minx),(maxy-miny),factor,0,channelnum-1);
            }
        } else{

            oct_navcal_cuda(data2,data2s, x, y, xs, ys, xdimsize,
                            ydimsize, minx, maxx, miny, maxy, resVar.data.data, lat,
                            lon,cal,datf,xScale,xOffset,yScale, 
                            yOffset,radScale,radOffset,rpol,req, 
                            H,lam0,fk1,fk2,bc1,bc2,
                            kap1,maxch,minch,maxout,minout,donav,args);
        }


        if(channelnum == 1){
            resVar.latVal = lat;
            resVar.lonVal = lon;
            resVar.x = xs;
            resVar.y = ys;
            resVar.dataSVal = data2s;
            resVar.band =(int)band;
        }
        if(channelnum == 2){
            resVar.band2 =(int)band;
        }
        if(channelnum == 3){
            resVar.band3 =(int)band;
        }
        delete [] data2;
        delete [] x;
        delete [] y;
        if(channelnum > 1) delete [] data3;

    }catch(NcException& e)
        {
            e.what();
            cout<<"OCT_GOESREAD FAILURE, CHECK THAT ALL VARIABLES AND ATTS EXIST"<<endl;
            exit(1);
            return NC_ERR;
        }
    return 1;
}

int oct_polarread (string fpath,string cal,int donav,int channelnum, GOESVar &resVar,OFFlags &args)
{
    //This is a function designed for reading GOES data files
    using namespace std;

    long nv;
    long xdimsize,ydimsize;
    float *lat,*lon,*data3int,*data3int2,*data3int3;
    float *data2;
    short *y;
    short *x;
    short *data2s;
    short *ys;
    short *xs;
    int band;
    float xScale,yScale,xOffset,yOffset;
    float lat1,lon0,R;
    int gipv;
    string tUnitString;
    NcVarAtt reqVar;
    try
    {
        NcFile dataFile(fpath, NcFile::read);
        NcVar xVar, yVar,dataVar,gipVar,tVar,bandVar;
        int xv = dataFile.getVarCount();
        multimap< string, NcDim > xxv=dataFile.getDims();
        NcDim ydim=xxv.find("y")->second;
        NcDim xdim=xxv.find("x")->second;

        ydimsize=ydim.getSize();
        xdimsize=xdim.getSize();
        nv = xdimsize*ydimsize;
        data2= new float[nv];
        if(!data2){
            cout << "Memory Allocation Failed\n";
            exit(0);
        }

        x = new short[xdimsize];
        if(!x){
            cout << "Memory Allocation Failed y\n";
            exit(0);
        }
        y = new short[ydimsize];
        if(!y){
            cout << "Memory Allocation Failed y\n";
            exit(0);
        }


        dataVar=dataFile.getVar("Rad");
        yVar = dataFile.getVar("y");
        xVar = dataFile.getVar("x");
        tVar = dataFile.getVar("t");
        reqVar=yVar.getAtt("scale_factor");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(&yScale);
        resVar.nav.yScale=yScale;
        reqVar=yVar.getAtt("add_offset");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(&yOffset);
        resVar.nav.yOffset=yOffset;
        reqVar=xVar.getAtt("scale_factor");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(&xScale);
        resVar.nav.xScale=xScale;
        reqVar=xVar.getAtt("add_offset");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(&xOffset);
        resVar.nav.xOffset = xOffset;
        reqVar=tVar.getAtt("units");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(tUnitString);
        resVar.tUnits=tUnitString;
        gipVar = dataFile.getVar("grid_mapping");
        gipVar.getVar(&gipv);
        resVar.nav.gipVal=(float) gipv;

        reqVar=gipVar.getAtt("lat1");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(&lat1);
        reqVar=gipVar.getAtt("lon0");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(&lon0);
        reqVar=gipVar.getAtt("R");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(&R);

        resVar.nav.lat1=lat1;
        resVar.nav.lon0=lon0;
        resVar.nav.R=R;
        float x1v, y1v, x1v2,y1v2,x1v3,y1v3,x1v4,y1v4;
        int minx, maxx, miny,maxy;
        minx = 0;
        maxx = xdimsize;
        miny = 0;
        maxy = ydimsize;

        if(channelnum == 1)
        {
			int nc = 1;
			if(args.doc2 == 1) nc++;
			if(args.doc3 == 1) nc++;

            resVar.data.setdims(maxx-minx,maxy-miny,nc);
            resVar.data.data = new float[(maxx-minx)*(maxy-miny)*nc];
        }
        data3int= new float[nv];
        if(args.doc2 == 1) data3int2= new float[nv];
        if(args.doc3 == 1) data3int3= new float[nv];
        lat = new float[nv];
        lon = new float[nv];
        xs = new short[maxx-minx];
        ys = new short[maxy-miny];
        data2s = new short[nv];
        if(channelnum == 1)
        {
            resVar.nav.nx=(maxx-minx);
            resVar.nav.ny=(maxy-miny);
        }
        if(channelnum == 2)
        {
            resVar.nav.nx2=(maxx-minx);
            resVar.nav.ny2=(maxy-miny);
        }
        if(channelnum == 3)
        {
            resVar.nav.nx3=(maxx-minx);
            resVar.nav.ny3=(maxy-miny);
        }
        yVar.getVar(y);
        xVar.getVar(x);
        if(channelnum == 1) tVar.getVar(&resVar.t);
        dataVar.getVar(data2);
        if(dataVar.isNull()) return NC_ERR;
        if(yVar.isNull()) return NC_ERR;
        if(xVar.isNull()) return NC_ERR;
        if(channelnum == 1)
        {
            resVar.nav.minXc = minx;
            resVar.nav.minYc = miny;
            resVar.nav.maxXc = maxx;
            resVar.nav.maxYc = maxy;
        }
        resVar.nav.minX = minx;
        resVar.nav.minY = miny;
        oct_polar_navcal_cuda(data2,data2s, x,y,xs,ys, xdimsize,
                     ydimsize,minx,maxx,miny,maxy, resVar.data.data, lat,
                     lon,xScale,xOffset,yScale,yOffset, lon0, lat1,
                     R,donav,channelnum,args);
        if(channelnum == 1)
        {
            resVar.latVal = lat;
            resVar.lonVal = lon;
            resVar.x = xs;
            resVar.y = ys;
            resVar.dataSVal = data2s;
            if(args.dointerp==1)
            {   
                resVar.dataSValfloat = data3int;
            }
            resVar.band =(int)band;
        }
        if(channelnum == 2)
        {
            if(args.dointerp==1)
            {
                resVar.dataSValfloat2 = data3int2;
            }
        }
        if(channelnum == 3)
        {
            if(args.dointerp==1)
            {
                resVar.dataSValfloat3 = data3int3;
            }
        }

        delete [] data2;
        delete [] x;
        delete [] y;
    }catch(NcException& e)
        {
            e.what();
            cout<<"OCT_POLARREAD FAILURE, CHECK THAT ALL VARIABLES AND ATTS EXIST"<<endl;
            exit(1);
            return NC_ERR;
        }
    return 1;
}
int oct_mercread (string fpath,string cal,int donav,GOESVar &resVar,OFFlags &args)
{
    //This is a function designed for reading Mercator Netcdfs
    using namespace std;

    long nv;
    long xdimsize,ydimsize;
    float *lat,*lon,*data3;
    float *data2;
    short *y;
    short *x;
    short *data2s;
    short *ys;
    short *xs;
    int band;
    float xScale,yScale,xOffset,yOffset;
    float lon1,R;
    int gipv;
    string tUnitString;
    NcVarAtt reqVar;
    try
    {
        NcFile dataFile(fpath, NcFile::read);
        NcVar xVar, yVar,dataVar,gipVar,tVar,bandVar;
        int xv = dataFile.getVarCount();
        multimap< string, NcDim > xxv=dataFile.getDims();
        NcDim ydim=xxv.find("y")->second;
        NcDim xdim=xxv.find("x")->second;

        ydimsize=ydim.getSize();
        xdimsize=xdim.getSize();
        nv = xdimsize*ydimsize;
        data2= new float[nv];
        if(!data2){
            cout << "Memory Allocation Failed\n";
            exit(1);
        }

        x = new short[xdimsize];
        if(!x){
            cout << "Memory Allocation Failed y\n";
            exit(1);
        }
        y = new short[ydimsize];
        if(!y){
            cout << "Memory Allocation Failed y\n";
            exit(1);
        }
        dataVar=dataFile.getVar("Rad");
        yVar = dataFile.getVar("y");
        xVar = dataFile.getVar("x");
        tVar = dataFile.getVar("t");
        reqVar=yVar.getAtt("scale_factor");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(&yScale);
        resVar.nav.yScale=yScale;
        reqVar=yVar.getAtt("add_offset");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(&yOffset);
        resVar.nav.yOffset=yOffset;
        reqVar=xVar.getAtt("scale_factor");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(&xScale);
        resVar.nav.xScale=xScale;
        reqVar=xVar.getAtt("add_offset");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(&xOffset);
        resVar.nav.xOffset = xOffset;
        reqVar=tVar.getAtt("units");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(tUnitString);
        resVar.tUnits=tUnitString;
        gipVar = dataFile.getVar("grid_mapping");
        gipVar.getVar(&gipv);
        resVar.nav.gipVal=(float) gipv;

        reqVar=gipVar.getAtt("lon1");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(&lon1);

        reqVar=gipVar.getAtt("R");
        if (reqVar.isNull()) return NC_ERR;
        reqVar.getValues(&R);

        resVar.nav.lon1=lon1;
        resVar.nav.R=R;
        float x1v, y1v, x1v2,y1v2,x1v3,y1v3,x1v4,y1v4;
        int minx, maxx, miny,maxy;
        minx = 0;
        maxx = xdimsize;
        miny = 0;
        maxy = ydimsize;



        resVar.data.setdims(maxx-minx,maxy-miny,1);
        resVar.data.data = new float[(maxx-minx)*(maxy-miny)*1];
        lat = new float[nv];
        lon = new float[nv];
        xs = new short[maxx-minx];
        ys = new short[maxy-miny];
        data2s = new short[nv];
        resVar.nav.nx=(maxx-minx);
        resVar.nav.ny=(maxy-miny);
        yVar.getVar(y);
        xVar.getVar(x);
        tVar.getVar(&resVar.t);
        dataVar.getVar(data2);
        if(dataVar.isNull()) return NC_ERR;
        if(yVar.isNull()) return NC_ERR;
        if(xVar.isNull()) return NC_ERR;
        resVar.nav.minXc = minx;
        resVar.nav.minYc = miny;
        resVar.nav.maxXc = maxx;
        resVar.nav.maxYc = maxy;
        resVar.nav.minX = minx;
        resVar.nav.minY = miny;
        oct_merc_navcal_cuda(data2,data2s, x,y,xs,ys, xdimsize,
                     ydimsize,minx,maxx,miny,maxy, resVar.data.data, lat,
                     lon,xScale,xOffset,yScale,yOffset, lon1,
                     R,donav,args);

        resVar.latVal = lat;
        resVar.lonVal = lon;
        resVar.x = xs;
        resVar.y = ys;
        resVar.dataSVal = data2s;
        resVar.band =(int)band;
        delete [] data2;
        delete [] x;
        delete [] y;

    }catch(NcException& e)
        {
            e.what();
            cout<<"OCT_MERCREAD FAILURE, CHECK THAT ALL VARIABLES AND ATTS EXIST"<<endl;
            exit(1);
            return NC_ERR;
        }



    return 1;
}

int oct_clavrxread (string fpath,GOESVar &resVar,OFFlags &args)
{
    using namespace std;

    long nv;
    long xdimsize,ydimsize;
    int xmin,xmax,ymin,ymax;
    float *data3;
    try
    {
        //open the file
        NcFile dataFile(fpath, NcFile::read);
        NcVar dataVar,dataVarInv,dataVarTemp;
        int xv = dataFile.getVarCount();
        multimap< string, NcDim > xxv=dataFile.getDims();
        NcDim ydim=xxv.find("ny")->second;
        NcDim xdim=xxv.find("nx")->second;

        ydimsize=ydim.getSize();
        xdimsize=xdim.getSize();
        nv = xdimsize*ydimsize;
        xmax = resVar.nav.maxXc;
        xmin = resVar.nav.minXc;
        ymax = resVar.nav.maxYc;
        ymin = resVar.nav.minYc;

        data3= new float[nv];
        if(!data3){
            cout << "Memory Allocation Failed\n";
            exit(1);
        }


        resVar.nav.CTHx=xmax-xmin;
        resVar.nav.CTHy=ymax-ymin;
        dataVar=dataFile.getVar("Cloud_Top_Height_Effective");
        dataVar.getVar(data3);  
        resVar.CTHVal = new float[resVar.nav.nx*resVar.nav.ny];
        if(resVar.nav.nx > (xmax-xmin))
        {
            oct_zoom_in_float(data3, resVar.CTHVal,(xmax-xmin),(ymax-ymin),resVar.nav.nx,resVar.nav.ny,0,args.interpcth);
        } else{
            double factor = (double) resVar.nav.nx / ((double) (xmax-xmin));
            double factor2 = (double)  resVar.nav.ny / ((double) (ymax-ymin));
            if(pow(factor-factor2,2) > 0.000001)
            {
                printf("Image x and y dimensions not compatable for scaling (factor not the same), CTH data problem");
                exit(0);
            }
            oct_zoom_out_float(data3,resVar.CTHVal,(xmax-xmin),(ymax-ymin),factor,0,0);
        }
        delete [] data3;
    }catch(NcException& e)
    {
        e.what();
        cout<<"OCT_CLAVRXREAD FAILURE, CHECK THAT ALL VARIABLES AND ATTS EXIST"<<endl;
        return NC_ERR;
    }

    return 1;
}
int oct_fgread (string fpath,GOESVar &resVar,OFFlags &args)
{
    using namespace std;

    long nv;
    long xdimsize,ydimsize;
    int xmin,xmax,ymin,ymax;
    float *data3;
    float *data4;
    try
    {
        //open the file
        NcFile dataFile(fpath, NcFile::read);
        NcVar dataVar,dataVarInv,dataVarTemp;
        int xv = dataFile.getVarCount();
        multimap< string, NcDim > xxv=dataFile.getDims();
        NcDim ydim=xxv.find("ny")->second;
        NcDim xdim=xxv.find("nx")->second;

        ydimsize=ydim.getSize();
        xdimsize=xdim.getSize();
        nv = xdimsize*ydimsize;
        xmax = resVar.nav.maxX;
        xmin = resVar.nav.minX;
        ymax = resVar.nav.maxY;
        ymin = resVar.nav.minY;

        data3= new float[nv];
        if(!data3){
            cout << "Memory Allocation Failed\n";
            exit(1);
        }
        data4= new float[nv];
        if(!data4){
            cout << "Memory Allocation Failed\n";
            exit(1);
        }
        dataVar=dataFile.getVar("UFG");
        dataVar.getVar(data3);
        dataVar=dataFile.getVar("VFG");
        dataVar.getVar(data4);
        resVar.uPix= data3;
        resVar.vPix= data4;
    }catch(NcException& e)
    {
        e.what();
        cout<<"OCT_FGREAD FAILURE, CHECK THAT ALL VARIABLES AND ATTS EXIST"<<endl;
        return NC_ERR;
    }

    return 1;
}

//Here is the wrapper for all the files to read
int oct_fileread(string fpath,string ftype,string cal,int donav,int channelnum, GOESVar &resVar,OFFlags &args)
{
    int t;
    if(ftype == "GOES")
    {
        t = oct_goesread(fpath,"RAW",donav,channelnum,resVar,args);
    }
    if(ftype == "POLAR")
    {
        t = oct_polarread(fpath,"RAW",donav,channelnum,resVar,args);
    }
    if(ftype == "MERC")
    {
        t = oct_mercread(fpath,"RAW",donav,resVar,args);
    }
    if(ftype == "CLAVRX")
    {
        t = oct_clavrxread(fpath,resVar,args);
    }
    if(ftype == "FIRSTGUESS")
    {
        t = oct_fgread(fpath,resVar,args);
    }
    return 1;
}
