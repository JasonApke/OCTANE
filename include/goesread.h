//A header containing class definitions for GOESREAD

class GOESNAVVar
{
    public:
        double pph,req,rpol,lam0,inverse_flattening,lat0;
        float gipVal,xScale,xOffset,yScale,yOffset,g2xOffset,g2yOffset,fk1,fk2,bc1,bc2,lpo,kap1,radScale,radOffset;
        float fk12,fk22,bc12,bc22,kap12,fk13,fk23,bc13,bc23,kap13;
        float radScale2, radScale3, radOffset2,radOffset3;
        long nx2,ny2,nx3,ny3; 
        long nx,ny,CTHx,CTHy;
        int minXc,maxXc,minYc,maxYc,minX,minY,maxX,maxY;
        float lat1,lon1,lon0,R;
};

class GOESVar
{
    //Access Specifier
    public:
        float *latVal;
        float *lonVal;
        short *x;
        short *y;
        short *CTP,*CTT;
        unsigned char *CTI;
        float *dataVal, *dataVal2, *dataVal3;
        Image data; 
        float *dataVal2i, *dataVal3i;
        short *occlusion;
        short *uVal;
        short *vVal;
        short *uVal2;
        short *vVal2;
        short *cnrarr;
        float *uPix;
        float *vPix;
        double *u1;
        double *v1;
        float *UFG;
        float *VFG;
        double *u2;
        double *v2;
        short *accel;
        float *CTHVal;
        float *CTTVal;
        unsigned char *CTHInv;
        short *dataSVal,*dataSVal2, *dataSVal3;
        short *dataSValint,*dataSValint2,*dataSValint3;
        float *dataSValfloat,*dataSValfloat2,*dataSValfloat3;
        double t;
        double tint;
        float dT; 
        float frdt; 
        int band,band2,band3;
        GOESNAVVar nav;
        std::string tUnits;
};
