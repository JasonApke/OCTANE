//A header containing class definitions and default settings for OF Flags
#include <string>

class OFFlags
{
    //Access Specifier
    public:
        int farn; //Do farneback optical flow instead 
        int pixuv; //Output pixel displacement instead of UV displacement
        int dopolar; //Do a polar grid (this is an orthonormal grid I designed in convert_dnbh5 and polar_grid_module.py)
        int domerc; //Do a Mercator grid (this is a mercator grid that comes out of my read_alpw.py module)
        int doahi; //do AHI data
        int dosrsal;
        int dososm;
        int dofirstguess; //an arguement to tell optical flow to use a first guess flow field (MUST HAVE NETCDF OF SAME SIZE AS INPUT FILE!!!!!!!!!!)
        std::string ftype;
        int dointerp; //the arg to tell optical flow to interpolate
        int docorn; //An argument to return corners in the image (used for image navigation for winds products)
        int putinterp; //an extra argument to handle interp writing portion, not set by user
        int interpcth; //A flag to switch CTH (or IR) between bilinear and nearest neighbor interpolation 
                       //for zooming in (to visible resolution, doesn't matter for IR resolution)
        int doinv; //an argument to read the inversion flag data out of the clavrx files, and output it with the OF file
        int doctt; //an argument to read the cloud-top temperature flag data out of the clavrx files and output it with the OF file
        int dozim; //a flag to turn on zimmerman etal (2011) data term normalization 
        int oftype;
        int doc2;
        int doc3;
        int ir;
        int rad; //sosm target size
        int srad; //sosm search radius size
        int setdevice; //integer to set which gpu to use (useful for multi-gpu machines)
        //Farneback Defaults
        float fpyr_scale;
        float flevels;
        int fwinsize;
        int fiterations;
        int poly_n;
        float poly_sigma;
        float deltat;
        int uif;
        int fg; //this is Farneback Gaussian settings, NOT do first guess***
        int doCTH;
        //Modified Sun Defaults
        double lambda;
        double alpha;
        double alpha2;
        double lambdac;
        double scsig;
        double filtsigma;
        double scaleF;  //pyramid scale factor, not changable on command line at the moment
        int kiters; //outer iterations or pyramid levels +1
        int liters; //inner iterations or number of cg solving update steps per GNC level
        int cgiters; //conjugate gradient iterations maximum, will stop when error is permissably low
        int miters;
        int setnorms; //flag to set the normalization min max values
        float NormMax; //currently not set by user, lets change that!
        float NormMin;
        float NormMax2;
        float NormMin2;
        float NormMax3;
        float NormMin3;
        bool outnav; //GOES output options, all default to true
        bool outraw;
        bool outrad;
        bool outctp;
        bool setNormMax;
        bool setNormMin;
        bool setNormMax2;
        bool setNormMin2;
        bool setNormMax3;
        bool setNormMin3;
};

