//Definitions for bicubic
static double oct_cell( double v[4], double x);
static double oct_bicubic_cell( double p[4][4], double x, double y);
double oct_bicubic(double * input, double uu, double vv, int nx, int ny,int inout);
double oct_bicubic_float(float * input, double uu, double vv, int nx, int ny,int inout);
