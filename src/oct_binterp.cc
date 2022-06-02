#include <math.h>
#include <cstdlib>
#include <cstdio>
using namespace std;

//Function: oct_binterp
//Purpose: This is a C++ function to perform bilinear interpolation
//Returns: Bilinear interpolation of value at point x, y as a double, coefs and coef_binterp are functions which
//         keep and use all coefs for computational efficiency when needed
//
//Author: Jason Apke, Updated 9/10/2018
double oct_binterp (double x, double y,double x1, double x2, double y1, double y2,double f11,double f21,double f12,double f22)
{
    double fv1,fv2,ans;
    double p1 = (x2-x)/(x2-x1);
    double p2 = (x-x1)/(x2-x1);
    fv1 = (p1)*f11+(p2)*f21;
    fv2 = (p1)*f12+(p2)*f22;
    ans = ((y2-y)/(y2-y1))*fv1+((y-y1)/(y2-y1))*fv2;
    return ans;
}
double oct_binterp_coefs (double x, double y,double x1, double x2, double y1, double y2,double f11,double f21,double f12,double f22,double & p1, double & p2,double & p3,double & p4)
{
    double fv1,fv2,ans;
    p1 = (x2-x)/(x2-x1);
    p2 = (x-x1)/(x2-x1);
    fv1 = (p1)*f11+(p2)*f21;
    fv2 = (p1)*f12+(p2)*f22;
    p3 = ((y2-y)/(y2-y1));
    p4 = ((y-y1)/(y2-y1));
    ans = p3*fv1+p4*fv2;
    return ans;
}
double oct_coef_binterp(double p1, double p2, double p3, double p4, double f11,double f21,double f12,double f22)
{
    double fv1, fv2, ans;
    fv1 = (p1)*f11+(p2)*f21;
    fv2 = (p1)*f12+(p2)*f22;
    ans = p3*fv1+p4*fv2;
    return ans;
}
