#include <iostream>
#include <stdlib.h>
#include <stdio.h>
using namespace std;
//Function: oct_normalize_geo
//Purpose: This function normalizes a geo image to values between maxout and minout
//Author: Jason Apke, Updated 9/10/2018

void oct_bandminmax(int gb, float &maxch,float &minch)
{
        if(gb == 1)
        {
            maxch = 804.03605737;
            minch = -25.93664701;
        } else if(gb == 2)
        {
            maxch = 628.98723908;
            minch = -20.28991094;
        } else if(gb == 3)
        {
            maxch = 373.16695681;
            minch = -12.03764377;
        } else if(gb == 4)
        {
            maxch = 140.19342584;
            minch = -4.52236858;
        } else if(gb == 5)
        {
            maxch = 94.84802665;
            minch = -3.05961376;
        } else if(gb == 6)
        {
            maxch = 29.78947040;
            minch = -0.96095066;
        } else if(gb == 7)
        {
            //maxch = 24.962;
            //minch = -0.0114;
            //Above is actual range
            //below is meteorological range
            maxch = 2.;
            minch = 0.;
        } else if(gb == 8)
        {
            //These are documented
            //maxch = 28.366;
            //minch = -0.1692;
            //This is an experimental meteorological range
            maxch = 6.;
            minch = 3.;
        } else if(gb == 9)
        {
            maxch = 44.998;
            minch = -0.2472;
        } else if(gb == 10)
        {
             maxch = 79.831;
             minch = -0.2871;
        } else if(gb == 10)
        {
             maxch = 79.831;
             minch = -0.2871;
        } else if(gb == 11)
        {
             maxch = 134.93;
             minch = -0.3909;
        } else if(gb == 12)
        {
             maxch = 108.44;
             minch = -0.4617;
        } else if(gb == 13)
        {
             maxch = 185.5699;
             minch = -1.6443;
        } else if(gb == 14)
        {
             maxch = 198.71;
             minch = -0.5154;
        } else if(gb == 15)
        {
             maxch = 212.28;
             minch = -0.5262;
        } else if(gb == 16)
        {
             maxch = 170.19;
             minch = -1.5726;
        }
}
//Note, the normalize function does not censor above or below the min/max, just scales the data such that
//maxin = maxout, minin = minout
void oct_normalize_geo(double ** image, double maxin, double minin,double maxout,double minout,int nx,int ny,int nc)
{
    for (int jj2 = 0; jj2<ny; ++jj2)
    {
        long nxtjj = nx*jj2;
        for(int ii2 = 0; ii2<nx; ++ii2)
        {
            long lxyz = ii2+nxtjj;
            image[lxyz][nc] = ((image[lxyz][nc]-minin)/(maxin-minin))*(maxout-minout)+minout;
        }
    }
}
