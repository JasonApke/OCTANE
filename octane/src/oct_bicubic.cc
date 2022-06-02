#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include "oct_bc.h"
using namespace std;
//Function: oct_bicubic
//Purpose: Does bicubic interpolation of a point in 2 dimensions

static double oct_cell (
    double v[4],  //interpolation points
    double x      //point to be interpolated
)
{
    return  v[1] + 0.5 * x * (v[2] - v[0] +
            x * (2.0 *  v[0] - 5.0 * v[1] + 4.0 * v[2] - v[3] +
            x * (3.0 * (v[1] - v[2]) + v[3] - v[0])));
}
//Bicubic interpolation (cubic for 2 dimensions)
static double oct_bicubic_cell (
    double p[4][4], //array containing the interpolation points
    double x,       //x position to be interpolated
    double y        //y position to be interpolated
)
{
    double v[4];
    v[0] = oct_cell(p[0], y);
    v[1] = oct_cell(p[1], y);
    v[2] = oct_cell(p[2], y);
    v[3] = oct_cell(p[3], y);

    return oct_cell(v, x);
}


double oct_bicubic(double * input, double uu, double vv, int nx, int ny,int inout)
{
    int sx = 1;
    int sy = 1;
    int x, y, mx, my, dx,dy,ddx,ddy;
    bool bc;

    //I always use reflecting boundary conditions, so this should work
    x =   oct_bc<int>((int) uu,nx,bc);
    y =   oct_bc<int>((int) vv,ny,bc);
    mx =  oct_bc<int>((int) (uu-sx),nx,bc);
    my =  oct_bc<int>((int) (vv-sy),ny,bc);
    dx =  oct_bc<int>((int) (uu+sx),nx,bc);
    dy =  oct_bc<int>((int) (vv+sy),ny,bc);
    ddx = oct_bc<int>((int) (uu+2*sx),nx,bc);
    ddy = oct_bc<int>((int) (vv+2*sy),ny,bc);
    //Recent JMAMOD I got a hunch....
    //ddx = oct_bc((int) (uu-2*sx),nx);
    //ddy = oct_bc((int) (vv-2*sy),ny);
    int nxtmy = nx*my;
    int nxty = nx*y;
    int nxtdy = nx*dy;
    int nxtddy = nx*ddy;

    const double p11 = input[mx  + nxtmy];
    const double p12 = input[x   + nxtmy];
    const double p13 = input[dx  + nxtmy];
    const double p14 = input[ddx + nxtmy];

    const double p21 = input[mx  + nxty];
    const double p22 = input[x   + nxty];
    const double p23 = input[dx  + nxty];
    const double p24 = input[ddx + nxty];

    const double p31 = input[mx  + nxtdy];
    const double p32 = input[x   + nxtdy];
    const double p33 = input[dx  + nxtdy];
    const double p34 = input[ddx + nxtdy];

    const double p41 = input[mx  + nxtddy];
    const double p42 = input[x   + nxtddy];
    const double p43 = input[dx  + nxtddy];
    const double p44 = input[ddx + nxtddy];
    
    double pol[4][4] = {
        {p11, p21, p31, p41},
        {p12, p22, p32, p42},
        {p13, p23, p33, p43},
        {p14, p24, p34, p44}
    };
    double f = oct_bicubic_cell(pol,uu-x,vv-y);
    if(f != f)
    {
        printf("Bicubic failure, possible data issue\n");

        exit(0);

    }
    return f;

}
double oct_bicubic_float(float * input, double uu, double vv, int nx, int ny,int inout)
{
    int sx = 1;
    int sy = 1;
    int x, y, mx, my, dx,dy,ddx,ddy;
    bool bc;

    x =   oct_bc<int>((int) uu,nx,bc);
    y =   oct_bc<int>((int) vv,ny,bc);
    mx =  oct_bc<int>((int) (uu-sx),nx,bc);
    my =  oct_bc<int>((int) (vv-sy),ny,bc);
    dx =  oct_bc<int>((int) (uu+sx),nx,bc);
    dy =  oct_bc<int>((int) (vv+sy),ny,bc);
    ddx = oct_bc<int>((int) (uu+2*sx),nx,bc);
    ddy = oct_bc<int>((int) (vv+2*sy),ny,bc);
    int nxtmy = nx*my;
    int nxty = nx*y;
    int nxtdy = nx*dy;
    int nxtddy = nx*ddy;

    const double p11 = input[mx  + nxtmy];
    const double p12 = input[x   + nxtmy];
    const double p13 = input[dx  + nxtmy];
    const double p14 = input[ddx + nxtmy];

    const double p21 = input[mx  + nxty];
    const double p22 = input[x   + nxty];
    const double p23 = input[dx  + nxty];
    const double p24 = input[ddx + nxty];

    const double p31 = input[mx  + nxtdy];
    const double p32 = input[x   + nxtdy];
    const double p33 = input[dx  + nxtdy];
    const double p34 = input[ddx + nxtdy];

    const double p41 = input[mx  + nxtddy];
    const double p42 = input[x   + nxtddy];
    const double p43 = input[dx  + nxtddy];
    const double p44 = input[ddx + nxtddy];
    
    double pol[4][4] = {
        {p11, p21, p31, p41},
        {p12, p22, p32, p42},
        {p13, p23, p33, p43},
        {p14, p24, p34, p44}
    };
    double f = oct_bicubic_cell(pol,uu-x,vv-y);
    if(f != f)
    {
        printf("Bicubic failure, possible data issue\n");
        exit(0);
    }
    return f;
}
