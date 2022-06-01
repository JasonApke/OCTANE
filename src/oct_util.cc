#include <algorithm>
#include <iostream>
using namespace std;
//Purpose: This is a set of utility functions to clean up the optical flow code
// dMatrix is a double 2d pointer allocation function, run free_dMatrix to free
// Same goes for float
//Author: Jason Apke, Updated 9/10/2018

double **dMatrix (int nRows, int nCols)
{
    double **mat;
    mat = new double *[nRows];
    if(!mat)
    {
        cout << "Not enough memory\n";
        exit(0);
    }
        
    for(int mi = 0; mi < nRows; mi++){
        mat[mi] = new double [nCols];
        if(!mat[mi])
        {
            cout << "Not enough memory\n";
            exit(0);
        }
    }
    return mat;
}
double ***dImage (int nrow, int ncol,int nchannels)
{
    double ***data;
	data = new double **[nrow];
	if(!data)
	{
		std::cout << "Not enough memory\n";
		exit(0);
	}
	for(int mi = 0; mi < nrow; mi++){
		data[mi] = new double *[ncol];
		if(!data[mi])
		{
			std::cout << "Not enough memory\n";
			exit(0);
		}
		for(int mj = 0; mj < ncol; mj++){
			data[mi][mj] = new double [nchannels];
			if(!data[mi][mj])
			{
				std::cout << "Not enough memory\n";
				exit(0);
			}
		}
	}    
    return data;
}
float **fMatrix (int nRows, int nCols)
{
    float **mat;
    mat = new float *[nRows];
    if(!mat)
    {
        cout << "Not enough memory\n";
        exit(0);
    }
        
    for(int mi = 0; mi < nRows; mi++){
        mat[mi] = new float [nCols];
        if(!mat[mi])
        {
            cout << "Not enough memory\n";
            exit(0);
        }
    }
    return mat;
}
void dMatrix_initzero(double **mat,int nRows, int nCols)
{
    for(int mi = 0; mi < nRows; mi++)
    {
        for(int mj = 0; mj < nCols; mj++)
        {
            mat[mi][mj] = 0;
        }
    }
}
void free_dMatrix (double **mat, int nRows)
{
    for(int mi = 0; mi < nRows; mi++)
        delete [] mat[mi];
    delete [] mat;
}

void free_dImage (double ***data, int nrow,int ncol)
{
	for(int ni = 0; ni < nrow; ni++)
	{
		for(int nj = 0; nj < ncol; nj++)
			delete[] data[ni][nj];
		delete[] data[ni];
	}
	delete [] data;
}
void free_fMatrix (float **mat, int nRows)
{
    for(int mi = 0; mi < nRows; mi++)
        delete [] mat[mi];
    delete [] mat;
}


