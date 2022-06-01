#include <cstdlib>

class Image
{
    public:
        float *data;
        int nrow, ncol,nchannels;
        Image(){
            nrow = 0;
            ncol = 0;
            nchannels=0;
        };
        Image(int x, int y, int c){
            nrow = x;
            ncol = y;
            nchannels=c;
        };
        void setdims(int x, int y, int c){
            nrow = x;
            ncol = y;
            nchannels = c;
        }

};

