template <class T>
T oct_bc(T x, int nx,bool &bc)
{
    bc=false;
    T result;
    result = x;
    if(x < 0)
    {
        result = 0;
        bc=true;
    }
    if(x >= nx)
    {
        result = nx-1;
        bc=true;
    }
    return result;
}


