#include "regression.h"


int main(int argc,char *argv[]){
    int n=11,m=2;
    double **X=generateMatrix(n,m);

    double *y=generate(n);
    double *beta=fit(X,y,n,m,0.00001,0.005,4,100);

    print(beta,m+1);
    free(beta);
    free(y);
    for (int i = 0; i < n; i++)
    {
        free(X[i]);
    }
    return EXIT_SUCCESS;
}