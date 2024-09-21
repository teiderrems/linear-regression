#include "regression.h"

double *fit(double **X,double *Y,int n,int m,double precision, double learning_rate,int max_iter)
{
    double *beta=generate(m+1);
    print(beta,m+1);
    double *x=get_array(X,m,0);
    double * grad=gradient(x,beta,Y[0],m);
    double norm=norme(grad,m+1);
    int k=0;
    while (k<=max_iter) //norm>precision || 
    {
        beta=update(beta,grad,learning_rate,m);
        for (int i = 1; i < n; i++)
        {
            x=get_array(X,m,i);
            grad=gradient(x,beta,Y[i],m);
            norm=norme(grad,m+1);
        }
        printf("\nlost_iter_%d==%.3f\n",k,lost(X,Y,beta,n,m));
        k++;
    }
    return beta;
}

double * gradient(double * x,double * beta,double t,int n){

    double * grad=malloc((n+1)*sizeof(double));
    double val=(-2.0)*(t-helper(x,beta,n));
    grad[0]=val;
    for (int i = 0; i < n; i++)
    {
        grad[i+1]=val*x[i];
    }
    return grad;
}

double helper(double * x,double * beta,int n){
    double result=beta[0];
    for (int i = 0; i < n; i++)
    {
        result+=beta[i+1]*x[i];
    }
    return result;
}

double predict(double * x,double * beta,int n){
    return helper(x,beta,n);
}

double * update(double *beta0,double *grad, double learning_rate,int n){
    double * beta=malloc((n+1)*sizeof(double));
    for (int i = 0; i < n+1; i++)
    {
        beta[i]=beta0[i]-learning_rate*grad[i];
    }
    return beta;
}

double * get_array(double **X,int n,int j){
    double * vector=malloc(n*sizeof(double));
    for (int i = 0; i < n; i++)
    {
        vector[i]=X[j][i];
    }
   return vector; 
}

double *generate(int n)
{
    time(NULL);
    double * beta=malloc(n*sizeof(double));
    for (int i = 0; i < n; i++)
    {
        beta[i]=(double)rand()/(double)RAND_MAX;
    }
    return beta;
}

double **generateMatrix(int n, int m)
{
    time(NULL);
    double **X=malloc(n*sizeof(double));
    for (int i = 0; i < n; i++)
    {
        X[i]=malloc(m*sizeof(double));
        for (int j = 0; j < m; j++)
        {
            X[i][j]=rand()/RAND_MAX;
        } 
    }
    return X;
}

double norme(double *grad, int n)
{
    double norm=0.0;
    for (int i = 0; i < n; i++)
    {
        norm+=pow(grad[i],2.0);
    }

    return sqrt(norm);
}

double lost(double **X, double *y, double *beta,int n,int m)
{
    double result=0.0;
    for (int i = 0; i < n; i++)
    {
        double *x=get_array(X,m,i);
        result+=pow(y[i]-helper(x,beta,m),2.0);
    }
    return result/n;
}


void print(double *d,int n){
    printf("\n\n");
    for (int i = 0; i < n; i++)
    {
        printf("%.3f \t",d[i]);
    }
    printf("\n\n");
}