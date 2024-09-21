#ifndef ___LINEAR_REGRESSION___
#define ___LINEAR_REGRESSION___
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define MAX_LENGTH 10

double * gradient(double *,double *,double,int);

double helper(double *,double *,int n);

double * fit(double **,double *,int,int,double, double,int);

double predict(double *,double *,int);

double * update(double *,double *, double,int);

double * get_array(double **,int,int);

double * generate(int);
double ** generateMatrix(int,int);

double norme(double *, int);

double lost(double**,double*,double*,int,int);

void print(double *,int);


#endif // ___LINEAR_REGRESSION___