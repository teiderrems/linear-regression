#ifndef ___LINEAR_REGRESSION___
#define ___LINEAR_REGRESSION___
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define MAX_LENGTH 10

double * gradient(double *,double *,double,int);
double * gradient_batch(double **,double *,double*,int,int);

double helper(double *,double *,int n);

double * fit(double **,double *,int,int,double, double,int,int);

double predict(double *,double *,int);

double * update(double *,double *, double,int);

double * get_array(double **,int,int);

double * generate(int);
double ** generateMatrix(int,int);

double norme(double *, int);

double lost(double**,double*,double*,int,int);

void print(double *,int);
double * zero(int);

double ** get_matrix(double **,int,int,int,int);
double * get_target(double *,int,int,int);

void free_math(double **,int);

double ** grow_math(double **,int,int,int);
double * grow_array(double *,int,int);

void print_matrix(double **,int,int);


#endif // ___LINEAR_REGRESSION___