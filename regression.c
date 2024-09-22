#include "regression.h"


/// @brief fonction pour faire l'apprentissage des coéfficients de la droit de régression
/// @param X matrix des données
/// @param Y vecteur des cibles
/// @param n nombre d'individu ou nombre de ligne de la matrice X
/// @param m nombre de caractéristique ou feature ou nombre de colonne de la matrice X
/// @param precision marge d'érreur admissible
/// @param learning_rate la taille du pas de la descente de gradient
/// @param batch_size nombre d'individu pour l'apprentissage par mini-batch si sa valeur est égale à 0 ou 1, alors nous somme dans le cas d'une méthode stochastique
/// @param max_iter nombre d'itération maximal de l'algorithme d'apprentissage
/// @return vecteur contenant les coefficients de la droite de regression obtenuent après apprentissage avec la fonction de coût RMSE ( Root Mean Square Error)
double *fit(double **X,double *Y,int n,int m,double precision, double learning_rate,int batch_size,int max_iter)
{
    printf("\n\nbegin fit\n");
    double *beta=generate(m+1); // line pour générer les coéfficients initiaux
    print(beta,m+1);
    double *x=NULL;
    double * grad=NULL;
    double ** batch=NULL;
    double *t=NULL;
    if (batch_size>1)
    {
        int mod=n%batch_size;
        if (mod>0)
        {
            X=grow_math(X,n,mod,m); // augmentation du nombre d'individu
            Y=grow_array(Y,n,mod); // augmentation du nombre de cible
        }
        batch=get_matrix(X,0,m,batch_size,n);
        t=get_target(Y,0,batch_size,n);
        grad=gradient_batch(batch,beta,t,m,batch_size);
    }
    else if (batch_size==0 || batch_size==1)
    {
        x=get_array(X,m,0);
        grad=gradient(x,beta,Y[0],m);
    }
    double norm=norme(grad,m+1);
    int k=0;
    while (k<=max_iter) //norm>precision || 
    {
        beta=update(beta,grad,learning_rate,m); // Mise à jour des coéfficients de regression
        if (batch_size>1)
        {
            for (int i = batch_size; i < n; i+=batch_size)
            {
                batch=get_matrix(X,i,m,batch_size,n); // line pour extraire le nombre d'individu correspondant à la taille du batch
                t=get_target(Y,i,batch_size,n); // extraction des cibles associées aux individus contenus dans la matrice batch
                grad=gradient_batch(batch,beta,t,m,batch_size); // calcul du gradient
                norm=norme(grad,m+1);
            }
        }
        else
        {
            for (int i = 1; i < n; i++)
            {
                x=get_array(X,m,i);
                grad=gradient(x,beta,Y[i],m);
                norm=norme(grad,m+1);
            }
        }
        printf("\nlost_iter_%d==%.3f\n",k,lost(X,Y,beta,n,m));
        k++;
    }
    free(x);
    free(grad);
    free_math(batch,batch_size);
    free(t);
    printf("\n\nend fit\n");
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

    srand((unsigned int)time(NULL));
    double * beta=malloc(n*sizeof(double));
    for (int i = 0; i < n; i++)
    {
        beta[i]=((double)rand()/(double)RAND_MAX) * (1.0 - 0.0);
    }

    return beta;
}

double **generateMatrix(int n, int m)
{
    double **X=malloc(n*sizeof(double));
    for (int i = 0; i < n; i++)
    {
        X[i]=generate(m);
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

double *zero(int n)
{
    double *z=malloc((n)*sizeof(double));
    for (int i = 0; i < n; i++)
    {
        z[i]=0.0;
    }
    return z;
}

double **get_matrix(double **X, int start, int m,int batch_size,int nrow)
{
    double ** x=malloc(batch_size*sizeof(double));
    for (int i = start;i < start+batch_size; i++)
    {
        int k=i-start;
        x[k]=malloc(m*sizeof(double));
        for (int j = 0; j < m; j++)
        {
            x[k][j]=X[i][j];
        } 
    }
    return x;
}

double *get_target(double *T, int start, int batch_size, int nrow)
{
    double *y=NULL;
    if (nrow-start>=batch_size)
    {
        y=malloc(batch_size*sizeof(double));
        for (int i = start;i<nrow && i < start+batch_size; i++)
        {
            int k=i-start;
            y[k]=T[i];
        }
    }
    else
    {
        y=malloc((nrow-start)*sizeof(double));
        for (int i = start;i<nrow && i < start+(nrow-start); i++)
        {
            int k=i-start;
            y[k]=T[i];
        }
    }
    return y;
}

void free_math(double ** x,int n)
{

    if (x!=NULL)
    {
        for (int i = 0; i < n; i++)
        {
            free(x[i]);
        }
    }
    return;
}

double *grow_array(double *T, int init, int new)
{
    srand((unsigned int)time(NULL));
    T=realloc(T,(init+new)*sizeof(double));
    for (int i = init; i < init+new; i++)
    {
        T[i]=((double)rand()/(double)RAND_MAX) * (1.0 - 0.0);
    }
    return T;
}

void print_matrix(double **X, int n, int m)
{
    if (X!=NULL)
    {
        printf("[\n");
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                printf("%.5f\t",X[i][j]);
            }
            printf("\n");
        }
        printf("]\n");
    }
    return;
}

double **grow_math(double **X, int init, int new, int m)
{
    X=realloc(X,(new+init)*sizeof(double));
    for (int i = init; i < init+new; i++)
    {
        X[i]=generate(m);
    }
    return X;
}

double * gradient_batch(double **X,double *beta,double* t,int n,int batch_size){
    double *grad=zero(n+1);
    double *x=NULL;
    double * g=NULL;
    for (int i = 0; i < batch_size; i++)
    {
        x=get_array(X,n,i);
        g=gradient(x,beta,t[i],n);
        for (int j = 0; j < n+1; j++)
        {
            grad[j]+=g[j]/(double)batch_size;
        }
    }
    free(x);
    free(g);
    return grad;
}