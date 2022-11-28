
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "timer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

float *A;
float *Anew;

#define MAX(a,b) (((a) > (b)) ? (a) : (b))

int main(int argc, char** argv)
{
	int iter_max = 1000;
	int n, m;

	const float tol = 1.0e-6;
	float error     = 1.0;

	// For time
	double runtime;

	if (argc==3)
	{
		n = atoi(argv[1]);
		m = atoi(argv[2]);
	} else {
		n = 4096;
		m = 4096;
	}


	A    = (float *)malloc(n*m*sizeof(float));
	Anew = (float *)malloc(n*m*sizeof(float));

	// Init A and Anew
	for( int j = 0; j < n; j++){
		A[j*m+0]    = 1.0;
		Anew[j*m+0] = 1.0;
		for( int i = 1; i < m; i++)
		{
			A[j*m+i]    = 0.0;
			Anew[j*m+i] = 0.0;
		}
	}




	printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);

	int iter = 0;

#ifdef _OPENMP
#pragma omp target enter data map(to: error, A[0:n*m]) map (alloc: Anew[0:n*m])
#endif
{
	StartTimer();

	while ( error > tol && iter < iter_max )
	{
		error = 0.0;
#ifdef _OPENMP
		#pragma omp target teams distribute parallel for collapse (2) reduction (max:error)
#endif
		for( int j = 1; j < n-1; j++)
		{
			for( int i = 1; i < m-1; i++ )
			{
				Anew[j*m+i] = 0.25 * ( A[j*m+i+1] + A[j*m+i-1]
					+ A[(j-1)*m+i] + A[(j+1)*m+i]);
				error = MAX( error, fabs(Anew[j*m+i] - A[j*m+i]));
			}
			//#pragma omp target update from(error)
		}
#ifdef _OPENMP
		#pragma omp target teams distribute parallel for collapse(2)
#endif
		for( int j = 1; j < n-1; j++)
		{
			for( int i = 1; i < m-1; i++ )
			{
				A[j*m+i] = Anew[j*m+i];    
			}
		}

#ifdef _OPENMP
		#pragma omp target exit data map(from: Anew)
#endif
		if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);

		iter++;
	}
	runtime = GetTimer();
	printf("iter=%i error=%f\n", iter, error);
}

	printf(" total: %f s\n", runtime / 1000);

	free(A); free(Anew);
}

