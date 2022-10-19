#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void bubble_sort(int list[], long n)
{
  long c, d, t;
 
  for (c = 0 ; c < n - 1; c++)
  {
    for (d = 0 ; d < n - c - 1; d++)
    {
      if (list[d] > list[d+1])
      {
        /* Swapping */
 
        t         = list[d];
        list[d]   = list[d+1];
        list[d+1] = t;
      }
    }
  }
}
 

void main(int argc, char**argv)
{
	int n;
	int i, j, k=0;
	int not_flag;
	int *primes;

	if (argc<2)
	{
		printf("./exec num\n");
		exit(-1);
	} else
		n = atoi(argv[1]);

	primes = (int *)malloc(sizeof(int)*(n/2+1));

	double start = omp_get_wtime();

	//#pragma omp parallel...
	for (i=2; i<n; i++)
	{
		not_flag = 0;
		j=2;
		while(j*j<=i && !not_flag)
		{
			if(i%j==0) // not prime
				not_flag=1;
			j++;
		}
		if (j*j>=i && !not_flag)
			primes[k++] = i;
	}

	double stop = omp_get_wtime();

#ifdef DEBUG
	bubble_sort(primes, k);
	printf("List of primes=");
	for (i=0; i<k; i++)
		printf("%d ", primes[i]);
	printf("\n");
#endif
	printf("Number of primes between 1-%d = %d\n", n, k);
	printf("Computed in %f s.\n", stop-start);

	free(primes);

}
