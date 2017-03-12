/*
*   Task #1 of the course MO644 - Parallel Programming
*   This task consists in parallelizing the count sorting algorithm.
*
*   Gustavo CIOTTO PINTON RA 117136
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/*
*   Parallel implementation of count sorting algorithm. Besides the array and its length,
*   it receives the number of thread that should be created by OpenMP.
*   Returns the total time spent to sort the array.
*/
double count_sort_paralelo(double a[], int n, int n_threads) {
	int i, j, count;
	double *temp;
	double start, end, duracao;

	temp = (double *)malloc(n*sizeof(double));

    /* Starts counting the time */
	start = omp_get_wtime();
    /* Pragma directive to parallelize for block. Three variables are shared among the threads: the temporary
       and value arrays, and its size. In addition, three other are private: the local indexes i and j, and 
       the new calculated position of a[i] */
# 	pragma omp parallel for num_threads(n_threads) default(none) shared(n, a, temp) private(i, j, count)
	for (i = 0; i < n; i++) {
		count = 0;
		for (j = 0; j < n; j++)
			if (a[j] < a[i])
				count++;
			else if (a[j] == a[i] && j < i)
				count++;
        /* Since temp[count] can only be accessed by exactly one thread, it does not need omp critical directive. */
		temp[count] = a[i]; 
	}
	end = omp_get_wtime();

	duracao = end - start;

	memcpy(a, temp, n*sizeof(double));
	free(temp);

	return duracao;
}

/* Provided code */
int main(int argc, char * argv[]) {
	int i, n, nt;
	double  * a, t_s;

	scanf("%d",&nt);
	
	/* numero de valores */
	scanf("%d",&n);

	/* aloca os vetores de valores para o teste em serial(b) e para o teste em paralelo(a) */
	a = (double *)malloc(n*sizeof(double));

	/* entrada dos valores */
	for(i=0;i<n;i++)
		scanf("%lf",&a[i]);
	
	/* chama as funcao de count sort em paralelo */
	t_s = count_sort_paralelo(a, n, nt);
	
	/* Imprime o vetor ordenado */
	for(i=0;i<n;i++)
		printf("%.2lf ",a[i]);

	printf("\n");

	/* imprime os tempos obtidos e o speedup */
	printf("%lf\n",t_s);

	return 0;
}
