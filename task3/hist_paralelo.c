#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* Pthread API */
#include <pthread.h>

/* Semaphore API */
#include <semaphore.h>

int n, nval, nthreads, *vet, **local_vet;
double h, *val, max, min;
sem_t *semaphores;
/* Pthread array */
pthread_t *bin_thread;


/* funcao que calcula o minimo valor em um vetor */
double min_val(double * vet,int nval) {
	int i;
	double min;

	min = FLT_MAX;

	for(i=0;i<nval;i++) {
		if(vet[i] < min)
			min =  vet[i];
	}
	
	return min;
}

/* funcao que calcula o maximo valor em um vetor */
double max_val(double * vet, int nval) {
	int i;
	double max;

	max = FLT_MIN;

	for(i=0;i<nval;i++) {
		if(vet[i] > max)
			max =  vet[i];
	}
	
	return max;
}

void *hist_thread (void* my_rank) {

	double min_t, max_t;
	int i, j, count, 
		rank = ((int) my_rank),
		local_size = nval / nthreads,
		i_start = rank * local_size,
		i_end = (rank + 1) * local_size - 1;

	for(j = 0; j < n; j++) {
		count = 0;
		min_t = min + j*h;
		max_t = min + (j+1)*h;
		for(i = i_start; i <= i_end; i++) {
			if(val[i] <= max_t && val[i] > min_t) {
				count++;
			}
		}

		local_vet[rank][j] = count;
	}

	if (rank % 2) /* Odd threads */
		sem_post(&semaphores[rank]);

	else { /* Even threads */
	
		int next = 1;
	
		do {

			sem_wait(&semaphores[rank+next]);

			for(j = 0; j < n; j++)
				local_vet[rank][j] += local_vet[rank + next][j];

			next *= 2;

		} while (rank + next < nthreads && (rank / next) % 2 != 1);

		sem_post(&semaphores[rank]);
	}

	return NULL;
}

/* conta quantos valores no vetor estao entre o minimo e o maximo passados como parametros */
int * count() {
	int i;

	/* Creates all threads, according to the first value read from stdin */
	for (i = 0; i < nthreads; i++)
		pthread_create(&bin_thread[i], NULL, hist_thread, (void*) i);

	for (i = 0; i < nthreads; i++)
		pthread_join(bin_thread[i], NULL);

	free(bin_thread);

	return local_vet[0];
}

int main(int argc, char * argv[]) {

	int i;
	long unsigned int duracao;
	struct timeval start, end;

	/* reads number of threads */
	scanf("%d",&nthreads);

	/* entrada do numero de dados */
	scanf("%d",&nval);
	/* numero de barras do histograma a serem calculadas */
	scanf("%d",&n);

	/* vetor com os dados */
	val = (double *)malloc(nval*sizeof(double));
	local_vet = (int **) malloc (nthreads * sizeof(int*));
	semaphores = (sem_t*) malloc (nthreads * sizeof(sem_t));
	/* Allocates memory for thread handlers */
	bin_thread = (pthread_t*) malloc (nthreads * sizeof(pthread_t));

	for (i = 0; i < nthreads; i++) {
		sem_init(&semaphores[i], 0, 0);
		local_vet[i] = (int*) malloc (n * sizeof(int));
	}

	/* entrada dos dados */
	for(i=0;i<nval;i++) {
		scanf("%lf",&val[i]);
	}

	/* calcula o minimo e o maximo valores inteiros */
	min = floor(min_val(val,nval));
	max = ceil(max_val(val,nval));

	/* calcula o tamanho de cada barra */
	h = (max - min)/n;

	gettimeofday(&start, NULL);

	/* chama a funcao */
	vet = count();

	gettimeofday(&end, NULL);

	duracao = ((end.tv_sec * 1000000 + end.tv_usec) - \
	(start.tv_sec * 1000000 + start.tv_usec));

	printf("%.2lf",min);	
	for(i=1;i<=n;i++) {
		printf(" %.2lf",min + h*i);
	}
	printf("\n");

	/* imprime o histograma calculado */	
	printf("%d",vet[0]);
	for(i=1;i<n;i++) {
		printf(" %d",vet[i]);
	}
	printf("\n");

	/* imprime o tempo de duracao do calculo */
	printf("%lu\n",duracao);

	for (i = 0; i < nthreads; i++) {
		free (local_vet[i]);
		sem_destroy(&semaphores[i]);
	}

	free(semaphores);
	free(local_vet);
	free(val);

	return 0;
}

