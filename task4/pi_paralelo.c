#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/* Pthread API */
#include <pthread.h>

/* Avoiding rand_r warning, according Breno Leite's post in our courses' group. */
#define _GNU_SOURCE

/* Shared estimative */
long long unsigned int in = 0;
/* Number of threads and iterations */
unsigned int n_threads, nval, *id;
pthread_t *monte_carlo_threads;
/* Mutex to synchronize data */
pthread_mutex_t monte_carlo_mutex;

/* Monte Carlo's pi estimative computing thread */
void *monte_carlo_pi_thread(void *rank) {

	unsigned int my_rank = *((unsigned int*) rank);

	/* Computes loop fraction this thread should do */
	int local_size = nval / n_threads,
	    i_start = my_rank * local_size,
	    i_end = (my_rank == n_threads - 1 ? nval - 1 : (my_rank + 1) * local_size - 1);

	long long unsigned int local_in = 0, i;
	double x, y, d;

	for (i = i_start; i <= i_end; i++) {
		x = ((rand_r(&my_rank) % 1000000)/500000.0)-1;
		y = ((rand_r(&my_rank) % 1000000)/500000.0)-1;
		d = ((x*x) + (y*y));
		if (d <= 1) local_in += 1;
	}

	/* Aggregates local_in to shared variable */
	pthread_mutex_lock(&monte_carlo_mutex);
	in += local_in;
	pthread_mutex_unlock(&monte_carlo_mutex);

	return NULL;

}

/* Creates and starts all threads to compute pi estimative */
void monte_carlo_pi() {

	unsigned int i;

	/* Creates all threads, according to the first value read from stdin */
	for (i = 0; i < n_threads; i++) {
		id[i] = i;
		pthread_create(&monte_carlo_threads[i], NULL, monte_carlo_pi_thread, (void*) &id[i]);
	}

	/* Waits for all threads to end before returning result */
	for (i = 0; i < n_threads; i++)
		pthread_join(monte_carlo_threads[i], NULL);
}

int main(void) {

	double pi;
	long unsigned int duracao;
	struct timeval start, end;

	scanf("%u %u", &n_threads, &nval);

	/* Allocates memory for thread handlers */
	monte_carlo_threads = (pthread_t*) malloc (n_threads * sizeof(pthread_t));
	/* Creates mutex to protect final data */
	pthread_mutex_init(&monte_carlo_mutex, NULL);
	/* Array of thread ids : avoids cast warnings on pthread_create calls */
	id = (unsigned int *) malloc (n_threads * sizeof(unsigned int));

	srand (time(NULL));

	gettimeofday(&start, NULL);
	monte_carlo_pi();
	gettimeofday(&end, NULL);

	duracao = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));

	pi = 4*in / ((double) nval);
	printf("%lf\n%lu\n",pi,duracao);

	free(id);
	free(monte_carlo_threads);
	pthread_mutex_destroy(&monte_carlo_mutex);

	return 0;
}
