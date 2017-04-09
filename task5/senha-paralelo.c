/* 
* Task #5: A simple Pthread program to break a ZIP file.
* Gustavo Ciotto Pinton RA117136
* MO644 - Parallel Programming
*/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

/* Pthread API */
#include <pthread.h>

#define PASSWD_SIZE 500000

/* Shared variables used among all threads. If found is true, all threads stop their computing. */
unsigned int n_threads, *id, found = 0;
char filename[100];
const char finalcmd[300] = "unzip -P%d -t %s 2>&1";
pthread_t *break_zip_threads;

FILE *popen(const char *command, const char *type);

/* Thread function: iterates over a range of passwords. Sets found if it finds the password,
   signaling other threads to stop computing. */
void *break_zip_thread (void *rank) {

	FILE * fp;
	char ret[200], cmd[400];

	unsigned int my_rank = *((unsigned int*) rank), i;

	/* Computes loop fraction this thread should do */
	int local_size = PASSWD_SIZE / n_threads,
	    i_start = my_rank * local_size,
	    i_end = (my_rank == n_threads - 1 ? PASSWD_SIZE	 - 1 : (my_rank + 1) * local_size - 1);

	for(i = i_start; i <= i_end && !found; i++){

		sprintf((char*)&cmd, finalcmd, i, filename);

		fp = popen(cmd, "r");	
		while (!feof(fp)) {
			fgets((char*)&ret, 200, fp);
			if (strcasestr(ret, "ok") != NULL) {
				printf("Senha:%d\n", i);
				found = 1;
			}
		}
		pclose(fp);
	}

	return NULL;

}

/* Creates, starts and waits all threads to finish */
void break_zip_parallel () {

	unsigned int i;

	/* Creates all threads, according to the first value read from stdin */
	for (i = 0; i < n_threads; i++) {
		id[i] = i;
		pthread_create(&break_zip_threads[i], NULL, break_zip_thread, (void*) &id[i]);
	}

	/* Waits for all threads to end before returning result */
	for (i = 0; i < n_threads; i++)
		pthread_join(break_zip_threads[i], NULL);

}

double rtclock() {

    struct timezone Tzp;
    struct timeval Tp;

    int stat;

    stat = gettimeofday (&Tp, &Tzp);

    if (stat != 0) printf("Error return from gettimeofday: %d",stat);

    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


int main ()
{
	double t_start, t_end;

	scanf("%u", &n_threads);
	scanf("%s", filename);

	/* Allocates memory for thread handlers */
	break_zip_threads = (pthread_t*) malloc (n_threads * sizeof(pthread_t));
	/* Array of thread ids : avoids cast warnings on pthread_create calls */
	id = (unsigned int *) malloc (n_threads * sizeof(unsigned int));

	t_start = rtclock();
	/* Calls parallel function */
	break_zip_parallel();
	t_end = rtclock();

	fprintf(stdout, "%0.6lf\n", t_end - t_start);  

	free (break_zip_threads);
	free (id);
}
