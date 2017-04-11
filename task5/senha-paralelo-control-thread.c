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
#define BLOCK_SIZE 5000
#define DEFAULT_START 10000

struct queue_element {
	unsigned int i_start, i_end;

};

/* Circular queue */
struct queue {
	struct queue_element *circular_queue;
	unsigned int start, end, size;
	pthread_rwlock_t rwlock;
};

/* Shared variables used among all threads. If found is true, all threads stop their computing. */
unsigned int n_threads, *id, found = 0;
char filename[100];
const char finalcmd[300] = "unzip -P%d -t %s 2>&1";
pthread_t *break_zip_threads;
struct queue zip_queue;

/* Queue API */
/* Checks if queue q is empty. */
int isEmpty (struct queue *q) {

	int result;

	pthread_rwlock_rdlock(&q->rwlock);

	result = (q->start == q->end);

	pthread_rwlock_unlock(&q->rwlock);

	return result;
	
}

/* Checks if queue q is full. */
int isFull (struct queue *q) {

	int result;

	pthread_rwlock_rdlock(&q->rwlock);

	result = ( (q->end + 1) % q->size == q->start );

	pthread_rwlock_unlock(&q->rwlock);

	return result;
	
}

/* Enqueues a new element. Returns 1 if the queue is full. */
int enqueue(struct queue *q, unsigned int i_start, unsigned int i_end) {
	
	pthread_rwlock_wrlock(&q->rwlock);

	if ((q->end + 1) % q->size == q->start) {
		pthread_rwlock_unlock(&q->rwlock);
		return 1;
	}

	q->circular_queue[q->end].i_start = i_start;
	q->circular_queue[q->end].i_end = i_end;

	q->end = (q->end + 1) % q->size;

	pthread_rwlock_unlock(&q->rwlock);

	return 0;
}

/* Dequeues an element from the queue. Returns 1 if it's empty */
int dequeue(struct queue *q, struct queue_element *e) {
	
	pthread_rwlock_wrlock(&q->rwlock);

	if (q->start == q->end) {
		pthread_rwlock_unlock(&q->rwlock);
		return 1;
	}

	e->i_start = q->circular_queue[q->start].i_start;
	e->i_end = q->circular_queue[q->start].i_end;

	q->start = (q->start + 1) % q->size;

	pthread_rwlock_unlock(&q->rwlock);

	return 0;
}

FILE *popen(const char *command, const char *type);

/* Control Thread: enqueue ranges of elements that will be consumed by the other 
   threads. */
void *break_zip_control_thread (void *arg) {

	/* According to the exercise, the password starts from i = 10000. */
	int i = DEFAULT_START;
	
	while (!found) {

		/* Tries to enqueue a new element. Yields CPU if the queue is full. */
		while (enqueue (&zip_queue, i, i + BLOCK_SIZE - 1) && !found)
			pthread_yield();

		i += BLOCK_SIZE;

	}
	
	return NULL;

}

/* Thread function: iterates over a range of passwords. Sets found if it finds the password,
   signaling other threads to stop computing. */
void *break_zip_thread (void *rank) {

	FILE * fp;
	char ret[200], cmd[400];
	unsigned int my_rank = *((unsigned int*) rank);
	struct queue_element e;

	while (!found) {

		while (dequeue(&zip_queue, &e) && !found) 
			pthread_yield();

		unsigned int i;

		printf ("Thread #%u working from %u to %u\n", my_rank, e.i_start, e.i_end);

		for(i = e.i_start; i <= e.i_end && !found; i++){

			sprintf((char*)&cmd, finalcmd, i, filename);

			fp = popen(cmd, "r");	
			while (!feof(fp)) {
				fgets((char*)&ret, 200, fp);
				if (strcasestr(ret, "ok") != NULL) {
					printf("Senha:%d\n", i);
					/* Considering only one thread is executing this code, we do not need
					   to protect 'found' variable. */
					found = 1;
				}
			}
			pclose(fp);
		}

	}

	return NULL;

}

/* Creates, starts and waits all threads to finish */
void break_zip_parallel () {

	unsigned int i;

	/* Creates all threads, according to the first value read from stdin */
	pthread_create(&break_zip_threads[n_threads], NULL, break_zip_control_thread, NULL);

	for (i = 0; i < n_threads; i++) {
		id[i] = i;
		pthread_create(&break_zip_threads[i], NULL, break_zip_thread, (void*) &id[i]);
	}

	/* Waits for all threads to end before returning result */
	for (i = 0; i <= n_threads; i++)
		pthread_join(break_zip_threads[i], NULL);

}

double rtclock() {

    struct timezone Tzp;
    struct timeval Tp;

    int stat = gettimeofday (&Tp, &Tzp);

    if (stat != 0) printf("Error return from gettimeofday: %d",stat);

    return (Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


int main ()
{
	double t_start, t_end;

	scanf("%u", &n_threads);
	scanf("%s", filename);

	/* Allocates memory for thread handlers */
	break_zip_threads = (pthread_t*) malloc ((n_threads + 1)* sizeof(pthread_t));
	/* Array of thread ids : avoids cast warnings on pthread_create calls */
	id = (unsigned int *) malloc (n_threads * sizeof(unsigned int));
	/* Allocates queue */
	zip_queue.size = 2 * n_threads;
	zip_queue.circular_queue = (struct queue_element*) malloc (zip_queue.size * sizeof(struct queue_element));
	zip_queue.start = 0;
	zip_queue.end = 0;
	pthread_rwlock_init (&zip_queue.rwlock, NULL);

	t_start = rtclock();
	/* Calls parallel function */
	break_zip_parallel();
	t_end = rtclock();

	fprintf(stdout, "%0.6lf\n", t_end - t_start);  

	/* Frees and releases everything */
	free (break_zip_threads);
	free (id);
	free (zip_queue.circular_queue);
	pthread_rwlock_destroy(&zip_queue.rwlock);
}

/*

Machine Info:

Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                8
On-line CPU(s) list:   0-7
Thread(s) per core:    2
Core(s) per socket:    4
Socket(s):             1
NUMA node(s):          1
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 94
Model name:            Intel(R) Xeon(R) CPU E3-1270 v5 @ 3.60GHz
Stepping:              3
CPU MHz:               800.024
CPU max MHz:           4000,0000
CPU min MHz:           800,0000
BogoMIPS:              7202.00
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              8192K
NUMA node0 CPU(s):     0-7
Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch epb intel_pt tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx rdseed adx smap clflushopt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp


Tests results:

arq1.in:
Senha:10000
0.009257
Speedup: 5719.24446

arq2.in:
Senha:100000
119.072239
Speedup: 3.25051

arq3.in:
Senha:450000
333.466017
Speedup: 7.18077

arq4.in:
Senha:310000
302.835095
Speedup: 5.012969

arq5.in:
Senha:65000
23.988789
Speedup: 13.55020

arq6.in:
Senha:245999
175.049148
Speedup: 6.32518

*/
