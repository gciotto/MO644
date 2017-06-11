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

/*

Machine Info:

Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                4
On-line CPU(s) list:   0-3
Thread(s) per core:    2
Core(s) per socket:    2
Socket(s):             1
NUMA node(s):          1
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 42
Model name:            Intel(R) Core(TM) i3-2330M CPU @ 2.20GHz
Stepping:              7
CPU MHz:               2199.865
CPU max MHz:           2200.0000
CPU min MHz:           800.0000
BogoMIPS:              4392.29
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              3072K
NUMA node0 CPU(s):     0-3
Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer xsave avx lahf_lm epb tpr_shadow vnmi flexpriority ept vpid xsaveopt dtherm arat pln pts

Tests results:

arq1.in:
Senha:10000
62.247894

arq2.in:
Senha:100000
623.336350

arq3.in:
Senha:450000
577.715237

arq4.in:
Senha:310000
485.998859

arq5.in:
Senha:65000
40.785986

arq6.in:
Senha:245999
898.885159

*/
