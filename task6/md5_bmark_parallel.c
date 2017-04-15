/*
 *   MD5 Benchmark
 *   -------------
 *   File: md5_bmark.c
 *
 *   This is the main file for the md5 benchmark kernel. This benchmark was
 *   written as part of the StarBENCH benchmark suite at TU Berlin. It performs
 *   MD5 computation on a number of self-generated input buffers in parallel,
 *   automatically measuring execution time.
 *
 *   Copyright (C) 2011 Michael Andersch
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include "md5.h"
#include "md5_bmark.h"

typedef struct timeval timer;

/* Number of threads became global */
int nt; 

#define TIME(x) gettimeofday(&x, NULL)

/* Function declarations */
int initialize(md5bench_t* args);
int finalize(md5bench_t* args);
void run(md5bench_t* args);
void process(uint8_t* in, uint8_t* out, int bufsize);
void listInputs();
long timediff(timer* starttime, timer* finishtime);


// Input configurations
static data_t datasets[] = {
    {64, 512, 0},
    {64, 1024, 0},
    {64, 2048, 0},
    {64, 4096, 0},
    {128, 1024*512, 1},
    {128, 1024*1024, 1},
    {128, 1024*2048, 1},
    {128, 1024*4096, 1},
};

/*
 *   Function: initialize
 *   --------------------
 *   To initialize the benchmark parameters. Generates the input buffers from random data.
 */
int initialize(md5bench_t* args) {
    int index = args->input_set;
    if(index < 0 || index >= sizeof(datasets)/sizeof(datasets[0])) {
        fprintf(stderr, "Invalid input set specified! Clamping to set 0\n");
        index = 0;
    }

    args->numinputs = datasets[index].numbufs;
    args->size = datasets[index].bufsize;
    args->inputs = (uint8_t**)calloc(args->numinputs, sizeof(uint8_t*));
    args->out = (uint8_t*)calloc(args->numinputs, DIGEST_SIZE);
    if(args->inputs == NULL || args->out == NULL) {
        fprintf(stderr, "Memory Allocation Error\n");
        return -1;
    }

    //fprintf(stderr, "Reading input set: %d buffers, %d bytes per buffer\n", datasets[index].numbufs, datasets[index].bufsize);

    // Now the input buffers need to be generated, for replicability, use same seed
    srand(datasets[index].rseed);

    for(int i = 0; i < args->numinputs; i++) {

        args->inputs[i] = (uint8_t*)malloc(sizeof(uint8_t)*datasets[index].bufsize);
        uint8_t *p = args->inputs[i];
        if(p == NULL) {
            fprintf(stderr, "Memory Allocation Error\n");
            return -1;
        }
        for(int j = 0; j < datasets[index].bufsize; j++)
            *(p + j) = rand() % 255;

    }

    return 0;
}

/*
 *   Function: process
 *   -----------------
 *   Processes one input buffer, delivering the digest into out.
 */
void process(uint8_t* in, uint8_t* out, int bufsize) {
    MD5_CTX context;
    uint8_t digest[16];

    MD5_Init(&context);
    MD5_Update(&context, in, bufsize);
    MD5_Final(digest, &context);

    memcpy(out, digest, DIGEST_SIZE);
}

/*
 *   Function: run
 *   --------------------
 *   Main benchmarking function. If called, processes buffers with MD5
 *   until no more buffers available. The resulting message digests
 *   are written into consecutive locations in the preallocated output
 *   buffer.
 */
void run(md5bench_t* args) {

    for(int i = 0; i < args->iterations; i++) {
        int buffers_to_process = args->numinputs;
        int next = 0;
        uint8_t** in = args->inputs;
        uint8_t* out = args->out;

		/* This for loop is a DOALL loop, so it can be parallelized with openMP.
   		'# pragma omp parallel num_threads(nt)' allocates nt threads which will be used to run process() calls
		parallelly and '# pragma omp task' will start a task to do the job */
		# pragma omp parallel num_threads(nt)
		{		
			# pragma omp for
			for (next = 0; next < buffers_to_process; next++) {
				# pragma omp task
			    process(in[next], out+next*DIGEST_SIZE, args->size);
			}
		}
    }
}

/*
 *   Function: finalize
 *   ------------------
 *   Cleans up memory used by the benchmark for input and output buffers.
 */
int finalize(md5bench_t* args) {

    char buffer[64];

    for(int i = 0; i < args->numinputs; i++) {
#ifdef DEBUG
        sprintf(buffer, "Buffer %d has checksum ", i);
        fwrite(buffer, sizeof(char), strlen(buffer)+1, stdout);
#endif

        for(int j = 0; j < DIGEST_SIZE*2; j+=2) {
            sprintf(buffer+j,   "%x", args->out[DIGEST_SIZE*i+j/2] & 0xf);
            sprintf(buffer+j+1, "%x", args->out[DIGEST_SIZE*i+j/2] & 0xf0);
        }
        buffer[32] = '\0';

#ifdef DEBUG            
        fwrite(buffer, sizeof(char), 32, stdout);
        fputc('\n', stdout);
#else
        printf("%s ", buffer);
#endif

    }
#ifndef DEBUG
    printf("\n");
#endif

    if(args->inputs) {
        for(int i = 0; i < args->numinputs; i++) {
            if(args->inputs[i])
                free(args->inputs[i]);
        }

        free(args->inputs);
    }

    if(args->out)
        free(args->out);

    return 0;
}


/*
 *   Function: timediff
 *   ------------------
 *   Compute the difference between timers starttime and finishtime in msecs.
 */
long timediff(timer* starttime, timer* finishtime)
{
    long msec;
    msec=(finishtime->tv_sec-starttime->tv_sec)*1000;
    msec+=(finishtime->tv_usec-starttime->tv_usec)/1000;
    return msec;
}

/** MAIN **/
int main(int argc, char** argv) {

    timer b_start, b_end;
    md5bench_t args;

    //Receber parâmetros
    scanf("%d", &nt);
    scanf("%d", &args.input_set);
    scanf("%d", &args.iterations);
    args.outflag = 1;


    // Parameter initialization
    if(initialize(&args)) {
        fprintf(stderr, "Initialization Error\n");
        exit(EXIT_FAILURE);
    }

    TIME(b_start);

    run(&args);

    TIME(b_end);

    // Free memory
    if(finalize(&args)) {
        fprintf(stderr, "Finalization Error\n");
        exit(EXIT_FAILURE);
    }


    double b_time = (double)timediff(&b_start, &b_end)/1000;

    printf("%.3f\n", b_time);

    return 0;
}

/*

Justificativas:

Como utilizo uma distribuição Linux nao compativel com o Vtune (Archlinux, kernel versao 4.10.1-1 ), apresentarei
apenas as saidas do gprof e do perf.

Antes da paralelizacao, os comandos gprof e perf resultavam nas seguintes saídas para a entrada arq2.in:

Saida 1: Comando 'perf record ./md5 < arq2.in && perf report':

  77.95%  md5      md5               [.] body
   7.36%  md5      libc-2.25.so      [.] __random_r
   6.65%  md5      md5               [.] initialize
   5.85%  md5      libc-2.25.so      [.] __random
   1.43%  md5      libc-2.25.so      [.] rand
   [...]

As duas saidas demonstram que a funcao que mais consome a CPU e, portanto, que apresentara melhores ganhos
de performance caso paralelizada sera a body() contida em md5.c. Essa funcao, entretanto, apresenta um do-while 
loop do tipo doacross de dificil paralelizacao. Tal funcao body() eh chamada por MD5_Update(), uma funcao sem loops, que,
por sua vez, é invocada por process() de md5_bmark.c. Esta ultima tambem nao apresenta quaisquer lacos internos. A unica 
maneira, portanto, de paralelizarmos este benchmark eh modificarmos a funcao run(), que realiza diversas chamadas a process()
e, consequentemente, a body(). Os lacos internos de run (), ao contrario dos de body(), nao apresentam interdependencias
entre suas iteracoes, ou seja, sao do tipo doall e passiveis de paralelizacao. Adiciona-se assim, duas diretivas omp, uma
para alocar todas as threads e a outra para invocar uma task e computar as diversas chamdas de process() paralelamente.

O resultado apos a paralelizacao eh, portanto:

Saida 2: Comando 'perf record ./md5_parallel < arq2.in && perf report':
  84.41%  md5_parallel  md5_parallel      [.] body
   4.86%  md5_parallel  libc-2.25.so      [.] __random_r
   4.61%  md5_parallel  md5_parallel      [.] initialize
   4.05%  md5_parallel  libc-2.25.so      [.] __random
   0.93%  md5_parallel  libc-2.25.so      [.] rand
   [...]

A porcentagem correspondente a body() aumentou pelo motivo de que o tempo total do programa diminuiu.
*/
