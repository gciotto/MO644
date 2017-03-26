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

/* Variables shared among all the threads. All variables which were passed as parameters to count()
   became global. */
/* 'local_vet' is an array of nthreads integer arrays. Each thread uses one these arrays
   to compute the histograms locally.
   'semaphores' is an array of sem_t and contains one semaphore per thread.
   'bin_thread' is an arrays containing the threads ids */
int n, nval, nthreads, *vet, **local_vet;
double h, *val, max, min
/* Semaphore array */;
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
		/* Computes the share of the input array which will be used by this thread*/
		local_size = nval / nthreads,
		i_start = rank * local_size,
		i_end = (rank == nthreads - 1 ? nval - 1 : (rank + 1) * local_size - 1);

	for(j = 0; j < n; j++) {
		count = 0;
		min_t = min + j*h;
		max_t = min + (j+1)*h;
		/* Iterates only over a part of the input array */
		for(i = i_start; i <= i_end; i++) {
			if(val[i] <= max_t && val[i] > min_t) {
				count++;
			}
		}

		/* Updates local bin array */
		local_vet[rank][j] = count;
	}

	/* Parallel algorithm according to histogram.pdf */
	if (rank % 2) /* Odd threads: increments semaphore and exits */
		sem_post(&semaphores[rank]);

	else { /* Even threads: combines local_vet computed in other threads  */
	
		int next = 1;
	
		do {

			/* Waits for other thread to compute its internal array */
			sem_wait(&semaphores[rank+next]);

			/* joins two internal arrays from two different threads*/
			for(j = 0; j < n; j++)
				local_vet[rank][j] += local_vet[rank + next][j];

			next *= 2;

		} while (rank + next < nthreads && (rank / next) % 2 != 1);

		/* Signs this thread finished its computing */
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

	/* Waits for all threads to end before returning result */
	for (i = 0; i < nthreads; i++)
		pthread_join(bin_thread[i], NULL);

	/* Final result is stored in local_vet[0] */
	return local_vet[0];
}

/* Allocates and initializes structures (semaphores, internal arrays ...) used in parallel computing */
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

	/* Allocates memory for each thread's internal bin array  */
	local_vet = (int **) malloc (nthreads * sizeof(int*));
	/* Allocates memory for the semaphores */
	semaphores = (sem_t*) malloc (nthreads * sizeof(sem_t));
	/* Allocates memory for thread handlers */
	bin_thread = (pthread_t*) malloc (nthreads * sizeof(pthread_t));

	/* Initializes semaphores and internal arrays */
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

	/* count function does not need parameters anymore: all variables shared variables became global */
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

	/* Frees used memory and semaphores */
	for (i = 0; i < nthreads; i++) {
		free (local_vet[i]);
		sem_destroy(&semaphores[i]);
	}

	free(bin_thread);
	free(semaphores);
	free(local_vet);
	free(val);

	return 0;
}

/* 

Tarefa complementar

Resultado do Comando 'lscpu'

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
CPU MHz:               921.679
CPU max MHz:           2200.0000
CPU min MHz:           800.0000
BogoMIPS:              4391.00
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              3072K


Numero de cores disponiveis = p = 4
Speedup = Ts / Tp
Eficiencia = Ts / (p * Tp)

Tabela 1: speedup e eficiencia para diversas execucoes em funcao 
do numero de threads e arquivo de entrada.
		Threads		1		2		4		8		16
arq1.in	speedup		1.000	1.713	1.553	1.551	1.049
		eficiencia	1.000	0.856	0.388	0.388	0.262
arq2.in	speedup		1.000	1.751	2.158	2.545	2.447
		eficiencia	1.000	0.876	0.540	0.636	0.612
arq3.in	speedup		1.000	2.020	2.676	2.549	2.559
		eficiencia	1.000	1.010	0.669	0.637	0.640

Uma analise da execucao serial do programa hist_s atraves do uitlitario gprof nos 
fornece a seguinte tabela:

comando 'gprof -b -p hist_s gmon.out'

Tabela 2: resultado do gprof, executado com o programa serial.
Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  ms/call  ms/call  name    
100.47      0.58     0.58        1   582.75   582.75  count
  0.00      0.58     0.00        1     0.00     0.00  max_val
  0.00      0.58     0.00        1     0.00     0.00  min_val

Conforme esperado, todo o tempo de execucao eh realizado na funcao count(). Dessa forma, conforme realizado,
tal funcao foi a escolhida para ser paralelizada.

A Tabela 1 fornece informacoes sobre o speedup e eficiencia para diversos valores de threads e arquivos de
entrada. De imediato, eh possivel observar que as duas metricas crescem a medida que o tamanho do vetor de entrada
aumenta (arq3.in possui um vetor com o maior numero de entradas). Alem do tempo gasto para a criacao e lancamento das threads
ser relativamente menor, isto eh, a fracao entre o tempo deste overhead e da computacao na execucao de arq3.in eh menor que aquele
em arq1.in, a carga eh melhor balanceada entre as threads. Para o caso 1, por exemplo, que contem apenas 1000 elementos em seu
vetor, se utilizarmos 2 threads, cada uma recebe somente 500 e tal divisao quase nao compensa os custos requiridos para sua execucao
em paralelo.

Eh possivel visualizar tambem que as metricas para 4, 8 e 16 sao bastante similares para os arquivos arq2.in e arq3.in. Isso ocorre
pelo motivo de que minha maquina possui apenas 4 unidades de processamento e, portanto, execucoes que utilizem um numero de threads 
superior a esse numero nao rodarao verdadeiramente em paralelo, visto que nem todas as threads farao uso de seu proprio core. Para o 
caso de arq1.in, o desempenho para 16 threads foi quase igual aquele serial. Isso pode ser explicado pelo desbalanceamento entre threads
(1000 nao eh divisivel por 16, logo a ultima thread calcula 8 elementos a mais que as anteriores) e pela elevada fracao 
t_overhead/t_processamento.

A seguinte tabela foi gerada a partir do comando perf para arq3.in e 16 threads:

Tabela 3: resultado do comando perf para 16 threads e arq3.in
 Performance counter stats for './hist_paralelo':

        855.470588      task-clock:u (msec)       #    3.193 CPUs utilized          
                 0      context-switches:u        #    0.000 K/sec                  
                 0      cpu-migrations:u          #    0.000 K/sec                  
               409      page-faults:u             #    0.478 K/sec                  
     1,721,951,131      cycles:u                  #    2.013 GHz                      (23.31%)
       989,150,492      stalled-cycles-frontend:u #   57.44% frontend cycles idle     (24.79%)
       416,818,213      stalled-cycles-backend:u  #   24.21% backend cycles idle      (24.64%)
     1,319,722,172      instructions:u            #    0.77  insn per cycle         
                                                  #    0.75  stalled cycles per insn  (29.73%)
       221,384,305      branches:u                #  258.787 M/sec                    (27.70%)
        16,357,631      branch-misses:u           #    7.39% of all branches          (29.22%)
       599,079,020      L1-dcache-loads:u         #  700.292 M/sec                    (24.07%)
         7,627,535      L1-dcache-load-misses:u   #    1.27% of all L1-dcache hits    (24.23%)
            53,172      LLC-loads:u               #    0.062 M/sec                    (21.94%)
   <not supported>      LLC-load-misses:u                                           
   <not supported>      L1-icache-loads:u                                           
            12,503      L1-icache-load-misses:u                                       (28.74%)
       604,154,943      dTLB-loads:u              #  706.225 M/sec                    (21.19%)
             3,959      dTLB-load-misses:u        #    0.00% of all dTLB cache hits   (20.10%)
               504      iTLB-loads:u              #    0.589 K/sec                    (16.26%)
               301      iTLB-load-misses:u        #   59.72% of all iTLB cache hits   (20.95%)
   <not supported>      L1-dcache-prefetches:u                                      
        11,471,308      L1-dcache-prefetch-misses:u #   13.409 M/sec                    (25.69%)

       0.267906490 seconds time elapsed

A seguinte tabela, por sua vez, foi gerada a partir do comando perf para arq3.in e 16 threads:

Tabela 4: resultado do comando perf para programa serial e arq3.in

 Performance counter stats for './hist_s':

        631.925079      task-clock:u (msec)       #    0.999 CPUs utilized          
                 0      context-switches:u        #    0.000 K/sec                  
                 0      cpu-migrations:u          #    0.000 K/sec                  
               370      page-faults:u             #    0.586 K/sec                  
     1,366,762,721      cycles:u                  #    2.163 GHz                      (26.86%)
       442,021,019      stalled-cycles-frontend:u #   32.34% frontend cycles idle     (26.89%)
       233,040,787      stalled-cycles-backend:u  #   17.05% backend cycles idle      (26.89%)
     1,465,985,615      instructions:u            #    1.07  insn per cycle         
                                                  #    0.30  stalled cycles per insn  (33.54%)
       230,631,598      branches:u                #  364.967 M/sec                    (33.54%)
        20,065,539      branch-misses:u           #    8.70% of all branches          (33.27%)
       721,269,617      L1-dcache-loads:u         # 1141.385 M/sec                    (31.82%)
         9,350,275      L1-dcache-load-misses:u   #    1.30% of all L1-dcache hits    (13.82%)
           449,044      LLC-loads:u               #    0.711 M/sec                    (13.75%)
   <not supported>      LLC-load-misses:u                                           
   <not supported>      L1-icache-loads:u                                           
            12,074      L1-icache-load-misses:u                                       (20.53%)
       726,230,512      dTLB-loads:u              # 1149.235 M/sec                    (19.95%)
             3,888      dTLB-load-misses:u        #    0.00% of all dTLB cache hits   (13.55%)
               393      iTLB-loads:u              #    0.622 K/sec                    (13.49%)
                89      iTLB-load-misses:u        #   22.65% of all iTLB cache hits   (20.14%)
   <not supported>      L1-dcache-prefetches:u                                      
        12,983,410      L1-dcache-prefetch-misses:u #   20.546 M/sec                    (26.72%)

       0.632806622 seconds time elapsed


Destacam-se acima as elevadas taxas de branch-misses (7.39% e 8.70%) e iTLB-load-misses (59.72% e 22.65%) para 
ambos os casos. Observa-se tambem que o IPC do programa serial eh ligeiramente superior aquele do paralelo: 1.07 
contra 0.77.

 */

