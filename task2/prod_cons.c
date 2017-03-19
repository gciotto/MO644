/*
*	Task #2 - MO644 Parallel Programming
*	Gustavo CIOTTO PINTON - RA 117136
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* Parallel  producer_consumer funcion. It receives the number of threads in addition. */
void producer_consumer(int *buffer, int size, int *vec, int n, int n_threads) {
	int i, j;
	long long unsigned int sum = 0;
	
	/* We use a directive '# pragma omp for' for each internal loop, reducing the overhead to create/destroy threads every time. 
	This solution satisfies the exercise, once each thread acts as a producer and as a consumer according to i.
	A solution using locks was also implemented but its results were not satisfactory. */
#	pragma omp parallel num_threads(n_threads) default(none) shared(vec, size, n, buffer) private(i, j) reduction(+:sum)
	for(i=0;i<n;i++) {
		if(i % 2 == 0) {	// PRODUTOR
#	pragma omp for
			for(j=0;j<size;j++) {
				buffer[j] = vec[i] + j*vec[i+1];
			}
		}
		else {	// CONSUMIDOR
#	pragma omp for
			for(j=0;j<size;j++) {
				sum += buffer[j];
			}
		}
	}

	printf("%llu\n",sum);
}

/* Provided main function */
int main(int argc, char * argv[]) {
	double start, end;
	int i, n, size, nt;
	int *buff;
	int *vec;

	scanf("%d %d %d",&nt,&n,&size);

	buff = (int *)malloc(size*sizeof(int));
	vec = (int *)malloc(n*sizeof(int));

	for(i=0;i<n;i++)
		scanf("%d",&vec[i]);
	
	start = omp_get_wtime();
	producer_consumer(buff, size, vec, n, nt);
	end = omp_get_wtime();

	printf("%lf\n",end-start);

	free(buff);
	free(vec);

	return 0;
}

/*

1) Executar o comando cat /proc/cpuinfo no linux ou lscpu

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
CPU MHz:               839.367
CPU max MHz:           2200.0000
CPU min MHz:           800.0000
BogoMIPS:              4391.01
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              3072K
NUMA node0 CPU(s):     0-3
Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer xsave avx lahf_lm epb tpr_shadow vnmi flexpriority ept vpid xsaveopt dtherm arat pln pts


2) Executar a compilacao na maquina local utilizando a flag -g -pg para depois usar o gprof.

Comando do gprof: gprof -b T2_ser gmon.out

2.1) arq1.in

Flat profile:

Each sample counts as 0.01 seconds.
 no time accumulated

  %   cumulative   self              self     total           
 time   seconds   seconds    calls  Ts/call  Ts/call  name    
  0.00      0.00     0.00        1     0.00     0.00  producer_consumer

Call graph

granularity: each sample hit covers 2 byte(s) no time propagated

index % time    self  children    called     name
                0.00    0.00       1/1           main [7]
[1]      0.0    0.00    0.00       1         producer_consumer [1]
-----------------------------------------------

2.2) arq2.in

Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  ms/call  ms/call  name    
100.78      0.44     0.44        1   443.41   443.41  producer_consumer

Call graph

granularity: each sample hit covers 2 byte(s) for 2.26% of 0.44 seconds

index % time    self  children    called     name
                0.44    0.00       1/1           main [2]
[1]    100.0    0.44    0.00       1         producer_consumer [1]
-----------------------------------------------
                                                 <spontaneous>
[2]    100.0    0.00    0.44                 main [2]
                0.44    0.00       1/1           producer_consumer [1]
-----------------------------------------------

2.3) arq3.in

Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls   s/call   s/call  name    
100.78      4.50     4.50        1     4.50     4.50  producer_consumer

Call graph

granularity: each sample hit covers 2 byte(s) for 0.22% of 4.50 seconds

index % time    self  children    called     name
                4.50    0.00       1/1           main [2]
[1]    100.0    4.50    0.00       1         producer_consumer [1]
-----------------------------------------------
                                                 <spontaneous>
[2]    100.0    0.00    4.50                 main [2]
                4.50    0.00       1/1           producer_consumer [1]
-----------------------------------------------

3) Executar as flag -O0, -O1, -O2 e -O3 sem usar OpenMP com o programa serial e mostrar se teve algum ganho.

3.1) arq1.in

-O0:
t_serial = 0.007421

-O1: 
t_paralelo =  0.003031
t_serial = 0.001096
speedup (tO0/t01) = 6.771

-O2:
t_paralelo = 0.004332
t_serial = 0.001129
speedup (tO0/t02) = 6.573

-O3:
t_paralelo = 0.004701
t_serial = 0.000499
speedup (tO0/t03) = 14.892

3.2) arq2.in

-O0:
t_serial = 0.448865

-O1:
t_paralelo = 0.239290
t_serial = 0.106314
speedup (tO0/t01) = 4.222

-O2:
t_paralelo = 0.231441
t_serial = 0.085977
speedup (tO0/t02) = 5.221

-O3:
t_paralelo = 0.247042
t_serial = 0.051453
speedup (tO0/t03) = 8.7238


3.3) arq3.in

-O0:
t_serial = 4.350418

-O1:
t_paralelo = 2.287855
t_serial = 1.046652
speedup (tO0/t01) = 4.157

-O2:
t_paralelo = 2.240046
t_serial = 0.812152
speedup (tO0/t02) = 5.357

-O3:
t_paralelo = 2.372490
t_serial = 0.490515
speedup (tO0/t03) = 8.869

*/
