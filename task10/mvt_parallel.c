/*
* Parallel Computing - Gustavo Ciotto RA117136
* Task #10
*/
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#include <omp.h>

#ifdef MEDIUM
  #define N 2048
#elif LARGE
  #define N 4096
#elif EXTRALARGE
  #define N 8192
#endif

#define GPU 1

double rtclock()
{
        struct timezone Tzp;
        struct timeval Tp;
        int stat;
        stat = gettimeofday (&Tp, &Tzp);
        if (stat != 0) printf("Error return from gettimeofday: %d",stat);
        return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

void init_array(float *A,float *x1,float *x2,float *y1,float *y2){
        int i,j;


/* We could have added these two directives, but there is no increase in performance 
# pragma omp target device(GPU) map (from: x1[:N], x2[:N], y1[:N], y2[:N], A[:N*N])
# pragma omp parallel for 
*/
        for(i = 0 ; i < N ; i++){
                x1[i] = ((float)i)/N;
                x2[i] = ((float)i + 1)/N;
                y1[i] = ((float)i + 3)/N;
                y2[i] = ((float)i + 4)/N;
                for(j = 0 ; j < N ; j++)
                A[i*N + j] = ((float)i*j)/N;
        }

}

void runMvt(float *a,float *x1,float *x2,float *y1,float *y2){

        int i , j;

/* Maps data to the GPU. It's better to copy the data at once. We do not suffer from offerloading delays twice. */
# pragma omp target data device(GPU) \
                    map (to: y1[:N], a[:N*N], y2[:N]) \
                    map (tofrom: x1[:N], x2[:N])
{

        /* The following two directives move computing to the device */

	/* collapse(1) explanation in the end of the file */
        # pragma omp parallel for collapse(1)
          for(i = 0; i < N ; i++)
            for(j = 0 ; j < N ; j++)
              x1[i] += a[i*N + j] * y1[j];

        # pragma omp parallel for collapse(1)
          for(i = 0; i < N ; i++)
            for(j = 0 ; j < N ; j++)
              x2[i] += a[j*N + i] * y2[j];
}

}

int main(){

        double t_start, t_end;

        float *A,*x1,*x2,*y1,*y2;
        A = (float*)malloc( N * N * sizeof(float) );
        x1 = (float*)malloc( N * sizeof(float) );
        x2 = (float*)malloc( N * sizeof(float) );
        y1 = (float*)malloc( N * sizeof(float) );
        y2 = (float*)malloc( N * sizeof(float) );

        init_array(A,x1,x2,y1,y2);

        t_start = rtclock();
        runMvt( A , x1 , x2 , y1 , y2 );
        t_end = rtclock();

        float m = 0 , n = 0;

        for(int i = 0 ; i < N ; i++)
        m += x1[i] , n += x2[i];

        fprintf(stdout, "%0.4lf  %0.4lf\n", m, n);
        fprintf(stdout, "%0.4lf\n", t_end - t_start);

        free(A);
        free(x1);
        free(x2);
        free(y1);
        free(y2);
}

/**

Analise dos resultados
----------------------

Os dados da tabela 1, a seguir, foram coletados a partir da execucao de um programa paralelizado atraves de diretivas OpenMP e compilado pelo compilador
aclang, desenvolvido por Marcio M Pereira no Instituto de Computacao da Unicamp. No total, foram realizados 9 testes, que abordaram tamanho variados de entradas e 
diferentes tecnicas de otimizacao implementados pelo aclang e especificados pelo usuario por flags especificas. A primeira flag, none, comunica ao compilador
que nenhuma tecnica de otimizacao deve ser utilizada. A segunda e a terceira, por sua vez, ativam, respectivamete, as otimizacoes de tiling e vetorizacao. 

A tabela 2 apresenta os resultados para as execucoes seriais do programa original.

Tabela 1: Media dos tempos de execucao (em s) obtidos para diferentes entradas e metodos de otimizacao
---------

Entrada \ Otimiz.	none		tiling		vectorization
MEDIUM			0.02316		0.02060		0.02092
LARGE			0.03370		0.03378 	0.03426
EXTRA_LARGE		0.07694		0.07694		0.07870

Tabela 2: Tempos de execucao para as execucoes seriais
---------
Entrada 	Tempo (em s)
MEDIUM		0.0311		
LARGE		0.1986
EXTRA_LARGE	0.8863

Enfim, combinando-se as duas tabelas acima, obtem-se a tabela 3, que contem os speedups obtidos nas paralelizacoes.

Tabela 3: Speedups obtidos pela paralelizacao
---------
Entrada \ Otimiz.	none		tiling		vectorization
MEDIUM			1.34283		1.50970 	1.48661
LARGE			5.89317		5.87921		5.79684		
EXTRA_LARGE		11.5193		11.2904		11.2617

Observa-se que quanto maior o conjunto de entrada maior é o speedup. Isso pode ser explicado pelo fato de que a fracao entre o tempo de overhead gerado pela paralelizacao e
o proprio tempo de execucao torna-se cada vez mais pequena a medida que o conjunto de dados eh maior. A tabela 4, abaixo, que contem as porcentagens entre os tempos de transferencia de
dados (_cl_offloading_read_* + _cl_read_buffer) para o dispositivo e o tempo de execucao serial, reflete bem esta afirmacao, ja que, para entradas menores, o razao correspondente a 
transferencia eh superior. Por fim, a tecnica de otimizacao que apresentou melhores aumentos de performance eh a de vetorizacao.

Tabela 4: Porcentagens entre os tempos de transferencia de dados (_cl_offloading_read_* + _cl_read_buffer) para o dispositivo e o tempo de execucao serial. Valor absoluto, em s, entre parenteses 
---------
Entrada \ Otimiz.	none			tiling			vectorization
MEDIUM			0.09293 (0.00289)	0.09015	(0.00280)	0.09125 (0.00283)
LARGE			0.05160	(0.01024)	0.05275 (0.01047)	0.05273 (0.01047)
EXTRA_LARGE		0.04098 (0.03632)	0.04093 (0.03628)	0.04117 (0.03649)

Uso de collapse(1)
------------------

O uso de collapse produziu o seguinte kernel em openCL

__kernel void kernel_4ZxIoe(__global float *y1, __global float *a, __global float *y2, __global float *x1, __global float *x2, int _UB_0, int _MIN_0, int _INC_0, int j)
{
  int _ID_0 = get_global_id(0);
  int i = (_INC_0 * _ID_0) + _MIN_0;
  if (_ID_0 < _UB_0)
  {
    for (j = 0; j < 2048; j++)
      x1[i] += a[(i * 2048) + j] * y1[j];

    ;
  }
}

enquanto que sua ausencia, por sua vez, resultou no seguinte codigo

__kernel void kernel_JD3JIP(__global float *y1, __global float *a, __global float *y2, __global float *x1, __global float *x2, int _UB_0, int _MIN_0, int _INC_0, int _UB_1, int _MIN_1, int _INC_1)
{
  int _ID_0 = get_global_id(0);
  int _ID_1 = get_global_id(1);
  int i = (_INC_0 * _ID_0) + _MIN_0;
  int j = (_INC_1 * _ID_1) + _MIN_1;
  if ((_ID_0 < _UB_0) && (_ID_1 < _UB_1))
  {
    x1[i] += a[(j * 2048) + i] * y1[j];
  }

}

Quando utilizamos o parametro collapse(n), estamos indicando ao compilador que n loops internos devem incorporados ao 
codigo que sera dividido entre as threads. No exemplo acima, verificamos, no primeiro caso, exatamente esta afirmacao
a medida que todo o loop interno eh copiado para dentro do kernel. No segundo, por outro lado, realiza-se apenas uma soma, 
o que produzia um resultado invalido e diferente a cada execucao do programa.
**/
