#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

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

//# pragma omp target device(GPU) map (to: x1[:N], x2[:N], y1[:N], y2[:N], A[:N*N])
//# pragma omp parallel for
  for(i = 0 ; i < N ; i++){
    x1[i] = ((float)i)/N;
    x2[i] = ((float)i + 1)/N;
    y1[i] = ((float)i + 3)/N;
    y2[i] = ((float)i + 4)/N;
    for(j = 0 ; j < N ; j++)
      A[i*N + j] = ((float)i*j)/N;
  }
  return;
}

void runMvt(float *a,float *x1,float *x2,float *y1,float *y2){
  int i , j;

# pragma omp target device(GPU) \
                    map (from: y1[:N], a[:N*N]) \
                    map (tofrom: x1[:N])
# pragma omp parallel for
  for(i = 0; i < N ; i++)
    for(j = 0 ; j < N ; j++)
      x1[i] += a[i*N + j] * y1[j];

# pragma omp target device(GPU) \
                    map (from: y2[:N], a[:N*N]) \
                    map (tofrom: x2[:N])
# pragma omp parallel for
  for(i = 0; i < N ; i++)
    for(j = 0 ; j < N ; j++)
      x2[i] += a[j*N + i] * y2[j];

  return;
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
MEDIUM			0.02546		0.02616		0.02458
LARGE			0.06090		0.05080 	0.07932
EXTRA_LARGE		0.16762		0.16760		0.16934

Tabela 2: Tempos de execucao para as execucoes seriais
---------
Entrada 	Tempo (em s)
MEDIUM		0.0280		
LARGE		0.1660
EXTRA_LARGE	0.9121

Enfim, combinando-se as duas tabelas acima, obtem-se a tabela 3, que contem os speedups obtidos nas paralelizacoes.

Tabela 3: Speedups obtidos pela paralelizacao
---------
Entrada \ Otimiz.	none		tiling		vectorization
MEDIUM			1.09976		1.07034		1.13913
LARGE			2.72578		3.26771		2.09279		
EXTRA_LARGE		5.44147		5.44212		5.38621

Observa-se que quanto maior o conjunto de entrada maior é o speedup. Isso pode ser explicado pelo fato de que a fracao entre o tempo de overhead gerado pela paralelizacao e
o proprio tempo de execucao torna-se cada vez mais pequena a medida que o conjunto de dados eh maior. A tabela 4, abaixo, que contem as porcentagens entre os tempos de transferencia de
dados (_cl_offloading_read_write + _cl_read_buffer) para o dispositivo e o tempo de execucao serial, reflete bem esta afirmacao, ja que, para entradas menores, o razao correspondente a 
transferencia eh superior. Por fim, a tecnica de otimizacao que apresentou melhores aumentos de performance eh a de tiling, com destaque para a entrada LARGE.

Tabela 4: Porcentagens entre os tempos de transferencia de dados (_cl_offloading_read_write + _cl_read_buffer) para o dispositivo e o tempo de execucao serial
---------
Entrada \ Otimiz.	none		tiling		vectorization
MEDIUM			0.25067		0.25287		0.25498
LARGE			0.15276		0.15586 	0.15567
EXTRA_LARGE		0.13072		0.12658		0.12973

**/
