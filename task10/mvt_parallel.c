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

# pragma omp target device(GPU) map (to: x1[:N], x2[:N], y1[:N], y2[:N], A[:N*N])
# pragma omp parallel for
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

Tabela 1: Tempos de execucao (em s) obtidos para diferentes entradas e metodos de otimizacao
---------

Entrada \ Otimiz.	none		tiling		vectorization
MEDIUM			0.0301		0.0304		0.0302
LARGE			0.1930		0.1915		0.1919
EXTRA_LARGE		0.8688		0.7680		0.8437

Tabela 2: Tempos de execucao para as execucoes seriais
---------
Entrada 	Tempo (em s)
MEDIUM		0.0369		
LARGE		0.1979
EXTRA_LARGE	0.8588

Enfim, combinando-se as duas tabelas acima, obtem-se a tabela 3, que contem os speedups obtidos nas paralelizacoes.

Tabela 3: Speedups obtidos pela paralelizacao
---------
Entrada \ Otimiz.	none		tiling		vectorization
MEDIUM			1.2260		1.2138		1.2219
LARGE			1.0254		1.0334		1.0317		
EXTRA_LARGE		0.9885		1.1182		1.0179

Observa-se que nenhuma das optimizacoes propostas obteve um ganho de desempenho consideravel. Isso pode ser justificado pelo overhead gerado justamente para introduzir e produzir o codigo paralelo. Mesmo em grandes entradas, tais como a EXTRALARGE, nao foram obtidos quaisquer ganhos satisfatorios.


**/
