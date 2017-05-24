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

#define GPU 0

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

# pragma omp target device(GPU) \
                    map (from: x1[:N], x2[:N]) \
                    map (to: n, m)
# pragma omp parallel for
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

