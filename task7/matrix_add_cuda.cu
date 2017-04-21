/*
  Task #7 - Gustavo Ciotto Pinton
  MO644 - Parallel Programming
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>

#define THREAD_PER_BLOCK 32 /* Tesla k40 supports 1024 threads (32 x 32 = 1024 in 2D grids) */

__global__ void addMatrix2d (int *A, int *B, int *C, int rows, int columns) {

	int col = blockDim.x * blockIdx.x + threadIdx.x, 
	    row = blockDim.y * blockIdx.y + threadIdx.y;

	int index = row * columns + col;

	if (col < columns && row < rows)
		C[index] = A[index] + B[index];

}

int main()
{

	int *A, *B, *C;
	/* Memory pointers used by the device */
	int *d_A, *d_B, *d_C;
	int i, j, m_size;

	/* Matrix dimensions */
	int linhas, colunas;

	scanf("%d", &linhas);
	scanf("%d", &colunas);

	m_size = sizeof(int) * linhas * colunas;

	A = (int *) malloc (m_size);
	B = (int *) malloc (m_size);
	C = (int *) malloc (m_size);

	for(i = 0; i < linhas; i++)
		for(j = 0; j < colunas; j++)
			A[i*colunas+j] =  B[i*colunas+j] = i+j;

	/* Allocating memory for CUDA pointers in the device */
	cudaMalloc( (void**) &d_A, m_size);
	cudaMalloc( (void**) &d_B, m_size);
	cudaMalloc( (void**) &d_C, m_size);

	/* Copying data into device memory */
	cudaMemcpy(d_A, A, m_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, m_size, cudaMemcpyHostToDevice);

	/* Computes block grid dimensions (X,Y) */
	dim3 dimGrid( ceil( (float) colunas / THREAD_PER_BLOCK ), ceil( (float) linhas / THREAD_PER_BLOCK));

	/* Computes block dimensions (X, Y) */
	dim3 dimBlock(THREAD_PER_BLOCK, THREAD_PER_BLOCK);

	/* Launches computing in GPU */
	addMatrix2d <<< dimGrid, dimBlock >>> (d_A, d_B, d_C, linhas, colunas);

	/* Copying computed result from device memory */
	cudaMemcpy(C, d_C, m_size, cudaMemcpyDeviceToHost);

	long long int somador = 0;
	
	//Manter esta computação na CPU
	for(i = 0; i < linhas; i++)
		for(j = 0; j < colunas; j++)
			somador += C[i*colunas+j];
    
	printf("%lli\n", somador);

	/* Cleaning everything up */
	free(A);
	free(B);
	free(C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

