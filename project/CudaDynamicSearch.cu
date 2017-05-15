/*
 * CudaDynamicSearch.cpp
 *
 *  Created on: May 14, 2017
 *      Author: gciotto
 */

#include "CudaDynamicSearch.h"
#include <cmath>

__global__ void dynamicSearchKernel(CudaDynamicSearch c) {

        double x[2] = {-0.10, +0.10};
        double y[2] = {+0.000, +0.003};

        int i = blockDim.x * blockIdx.x + threadIdx.x,
                j = blockDim.y * blockIdx.y + threadIdx.y;

        if (i < N_POINTS_X && j < N_POINTS_Y) {

                double posx = x[0] + i*(x[1] - x[0])/(N_POINTS_X - 1),
                           posy = y[0] + j*(y[1] - y[0])/(N_POINTS_Y - 1);

                pos_t r = {posx, 0, posy, 0, 0, 0};

                for (unsigned int k = 0; k < c.getTurns(); k++)
                        c.cuda_elements[k].pass(r);
        }
}


CudaDynamicSearch::~CudaDynamicSearch() {

	/* Clears everything up */
	this->ring.clear();

    cudaFree(this->cuda_elements);	
}

int CudaDynamicSearch::dynamical_aperture_search() {

	unsigned int byteCount = 0;

	this->size = this->ring.size();

	for (unsigned int i = 0; i < this->ring.size(); i++)
		/* As this->ring contains many different types of objects, we need to count its byte size individually */
		byteCount += sizeof(*this->ring[i]); /* ring[i] is a pointer to an object */

	/* Allocates memory in the device */
	cudaMalloc ((void**) this->cuda_elements, byteCount);

	/* Copies ring element array to the device */
	byteCount = 0;
	for (unsigned int i = 0; i < this->ring.size(); i++) {
		cudaMemcpy((uint8_t *) this->cuda_elements + byteCount, this->ring[i], sizeof(*this->ring[i]), cudaMemcpyHostToDevice);
		byteCount += sizeof(*this->ring[i]);
	}

	/* Computes grid dimension */
	dim3 dimGrid( std::ceil( (float) N_POINTS_X / CudaDynamicSearch::THREAD_PER_BLOCK ), std::ceil( (float) N_POINTS_Y / CudaDynamicSearch::THREAD_PER_BLOCK));

	/* Computes block dimensions (X, Y) */
	dim3 dimBlock(CudaDynamicSearch::THREAD_PER_BLOCK, CudaDynamicSearch::THREAD_PER_BLOCK);

	/* Copies result to the host */
	dynamicSearchKernel<<< dimGrid, dimBlock >>>(*this);

	return 0;

}