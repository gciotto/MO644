/*
 * CudaDynamicSearch.cpp
 *
 *  Created on: May 14, 2017
 *      Author: gciotto
 */

#include "CudaDynamicSearch.h"
#include "RingElement.h"
#include <cmath>
#include <iostream>

typedef struct {

        unsigned int type;
        double length, focal_distance, sextupole_strength;

} ring_element_t;

__global__ void dynamicSearchKernel(ring_element_t* c, pos_t *d, unsigned int turns, unsigned int repeat, unsigned int size) {

        double x[2] = {-0.10, +0.10};
        double y[2] = {+0.000, +0.003};

        int i = blockDim.x * blockIdx.x + threadIdx.x,
            j = blockDim.y * blockIdx.y + threadIdx.y;

        if (i < N_POINTS_X && j < N_POINTS_Y) {

                double posx = x[0] + i*(x[1] - x[0])/(N_POINTS_X - 1),
                       posy = y[0] + j*(y[1] - y[0])/(N_POINTS_Y - 1);

                pos_t r = {posx, 0, posy, 0, 0, 0};

                for (unsigned int k = 0; k < turns; k++)
                        for (unsigned int l = 0; l < repeat; l++)
                                for (unsigned int m = 0; m < size; m++) {

                                        ring_element_t aux = c[m];

                                        if (aux.type == RingElement::DRIFT) {
                                              	r[0] += aux.length * r[1];
                                        	r[2] += aux.length * r[3];
                                        }
                                        else if (aux.type == RingElement::QUADRUPOLE) {
                                        	r[1] += -r[0]/aux.focal_distance;
                                        	r[3] += +r[2]/aux.focal_distance;
                                        }
                                        else if (c[m].type == RingElement::SEXTUPOLE) {
                                        	r[1] += aux.sextupole_strength * aux.length * (r[0]*r[0] - r[2]*r[2]);
                                        	r[3] += aux.sextupole_strength * aux.length * 2 * r[0]*r[2];
                                        }
                                }

                for (unsigned int k = 0; k < 6; k++)
                        d[i * N_POINTS_X + j][k] = r[k];
        }
}


CudaDynamicSearch::~CudaDynamicSearch() {

    /* Clears everything up */
    this->ring.clear();
}

int CudaDynamicSearch::dynamical_aperture_search() {
      
        pos_t *host_result = (pos_t*) malloc (N_POINTS_X * N_POINTS_Y * sizeof(pos_t)), 
	      *cuda_result = NULL;
        ring_element_t *ring_element = (ring_element_t*) malloc (this->ring.size() * sizeof(ring_element_t)),
                       *cuda_ring_element = NULL;

        for (unsigned int i = 0; i < this->ring.size(); i++) {
                ring_element[i].type = this->ring[i]->getType();
                ring_element[i].length = this->ring[i]->getLength();

                if (ring_element[i].type == RingElement::QUADRUPOLE)
                        ring_element[i].focal_distance = ((Quadrupole*) this->ring[i])->getFocalDistance();
                else if (ring_element[i].type == RingElement::SEXTUPOLE)
                        ring_element[i].sextupole_strength = ((Sextupole*) this->ring[i])->getSextupoleStrength();
                
        }

	/* Allocates memory in the device */
	cudaMalloc ((void**) cuda_ring_element, sizeof(pos_t));

	/* Copies ring element array to the device */

	cudaMalloc ((void**) &cuda_result, N_POINTS_X * N_POINTS_Y * sizeof(pos_t));

	/* Computes grid dimension */
	dim3 dimGrid( std::ceil( (float) N_POINTS_X / CudaDynamicSearch::THREAD_PER_BLOCK ), std::ceil( (float) N_POINTS_Y / CudaDynamicSearch::THREAD_PER_BLOCK));

	/* Computes block dimensions (X, Y) */
	dim3 dimBlock(CudaDynamicSearch::THREAD_PER_BLOCK, CudaDynamicSearch::THREAD_PER_BLOCK);

	/* Copies result to the host */
	// dynamicSearchKernel<<< dimGrid, dimBlock >>>(, cuda_result, this->turns, this->repeat, this->ring.size());

        cudaMemcpy(host_result, cuda_result, sizeof(pos_t), cudaMemcpyDeviceToHost);

        free(ring_element);
	free(host_result);

        cudaFree(cuda_ring_element);
        cudaFree(cuda_result);

	return 0;

}

