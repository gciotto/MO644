/*
 * CudaDynamicSearch.cpp
 *
 *  Created on: May 14, 2017
 *      Author: gciotto
 */

#include "CudaDynamicSearch.h"
#include "RingElement.h"
#include <math.h>
#include <iostream>
#include <fstream>


typedef struct {

        unsigned int type;
        double length, focal_distance, sextupole_strength;

} ring_element_t;

__global__ void summaryKernel(pos_t *r, double* s) {

        int j = blockDim.x * blockIdx.x + threadIdx.x,
            i = blockDim.y * blockIdx.y + threadIdx.y;

        if (i < N_POINTS_X && j < N_POINTS_Y) {

                int index = i * N_POINTS_X + j;

                double *t = r[index], sum = t[0] + t[1] + t[2] + t[3] + t[4] + t[5];

                if (!isfinite(sum))
                        s[index] = 1.0;
                else {
                        s[index] = 0.0;
                        for (unsigned int k = 0; k < 6; k++)
                                s[index] += t[k] * t[k];

                        s[index] = sqrt(s[index]);
                }
        }

}

__global__ void dynamicSearchKernel(ring_element_t* c, pos_t *d, unsigned int turns, unsigned int repeat, unsigned int size) {

        double x[2] = {-0.10, +0.10};
        double y[2] = {+0.000, +0.003};

        int j = blockDim.x * blockIdx.x + threadIdx.x,
            i = blockDim.y * blockIdx.y + threadIdx.y;

        if (i < N_POINTS_Y && j < N_POINTS_X) {

		int index = i * N_POINTS_X + j;

#ifdef CUDA_INTRINSICS
                double posx = __dadd_rn(x[0], __dmul_rn(i, __ddiv_rn(x[1] - x[0], N_POINTS_X - 1.0))),
                       posy = __dadd_rn(y[0], __dmul_rn(j, __ddiv_rn(y[1] - y[0], N_POINTS_Y - 1.0)));
#elif CUDA_FMA
		double  posx = __fma_rn(i, (x[1] - x[0]) * __drcp_rn(N_POINTS_X - 1.0), x[0]),
			posy = __fma_rn(j, (y[1] - y[0]) * __drcp_rn(N_POINTS_Y - 1.0), y[0]);
#else
                double posx = x[0] + i*(x[1] - x[0])/(N_POINTS_X - 1.0),
                       posy = y[0] + j*(y[1] - y[0])/(N_POINTS_Y - 1.0);
#endif

                pos_t r = {posx, 0, posy, 0, 0, 0};

                for (unsigned int k = 0; k < turns; k++) {
                        for (unsigned int l = 0; l < repeat; l++) 
                                for (unsigned int m = 0; m < size; m++) {

                                        ring_element_t aux = c[m];

                                        if (aux.type == RingElement::DRIFT) {

						#ifdef CUDA_INTRINSICS
						r[0] = __dadd_rn(r[0], __dmul_rn(aux.length, r[1]));
						r[2] = __dadd_rn(r[2], __dmul_rn(aux.length, r[3]));
						#elif CUDA_FMA
						r[0] = __fma_rn(aux.length, r[1], r[0]);
						r[2] = __fma_rn(aux.length, r[3], r[2]);
						#else
						r[0] += aux.length * r[1];
						r[2] += aux.length * r[3];
						#endif
                                        }
                                        else if (aux.type == RingElement::QUADRUPOLE) {

						#ifdef CUDA_INTRINSICS
						r[1] = __dadd_rn(r[1], __dmul_rn(-1.0, __ddiv_rn(r[0], aux.focal_distance)));
						r[3] = __dadd_rn(r[3], __ddiv_rn(r[2], aux.focal_distance));
						#elif CUDA_FMA
						r[1] = __fma_rn(-r[0], __drcp_rn(aux.focal_distance), r[1]);
						r[3] = __fma_rn( r[2], __drcp_rn(aux.focal_distance), r[3]);
						#else
						r[1] += -r[0]/aux.focal_distance;
						r[3] += r[2]/aux.focal_distance;
						#endif
                                        }
                                        else if (aux.type == RingElement::SEXTUPOLE) {

						#ifdef CUDA_INTRINSICS
						r[1] = __dadd_rn(r[1], __dmul_rn(__dmul_rn(aux.sextupole_strength, aux.length), __dadd_rn(__dmul_rn(r[0],r[0]), __dmul_rn(-1.0, __dmul_rn(r[2], r[2])))));
						r[3] = __dadd_rn(r[3], __dmul_rn(__dmul_rn(aux.sextupole_strength, aux.length), __dmul_rn(2.0, __dmul_rn(r[0], r[2]))));
						#elif CUDA_FMA
						r[1] = __fma_rn(aux.sextupole_strength * aux.length, (r[0]*r[0] - r[2]*r[2]), r[1]);
						r[3] = __fma_rn(aux.sextupole_strength * aux.length, 2.0 * r[0] * r[2], r[3]);
						#else
						r[1] += aux.sextupole_strength * aux.length * (r[0]*r[0] - r[2]*r[2]);
						r[3] += aux.sextupole_strength * aux.length * 2 * r[0]*r[2];
						#endif
                                        }
                                }
		}

                for (unsigned int k = 0; k < 6; k++)
                        d[index][k] = r[k];

        }
}


CudaDynamicSearch::~CudaDynamicSearch() {

    /* Clears everything up */
    this->ring.clear();

    cudaFree(this->cuda_result);
}

int CudaDynamicSearch::dynamical_aperture_search() {
      
        ring_element_t *ring_element = (ring_element_t*) malloc (this->ring.size() * sizeof(ring_element_t)),
                       *cuda_ring_element = NULL;

        for (unsigned int i = 0; i < this->ring.size(); i++) {
                ring_element[i].type = this->ring[i]->getType();
                ring_element[i].length = this->ring[i]->getLength();
                ring_element[i].focal_distance = 0;
                ring_element[i].sextupole_strength = 0;

                if (ring_element[i].type == RingElement::QUADRUPOLE)
                        ring_element[i].focal_distance = ((Quadrupole*) this->ring[i])->getFocalDistance();
                else if (ring_element[i].type == RingElement::SEXTUPOLE)
                        ring_element[i].sextupole_strength = ((Sextupole*) this->ring[i])->getSextupoleStrength();
        }

	/* Allocates memory in the device */
	cudaMalloc ((void**) &cuda_ring_element, this->ring.size() * sizeof(ring_element_t));
	cudaMemcpy(cuda_ring_element, ring_element, this->ring.size() * sizeof(ring_element_t), cudaMemcpyHostToDevice);

	/* Computes grid dimension */
	dim3 dimGrid( ceil( (float) N_POINTS_X / CudaDynamicSearch::THREAD_PER_BLOCK ), ceil( (float) N_POINTS_Y / CudaDynamicSearch::THREAD_PER_BLOCK));

	/* Computes block dimensions (X, Y) */
	dim3 dimBlock(CudaDynamicSearch::THREAD_PER_BLOCK, CudaDynamicSearch::THREAD_PER_BLOCK);

	/* Copies result to the host */
	dynamicSearchKernel<<< dimGrid, dimBlock >>>(cuda_ring_element, this->cuda_result, this->turns, this->repeat, this->ring.size());

        cudaMemcpy(this->result_set, this->cuda_result, N_POINTS_X * N_POINTS_Y * sizeof(pos_t), cudaMemcpyDeviceToHost);

	unsigned int p = 0;
	for (unsigned int i = 0; i < N_POINTS_X ; i++) 
		for (unsigned int j = 0; j < N_POINTS_Y ; j++){
			unsigned int index = i * N_POINTS_X + j;
			if (this->testSolution(this->result_set[index]))
				printf ("%f %f %f %f %f %f (%d / %d) - (%d)\n", this->result_set[index][0], this->result_set[index][1], this->result_set[index][2], this->result_set[index][3], 
                                                                        this->result_set[index][4], this->result_set[index][5], ++p , N_POINTS_X * N_POINTS_Y, index);
		
		}

        free(ring_element);

        cudaFree(cuda_ring_element);

	return 0;
}

void CudaDynamicSearch::plot() {

        if (this->result_set != NULL) {

                /* Allocates array for results */
                double *cuda_r, *host_r = (double*) malloc (N_POINTS_X * N_POINTS_Y * sizeof(double));
                std::ofstream out_file;

                cudaMalloc((void**) &cuda_r, N_POINTS_X * N_POINTS_Y * sizeof(double));

                /* Computes grid dimension */
                dim3 dimGrid( ceil( (float) N_POINTS_X / CudaDynamicSearch::THREAD_PER_BLOCK ), ceil( (float) N_POINTS_Y / CudaDynamicSearch::THREAD_PER_BLOCK));

                /* Computes block dimensions (X, Y) */
                dim3 dimBlock(CudaDynamicSearch::THREAD_PER_BLOCK, CudaDynamicSearch::THREAD_PER_BLOCK);

        	summaryKernel<<< dimGrid, dimBlock >>>(this->cuda_result, cuda_r);

                cudaMemcpy(host_r, cuda_r, N_POINTS_X * N_POINTS_Y * sizeof(double), cudaMemcpyDeviceToHost);

                out_file.open ("plot_cudadynamicsearch.dat");
                for (unsigned int i = 0; i < N_POINTS_X; i++) {
                        for (unsigned int j = 0; j < N_POINTS_Y; j++) {

                                int index = i * N_POINTS_X + j;
                                out_file << host_r[index] << " ";
                        }

                        out_file << std::endl;
                }

                out_file.close();

                free(host_r);
                cudaFree(cuda_r);
        }
}

