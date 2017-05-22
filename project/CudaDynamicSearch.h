/*
 * CudaDynamicSearch.h
 *
 *  Created on: May 14, 2017
 *      Author: gciotto
 */

#ifndef CUDADYNAMICSEARCH_H_
#define CUDADYNAMICSEARCH_H_

#include "DynamicSearch.h"
#include <cuda.h>

class CudaDynamicSearch : public DynamicSearch {
public:
	static const int THREAD_PER_BLOCK = 32;

	CudaDynamicSearch() : DynamicSearch() {
              	/* Copies ring element array to the device */
        	cudaMalloc ((void**) &this->cuda_result, N_POINTS_X * N_POINTS_Y * sizeof(pos_t));
        };

	CudaDynamicSearch(unsigned int e, double pThreshold, double angThreshold, double deviationEnergyThreshold,
							 unsigned int turns)
					: DynamicSearch(e, pThreshold, angThreshold,  deviationEnergyThreshold, turns) {
              	/* Copies ring element array to the device */
        	cudaMalloc ((void**) &this->cuda_result, N_POINTS_X * N_POINTS_Y * sizeof(pos_t));
        };

        CudaDynamicSearch(unsigned int e, unsigned int turns)
					: CudaDynamicSearch(e, POSITION_THRESHOLD, ANGULAR_THRESHOLD, DEVIATION_ENERGY_THRESHOLD, turns) {};

	virtual ~CudaDynamicSearch();

	int dynamical_aperture_search();
        void plot();

protected:
        pos_t *cuda_result = NULL;

};

#endif /* CUDADYNAMICSEARCH_H_ */
