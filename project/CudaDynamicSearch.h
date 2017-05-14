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

	CudaDynamicSearch() : DynamicSearch() {};
	CudaDynamicSearch(unsigned int e, double pThreshold, double angThreshold, double deviationEnergyThreshold,
							 unsigned int turns)
					: DynamicSearch(e, pThreshold, angThreshold,  deviationEnergyThreshold, turns) {};
	virtual ~CudaDynamicSearch();

	int dynamical_aperture_search();
	__host__ __device__ int performOneTurn(pos_t &e);

private:
	__global__ void dynamicSearchKernel();
};

#endif /* CUDADYNAMICSEARCH_H_ */
