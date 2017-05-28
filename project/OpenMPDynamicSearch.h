/*
 * OpenMPDynamicSearch.h
 *
 *  Created on: May 27, 2017
 *      Author: gciotto
 */

#ifndef OPENMPDYNAMICSEARCH_H_
#define OPENMPDYNAMICSEARCH_H_

#include "DynamicSearch.h"

const unsigned int N_THREADS = 8;

class OpenMPDynamicSearch: public DynamicSearch {

public:
	OpenMPDynamicSearch () : DynamicSearch() {
		this->n_threads = N_THREADS;
	};

	OpenMPDynamicSearch(unsigned int e, unsigned int turns)
	      : DefaultDynamicSearch(e, POSITION_THRESHOLD, ANGULAR_THRESHOLD, DEVIATION_ENERGY_THRESHOLD, turns) {
		this->n_threads = N_THREADS;
	};

	OpenMPDynamicSearch(unsigned int e, unsigned int turns, int threads)
		      : DefaultDynamicSearch(e, POSITION_THRESHOLD, ANGULAR_THRESHOLD, DEVIATION_ENERGY_THRESHOLD, turns) {
			this->n_threads = threads;
	};

	int dynamical_aperture_search();

protected:
	int n_threads;

};

#endif /* OPENMPDYNAMICSEARCH_H_ */
