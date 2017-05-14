/*
 * ParallelTracking.cpp
 *
 *  Created on: May 14, 2017
 *      Author: gciotto
 */

#include "CudaDynamicSearch.h"

int main(int argc, char **argv) {

	CudaDynamicSearch c;

	c.createRing();

	c.dynamical_aperture_search();

	return 0;
}




