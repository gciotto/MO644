/*
 * DynamicSearch.h
 *
 *  Created on: May 14, 2017
 *      Author: gciotto
 */

#ifndef DYNAMICSEARCH_H_
#define DYNAMICSEARCH_H_

#include <vector>
#include "RingElement.h"

const unsigned int DEFAULT_ELEMENT_COUNT = 2000;

const unsigned int DEFAULT_TURNS = 10;

const unsigned int N_POINTS_X = 100;
const unsigned int N_POINTS_Y = 100;

const double POSITION_THRESHOLD = 1.0;  		 // [1.0 meter]
const double ANGULAR_THRESHOLD = 0.01; 			 // [10 mrad]
const double DEVIATION_ENERGY_THRESHOLD  = 0.01; // [1% energy deviation]

class DynamicSearch {
public:
	DynamicSearch();
	DynamicSearch(	unsigned int e, double pThreshold,
					double angThreshold, double deviationEnergyThreshold,
					unsigned int turns );
	virtual ~DynamicSearch();

	void createRing();
	bool testSolution(pos_t r);

	virtual void performOneTurn(pos_t &e);
        virtual void plot();

	/* Abstract members */
	virtual int dynamical_aperture_search() = 0;

/* Protected members */
protected:
	std::vector<RingElement*> ring;

	unsigned int repeat, turns;

	double 	positionThreshold, angularThreshold, deviationEnergyThreshold;

        pos_t *result_set;
};

class DefaultDynamicSearch : public DynamicSearch {
public:

	DefaultDynamicSearch() : DynamicSearch() {};
	DefaultDynamicSearch(unsigned int e, double pThreshold, double angThreshold, double deviationEnergyThreshold,
						 unsigned int turns)
				: DynamicSearch(e, pThreshold, angThreshold,  deviationEnergyThreshold, turns) {};
	DefaultDynamicSearch(unsigned int e, unsigned int turns) 
                                : DefaultDynamicSearch(e, POSITION_THRESHOLD, ANGULAR_THRESHOLD, DEVIATION_ENERGY_THRESHOLD, turns) {};


	int dynamical_aperture_search();
};

#endif /* DYNAMICSEARCH_H_ */
