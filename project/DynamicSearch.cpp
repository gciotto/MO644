/*
 * DynamicSearch.cpp
 *
 *  Created on: May 14, 2017
 *      Author: gciotto
 */

#include "DynamicSearch.h"
#include <cmath>
#include <iostream>

/* DynamicSearch class */

DynamicSearch::DynamicSearch() {

	this->positionThreshold = POSITION_THRESHOLD;
	this->angularThreshold = ANGULAR_THRESHOLD;
	this->deviationEnergyThreshold = DEVIATION_ENERGY_THRESHOLD;

	this->repeat = DEFAULT_ELEMENT_COUNT;

	this->turns = DEFAULT_TURNS;

	this->ring.clear();
}

DynamicSearch::DynamicSearch(unsigned int e, double pThreshold,
					double angThreshold, double deviationEnergyThreshold,
					unsigned int turns )

					: DynamicSearch() {

	this->positionThreshold = pThreshold;
	this->angularThreshold = angThreshold;
	this->deviationEnergyThreshold = deviationEnergyThreshold;

	this->turns = turns;

	this->repeat = e;

}

void DynamicSearch::createRing() {

	  // very simple ring model: a FODO cell repeated many times;
	  // but it has the typical number of elements.
	  this->ring.clear();

          /* See performOneTurn(). Instead of adding many equal elements, we add only one
             instance for each one and iterate many times over this smaller set.  */
          this->ring.push_back(new Drift("Drift1", 1.0));
	  this->ring.push_back(new Quadrupole("Quadrupole1", 0.1, +2.0));
	  this->ring.push_back(new Drift("Drift2", 1.0));
	  this->ring.push_back(new Quadrupole("Quadrupole2", 0.1, -2.0));
	  this->ring.push_back(new Sextupole("Sextupole", 0.1, -30.0));

}

DynamicSearch::~DynamicSearch() {

	this->ring.clear();
}

bool DynamicSearch::testSolution(pos_t r){

	double s=r[0]+r[1]+r[2]+r[3]+r[4]+r[5];
	if ((std::isfinite(s)) && // is not a NaN or Inf
		((abs(r[0]) < this->positionThreshold) &&
		(abs(r[1]) < this->angularThreshold)  &&
		(abs(r[2]) < this->positionThreshold)  &&
		(abs(r[3]) < this->angularThreshold)  &&
		(abs(r[4]) < this->deviationEnergyThreshold)   &&
		(abs(r[5]) < this->positionThreshold)))
		return true;

	return false;

}

void DynamicSearch::performOneTurn(pos_t &e) {

        /* In order to make the cuda solution easier and to use less memory,
           we add fewer elements into the ring vector, but execute more iterations */
        for(unsigned int i = 0; i < this->repeat; ++i)
	        for(unsigned int j = 0; j < this->ring.size(); ++j)
	        	this->ring[j]->pass(e);

}

/* DefaultDynamicSearch class members*/

int DefaultDynamicSearch::dynamical_aperture_search() {

    double x[2] = {-0.10, +0.10};
    double y[2] = {+0.000, +0.003};

    unsigned int nr_stable_points = 0;

    for(unsigned int i = 0; i < N_POINTS_X; ++i) {
      for(unsigned int j = 0; j < N_POINTS_Y; ++j) {

        double posx = x[0] + i*(x[1] - x[0])/(N_POINTS_X - 1);
        double posy = y[0] + j*(y[1] - y[0])/(N_POINTS_Y - 1);

        pos_t r = {posx, 0, posy, 0, 0, 0};

        for(unsigned int i = 0; i < this->turns; ++i) {
        	this->performOneTurn(r);
        }

        if (this->testSolution(r)) {
                nr_stable_points++;
                std::cout << "nr_stable_points: " << nr_stable_points << "/" << N_POINTS_X * N_POINTS_Y << "(" << i << "," << j << ")" << std::endl;
            }
      }
    }

    return nr_stable_points;

}
