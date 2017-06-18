/*
 * DynamicSearch.cpp - The implemention of DynamicSearch and DefaultDynamicSearch classes.
 *
 * Gustavo Ciotto Pinton - RA117136
 * Parallel Programming - June/2017
 */

#include "DynamicSearch.h"
#include <cmath>
#include <iostream>
#include <fstream>

/* DynamicSearch class */

/* Default constructor */
DynamicSearch::DynamicSearch() {

	this->positionThreshold = POSITION_THRESHOLD;
	this->angularThreshold = ANGULAR_THRESHOLD;
	this->deviationEnergyThreshold = DEVIATION_ENERGY_THRESHOLD;

	this->repeat = DEFAULT_ELEMENT_COUNT;

	this->turns = DEFAULT_TURNS;

	this->ring.clear();

        this->result_set = (pos_t*) malloc (N_POINTS_X * N_POINTS_Y * sizeof(pos_t));
}

DynamicSearch::DynamicSearch(unsigned int e, double pThreshold, double angThreshold, double deviationEnergyThreshold, unsigned int turns )
		: DynamicSearch() {

	this->positionThreshold = pThreshold;
	this->angularThreshold = angThreshold;
	this->deviationEnergyThreshold = deviationEnergyThreshold;

	this->turns = turns;

	this->repeat = e;
}

void DynamicSearch::createRing() {

	  /* very simple ring model: a FODO cell repeated many times;
	     but it has the typical number of elements. */
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
        free(this->result_set);
}

/* Tests a solution r according to the defined constants */
bool DynamicSearch::testSolution(pos_t r){

	double s = r[0] + r[1] + r[2] + r[3] + r[4] + r[5];
	if ((std::isfinite(s)) && // is not a NaN or Inf
		((abs(r[0]) < this->positionThreshold) &&
		(abs(r[1]) < this->angularThreshold) &&
		(abs(r[2]) < this->positionThreshold) &&
		(abs(r[3]) < this->angularThreshold) &&
		(abs(r[4]) < this->deviationEnergyThreshold) &&
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

/* Computes the magnitude of all arrays of result set and saves it into a file */
void DynamicSearch::plot() {

        if (this->result_set != NULL) {

                std::ofstream out_file;
                out_file.open ("plot_dynamicsearch.dat");
                
                for (unsigned int i = 0; i < N_POINTS_X; i++) {
                        for (unsigned int j = 0; j < N_POINTS_Y; j++) {

                                int index = i * N_POINTS_X + j;
                                double *r = this->result_set[index];

                                double p, sum_r = r[0] + r[1] + r[2] + r[3] + r[4] + r[5];

                                if (!std::isfinite(sum_r)) p = INFINITY_MAGNITUDE;
                                else {
                                        p = 0.0;
                                        for (unsigned int k = 0; k < 6; k++)
                                                p += r[k] * r[k];

                                        p = sqrt(p);
                                }
                                out_file << p << " ";
                        }
                        out_file << std::endl;
                }
                out_file.close();
        }

}

/* DefaultDynamicSearch class members*/

/* Computes the dynamics of N_POINTS_X * N_POINTS_Y electrons in a single core */
int DefaultDynamicSearch::dynamical_aperture_search() {

    double x[2] = {-0.10, +0.10};
    double y[2] = {+0.000, +0.003};

    unsigned int nr_stable_points = 0;

    /* Iterates over N_POINTS_X * N_POINTS_Y initial states */
    for(unsigned int i = 0; i < N_POINTS_X; ++i) {
      for(unsigned int j = 0; j < N_POINTS_Y; ++j) {

        double posx = x[0] + i*(x[1] - x[0])/(N_POINTS_X - 1);
        double posy = y[0] + j*(y[1] - y[0])/(N_POINTS_Y - 1);

        int index = i * N_POINTS_X + j;

        this->result_set[index][0] = posx;
        this->result_set[index][2] = posy;
        this->result_set[index][1] = this->result_set[index][3] = this->result_set[index][4] = this->result_set[index][5] = 0;

        for(unsigned int k = 0; k < this->turns; ++k) {
        	this->performOneTurn(this->result_set[index]);
        }

        if (this->testSolution(this->result_set[index]))
                printf ("%f %f %f %f %f %f (%d / %d) - (%d)\n", 
				this->result_set[index][0], this->result_set[index][1], 
                                this->result_set[index][2], this->result_set[index][3], 
				this->result_set[index][4], this->result_set[index][5], 
                                ++nr_stable_points , N_POINTS_X * N_POINTS_Y, index);
      }
    }

    return nr_stable_points;
}
