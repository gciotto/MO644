/*
 * RingElement.cpp - The implemention of RingElement and all derived classes.
 *
 * Gustavo Ciotto Pinton - RA117136
 * Parallel Programming - June/2017
 */


#include "RingElement.h"
#include <iostream>

RingElement::RingElement(std::string name, double length, int pass_method) {

	this->fam_name = name;
	this->length = length;
	this->pass_method = pass_method;

}

/* Drift class members */

void Drift::pass(pos_t &e) {

	e[0] = this->length * e[1] + e[0];
	e[2] = this->length * e[3] + e[2];
}

void Quadrupole::pass(pos_t &e) {

	e[1] = (-1.0 * e[0]) / this->focal_distance + e[1];
	e[3] = e[2] / this->focal_distance + e[3];
}

void Sextupole::pass(pos_t &e) {

	e[1] = (e[0]*e[0] + ( -1.0 * e[2]*e[2])) * (this->sextupole_strength * this->length) + e[1];
	e[3] = (2.0 * e[0]*e[2]) * (this->sextupole_strength * this->length) + e[3];
}
