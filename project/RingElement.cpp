/*
 * RingElement.cpp
 *
 *  Created on: May 14, 2017
 *      Author: gciotto
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

	std::cout << "Passing through " << this->fam_name << std::endl;

	e[0] += this->length * e[1];
	e[2] += this->length * e[3];
}

void Quadrupole::pass(pos_t &e) {

	std::cout << "Passing through " << this->fam_name << std::endl;

	e[1] += -e[0]/this->focal_distance;
	e[3] += +e[2]/this->focal_distance;
}

void Sextupole::pass(pos_t &e) {

	std::cout << "Passing through " << this->fam_name << std::endl;

	e[1] += this->sextupole_strength * this->length * (e[0]*e[0] - e[2]*e[2]);
	e[3] += this->sextupole_strength * this->length * 2 * e[0]*e[2];
}
