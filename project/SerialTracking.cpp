/*
 * SerialTracking.cpp
 *
 *  Created on: May 14, 2017
 *      Author: gciotto
 */

#include "DynamicSearch.h"

int main(int argc, char **argv) {

	DefaultDynamicSearch d;
	d.createRing();

	d.dynamical_aperture_search();


	return 0;
}





