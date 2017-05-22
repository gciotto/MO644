/*
 * SerialTracking.cpp
 *
 *  Created on: May 14, 2017
 *      Author: gciotto
 */

#include "DynamicSearch.h"
#include <iostream>
#include <ctime>

int main(int argc, char **argv) {

        unsigned int repeat, turns;

        if (argc != 3) {
                std::cout << "Not enough parameters" << std::endl;            
                return 0;
        }

        repeat = atoi(argv[1]);
        turns = atoi(argv[2]);

	DefaultDynamicSearch d(repeat, turns);
	d.createRing();

        clock_t begin_time = std::clock(), search_time, plot_time;

	d.dynamical_aperture_search();

        search_time = std::clock();

        d.plot();

        plot_time = std::clock();

        std::cout << "Search elapsed time : " << double(search_time - begin_time) / CLOCKS_PER_SEC << std::endl;
        std::cout << "Plot elapsed time : " << double(plot_time - search_time) / CLOCKS_PER_SEC << std::endl;
        std::cout << "Total elapsed time : " << double(plot_time - begin_time) / CLOCKS_PER_SEC << std::endl;

	return 0;
}





