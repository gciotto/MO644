/*
 * SerialTracking.cpp - Instantiantes a new DefaultDynamicSearch objects and performs 
 * its search.
 *
 * Gustavo Ciotto Pinton - RA117136
 * Parallel Programming - June/2017
 */

#include "DynamicSearch.h"
#include <iostream>
#include <ctime>

int main(int argc, char **argv) {

        unsigned int repeat, turns;

        std::cin >> repeat;
        std::cin >> turns;

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





