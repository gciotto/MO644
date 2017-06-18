/*
 * RingElement.h - Provides the definition of a base class describing a simple ring actuator. All classes 
 * must inherit from it in order to provide the implementation of a customised actuator.
 *
 * Gustavo Ciotto Pinton - RA117136
 * Parallel Programming - June/2017
 */

#ifndef RINGELEMENT_H_
#define RINGELEMENT_H_

#include <string>

typedef double pos_t[6];

/* A single element of the ring */
class RingElement {
public:
	/* Static constants */
	static const int DRIFT = 0, QUADRUPOLE = 1, SEXTUPOLE = 2;

	RingElement(std::string name, double length, int pass_method);
	virtual ~RingElement() {};

        int getType () {
                return this->pass_method;
        }

        double getLength () {
                return this->length;
        }

        /* Abstract members that all classes inheriting from this  */
	virtual void pass(pos_t &e) = 0;

protected:
	std::string         fam_name;
	double              length;
	int                 pass_method;
};

class Drift : public RingElement {

public:

	Drift (std::string name, double length)
			: RingElement(name, length, RingElement::DRIFT) {}

	void pass(pos_t &e);
};


class Quadrupole : public RingElement {

public:

	Quadrupole (std::string name, double length, double focal_distance)
			: RingElement(name, length, RingElement::QUADRUPOLE) {
		this->focal_distance = focal_distance;
	}

        double getFocalDistance() {
                return this->focal_distance;
        }

	void pass(pos_t &e);

private:
	double focal_distance;
};

class Sextupole : public RingElement {

public:

	Sextupole (std::string name, double length, double sextupole_strength)
			: RingElement(name, length, RingElement::SEXTUPOLE) {
		this->sextupole_strength = sextupole_strength;
	}

        double getSextupoleStrength() {
                return this->sextupole_strength;
        }

	void pass(pos_t &e);

private:
	double sextupole_strength;
};

#endif /* RINGELEMENT_H_ */
