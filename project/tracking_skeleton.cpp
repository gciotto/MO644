#include <iostream>
#include <string>
#include <vector>
#include <cmath>

// compile with: g++ -std=c++11

// --- basic types ---

const int DRIFT = 0;
const int QUADRUPOLE = 1;
const int SEXTUPOLE = 2;

typedef double pos_t[6];

const double pos_threshold = 1.0;  // [1.0 meter]
const double ang_threshold = 0.01; // [10 mrad]
const double de_threshold  = 0.01; // [1% energy deviation]

class RingElement {
public:
  std::string         fam_name;
  double              length;
  int                 pass_method;
  double              focal_distance;
  double              sextupole_strength;
  double              other_parameters[200];
};

// --- pass methods and tracking functions ---

int drift_pass(pos_t& r, const RingElement& e) {
  r[0] += e.length * r[1];
  r[2] += e.length * r[3];
}

int quadrupole_pass(pos_t& r, const RingElement& e) {
  r[1] += -r[0]/e.focal_distance;
  r[3] += +r[2]/e.focal_distance;
}

int sextupole_pass(pos_t& r, const RingElement& e) {
    r[1] += e.sextupole_strength * e.length * (r[0]*r[0] - r[2]*r[2]);
    r[3] += e.sextupole_strength * e.length * 2 * r[0]*r[2];
}

int (*pass_methods[])(pos_t&, const RingElement&) = { drift_pass, quadrupole_pass };



// map that takes particle pos at entrance of the ring and propagates it to the end of the ring.
int one_turn_map(pos_t& r, const std::vector<RingElement>& ring) {

    for(auto i=0; i<ring.size(); ++i) {
      if (ring[i].pass_method == DRIFT) {
        drift_pass(r, ring[i]);
      } else if (ring[i].pass_method == QUADRUPOLE) {
        quadrupole_pass(r, ring[i]);
      } else if (ring[i].pass_method == SEXTUPOLE) {
        sextupole_pass(r, ring[i]);
      } else {
        std::cerr << "undefined pass method!" << std::endl;
      }
    }

}

// --- lattice aux. functions ---

void create_ring(std::vector<RingElement>& ring) {
  // very simple ring model: a FODO cell repeated many times;
  // but it has the typical number of elements.
  ring.clear();
  for(auto i=0; i<2000; ++i) {
    { RingElement e; e.fam_name="Drift1";      e.length=1.0; e.pass_method=DRIFT; ring.push_back(e); }
    { RingElement e; e.fam_name="Quadrupole1"; e.length=0.1; e.pass_method=QUADRUPOLE; e.focal_distance = +2.0; ring.push_back(e); }
    { RingElement e; e.fam_name="Drift2";      e.length=1.0; e.pass_method=DRIFT; ring.push_back(e); }
    { RingElement e; e.fam_name="Quadrupole2"; e.length=0.1; e.pass_method=QUADRUPOLE; e.focal_distance = -2.0; ring.push_back(e); }
    { RingElement e; e.fam_name="Sextupole";   e.length=0.1; e.pass_method=SEXTUPOLE; e.sextupole_strength = -30.0; ring.push_back(e); }
  }

}


int dynamical_aperture_search(const std::vector<RingElement>& ring) {

    int nrpts_x = 100;
    int nrpts_y = 100;
    double x[2] = {-0.10, +0.10};
    double y[2] = {+0.000, +0.003};

    unsigned int nr_turns = 10;
    unsigned int nr_stable_points = 0;
    
    for(auto i=0; i<nrpts_x; ++i) {
      for(auto j=0; j<nrpts_y; ++j) {

        double posx = x[0] + i*(x[1]-x[0])/(nrpts_x-1);
        double posy = y[0] + j*(y[1]-y[0])/(nrpts_y-1);      

        pos_t r = {posx,0,posy,0,0,0};

        for(auto i=0; i<nr_turns; ++i) {
            one_turn_map(r, ring);
        }
        
        double s=r[0]+r[1]+r[2]+r[3]+r[4]+r[5];
        if ((std::isfinite(s)) && // is not a NaN or Inf
            ((abs(r[0]) < pos_threshold) && 
            (abs(r[1]) < ang_threshold)  && 
            (abs(r[2]) < pos_threshold)  && 
            (abs(r[3]) < ang_threshold)  && 
            (abs(r[4]) < de_threshold)   && 
            (abs(r[5]) < pos_threshold))) {
                nr_stable_points++;
                std::cout << "nr_stable_points: " << nr_stable_points << "/" << nrpts_x*nrpts_y << std::endl;
            }
      }
    }
}

int main() {

  std::vector<RingElement> ring;
  create_ring(ring);

  // Calculates dynamical aperture for a specific accelerator configuration. 
  dynamical_aperture_search(ring);

  return EXIT_SUCCESS;

}
