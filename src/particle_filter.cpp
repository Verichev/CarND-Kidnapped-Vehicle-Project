/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::cout;
using std::normal_distribution;



void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  weights.clear();
  particles.clear();
  num_particles = 100;  // Set the number of particles
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for (int i = 0; i < num_particles; i++) {
    double x_part = dist_x(gen);
    double y_part = dist_y(gen);
    double theta_part = dist_theta(gen);
    particles.push_back(Particle {i, x_part, y_part, theta_part, 1.0});
    weights.push_back(1);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {

  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle& p = particles[i];
    if (fabs(yaw_rate) > 0.0001) {
      p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(yaw_rate * delta_t + p.theta));
      p.theta += yaw_rate * delta_t;
    } else {
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    }
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);
  }

}

Map::single_landmark_s ParticleFilter::dataAssociation(double x, double y, const vector<Map::single_landmark_s> &landmarks) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
  Map::single_landmark_s landmark = landmarks[0];
  double min = dist(x, y, landmarks[0].x_f, landmarks[0].y_f);
  for (int i = 0; i < landmarks.size(); i++) {
    double d = dist(x, y, landmarks[i].x_f, landmarks[i].y_f);
    if (d < min) {
      landmark = landmarks[i];
      min = d;
    }
  }
  return landmark;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {

  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  long double x_map, y_map;
  for (int i = 0; i < num_particles; i++) {
    Particle& p = particles[i];
    long double weight = 1.0;
    std::vector<Map::single_landmark_s> filtered_landmarks;
    for (Map::single_landmark_s landmark: map_landmarks.landmark_list) {
      if ((fabs(p.x - landmark.x_f) <= sensor_range) && (fabs(p.y - landmark.y_f) <= sensor_range))
        filtered_landmarks.push_back(landmark);
    }
    for (LandmarkObs obs: observations) {
      x_map = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
      y_map = p.y + (sin(p.theta) * obs.x) + (cos(p.theta) * obs.y);
      Map::single_landmark_s landmark = dataAssociation(x_map, y_map, filtered_landmarks);
      long double p_weight;
      p_weight = multiv_prob(std_landmark[0], std_landmark[1], x_map, y_map, landmark.x_f, landmark.y_f);
      weight *= p_weight;
    }
    weights[i] = weight;
    p.weight = weight;
  }
}

long double ParticleFilter::multiv_prob(long double sig_x, long double sig_y, long double x_obs, long double y_obs,
                   long double mu_x, long double mu_y) {
  long double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  long double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
  + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

  long double weight;
  weight = gauss_norm * exp(-exponent);
  return weight;
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::vector<Particle> new_particles;
  std::discrete_distribution<> d(weights.begin(), weights.end());
  for (int i = 0; i < num_particles; i++) {
    new_particles.push_back(particles[d(gen)]);
  }
  particles = new_particles;
  cout << std::endl << "after resemple: " << std::endl;
  for (auto const& c : particles) {
    std::cout << c << std::endl;
  }
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
