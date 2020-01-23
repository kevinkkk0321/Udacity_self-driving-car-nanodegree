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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   *   Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   *   Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
    num_particles = 200;
    
    std::default_random_engine gen;
    std::normal_distribution<> dist_x(x, std[0]);
    std::normal_distribution<> dist_y(y, std[1]);
    std::normal_distribution<> dist_theta(theta, std[2]);
    
    
    for (int i=0; i<num_particles; i++) {
        Particle particle;
        particle.id = i;
        particle.weight = 1.0;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particles.push_back(particle);
        weights.push_back(particle.weight);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
    std::default_random_engine gen;
     
    // Add measurements to each particle and add random Gaussian noise.
    for (unsigned int i = 0; i < num_particles; ++i)
    {
        double theta = particles[i].theta;
        
        if(abs(yaw_rate) < 0.0001){
            particles[i].x += velocity*cos(theta)*delta_t;
            particles[i].y += velocity*sin(theta)*delta_t;
        }
        
        else {
            particles[i].theta += yaw_rate*delta_t;
            particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta) - sin(theta));
            particles[i].y += (velocity/yaw_rate)*(cos(theta) - cos(particles[i].theta));
        }
        
        // Create normal distributions for x, y and theta around the new mean predictions
        std::normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
        std::normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
        std::normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
        
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   *   Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
    for (unsigned int i = 0; i < observations.size(); ++i) {
        
        // Current observation
        LandmarkObs obs_curr = observations[i];

        // Initialize minimum distance to max value
        double dist_min = 100000;

        // Initialize map_id
        int map_id = 0;
        
        for (unsigned int j = 0; j < predicted.size(); ++j) {
          // Current prediction
          LandmarkObs pred_curr = predicted[j];
          
          // Calculate distance between current ith observation and jth landmark
          double dist_curr = dist(obs_curr.x, obs_curr.y, pred_curr.x, pred_curr.y);

          // Store id of the nearest landmark
          if (dist_curr < dist_min) {
            dist_min = dist_curr;
            map_id = pred_curr.id;
          }
        }
        // Link observation id to that of nearest landmark
        observations[i].id = map_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   *   Update the weights of each particle using a mult-variate Gaussian
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
    
    for (int i=0; i<num_particles; i++) {
        double x_particle = particles[i].x;
        double y_particle = particles[i].y;
        double theta_particle = particles[i].theta;
        particles[i].weight = 1.0;
        
        std::vector<LandmarkObs> observations_map;
        LandmarkObs l_obs;
        
        for (int j=0; j<observations.size(); j++) {
            l_obs.x = x_particle + cos(theta_particle)*observations[j].x - sin(theta_particle)*observations[j].y;
            l_obs.y = y_particle + sin(theta_particle)*observations[j].x + cos(theta_particle)*observations[j].y;
            l_obs.id = observations[j].id;
            observations_map.push_back(l_obs);
        }
        
        // Placeholder for landmarks in sensor range
        vector<LandmarkObs> predictions;

        for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
          // Store landmark x, y, id in LandMarkObs struct
          l_obs.x = map_landmarks.landmark_list[j].x_f;
          l_obs.y = map_landmarks.landmark_list[j].y_f;
          l_obs.id = map_landmarks.landmark_list[j].id_i;
          
          // Append landmarks within sensor range from ith particle to predictions vector
          if (dist(l_obs.x, l_obs.y, x_particle, y_particle) <= sensor_range) {
            predictions.push_back(l_obs);
          }
        }
        dataAssociation(predictions, observations_map);
        // Update weights based on particle observations and actual observations
        for (int j = 0; j < observations_map.size(); ++j) {
                  
            // Nearest landmark id
            int lm_id = observations_map[j].id;

            double lm_x, lm_y;

            // Nearest landmark coordinates
            for (int k = 0; k < predictions.size(); ++k) {
                if (predictions[k].id == lm_id) {
                    lm_x = predictions[k].x;
                    lm_y = predictions[k].y;
                }
            }
            
            // Observed coordinates
            double o_x = observations_map[j].x;
            double o_y = observations_map[j].y;

            // Calculate importance weight with multivariate Gaussian
            double sig_x = std_landmark[0];
            double sig_y = std_landmark[1];
            double w_ = ( 1/(2*M_PI*sig_x*sig_y)) * exp( -( pow(lm_x-o_x,2)/(2*pow(sig_x, 2)) + (pow(lm_y-o_y,2)/(2*pow(sig_y, 2))) ) );

            // Update particle weight for this observation
            particles[i].weight *= w_;
        }
        // Updates weights vector
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
  /**
   *   Resample particles with replacement with probability proportional
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
    std::default_random_engine gen;
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    
    // Placeholder for storing resampled particles
    vector<Particle> resampled_particles;
    
    // Resample particles according to their weights
    for (int i = 0; i < num_particles; ++i)
    {
        resampled_particles.push_back(particles[dist(gen)]);
    }

    // Update particles with resampled particles
    particles = resampled_particles;

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
