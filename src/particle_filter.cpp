#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <float.h>
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  // TODO Complete

  num_particles = 25;  /* number of particles which is a hyper-parameter */
  default_random_engine gen; /* random_generator  */

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // =========================================================
  //  Creating particles and their state normally distributed
  // ======================================================
  for (unsigned int i = 0; i < num_particles; i++)
  {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    weights.push_back(particle.weight);
    particles.push_back(particle);
  }

  is_initialized = true;  /* setting is_initialized to True */
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
  default_random_engine gen;
  double predicted_x, predicted_y, predicted_theta;  /*declaring variables to hold predicted states of particle*/


  // =========================================
  // Adding measurement to each particle
  // =========================================
  for (int i = 0; i < num_particles; i++)
  {
    if (fabs(yaw_rate) >= 0.0001)
    {
      //===============================
      // If yaw_rate is greater then 0
      //===============================
      predicted_x = particles[i].x + (velocity/yaw_rate) *
                                     (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      predicted_y = particles[i].y + (velocity/yaw_rate) *
                                     (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
      predicted_theta = particles[i].theta + (yaw_rate * delta_t);
    }
    else
    {
      //=====================
      // If yaw_rate is zero
      //=====================
      predicted_x = particles[i].x  + velocity * cos(particles[i].theta) * delta_t;
      predicted_y = particles[i].y + velocity * sin(particles[i].theta) * delta_t;
      predicted_theta = particles[i].theta;
    }

    normal_distribution<double> dist_x(predicted_x, std_pos[0]);
    normal_distribution<double> dist_y(predicted_y, std_pos[1]);
    normal_distribution<double> dist_theta(predicted_theta, std_pos[2]);

    // =============================
    // Adding Random Gaussian Noise
    // =============================
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{

  double minimum_distance;
  double current_distance;

  /* Associate observations in map co-ordinates to predicted landmarks using nearest neighbor algorithm.*/
  for (int i = 0; i < observations.size(); i++)
  {
    minimum_distance = DBL_MAX;

    for (int j = 0; j < predicted.size(); j++)
    {
      current_distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (current_distance < minimum_distance)
      {
        minimum_distance = current_distance;
        observations[i].id = predicted[j].id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
//   according to the MAP'S coordinate system. You will need to transform between the two systems.
//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
//   The following is a good resource for the theory:
//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
//   and the following is a good resource for the actual equation to implement (look at equation 3.33)
//   http://planning.cs.uiuc.edu/node99.html


  double weight_normalizer = 0.0; // used for normalizing weights of all particles and bring them in the range of [0, 1]
  double sigma_x, sigma_x_2;
  double sigma_y, sigma_y_2;
  double normalizer;

  for (int i = 0; i < num_particles; i++)
  {
    // =============================================================================
    // Step 1: Transform observations from vehicle co-ordinates to map co-ordinates.
    // =============================================================================

    //Vector containing observations transformed to map co-ordinates w.r.t. current particle.
    vector<LandmarkObs> transformed_observations;
    transform_to_map_coordinates(transformed_observations, observations, i);


    // ===================================================================
    // Step 2: Filter map landmarks to keep only those which are in the
    // sensor_range of current particle. Push them to predictions vector
    // ===================================================================
    vector<LandmarkObs> predicted_landmarks;
    get_landmarks_with_in_sensor_range(predicted_landmarks,
                                       map_landmarks, sensor_range,
                                       i);

    // ===================================================================
    // Step 3: Associate observations to predicted landmarks using nearest
    // neighbor algorithm. Associate observations with predicted landmarks
    // ===================================================================
    dataAssociation(predicted_landmarks, transformed_observations);

    // ===================================================================
    // Step 4: Calculate the weight of each particle using Multivariate
    // Gaussian distribution.
    // ===================================================================
    particles[i].weight = 1.0;        // Reset the weight of particle to 1.0

    sigma_x = std_landmark[0];
    sigma_y = std_landmark[1];
    sigma_x_2 = pow(sigma_x, 2);
    sigma_y_2 = pow(sigma_y, 2);
    normalizer = (1.0/(2.0 * M_PI * sigma_x * sigma_y));

    // ===================================================================
    // Calculate the weight of particle based on the multivariate
    // Gaussian probability function
    // ===================================================================
    for (int j = 0; j < transformed_observations.size(); j++)
    {
      double multi_prob;

      for (int k = 0; k < predicted_landmarks.size(); k++)
      {

        if (transformed_observations[j].id == predicted_landmarks[k].id)
        {
          multi_prob = normalizer *
              exp(-1.0 * ((pow((transformed_observations[j].x - predicted_landmarks[k].x), 2)/(2.0 * sigma_x_2))
                          + (pow((transformed_observations[j].y - predicted_landmarks[k].y), 2)/(2.0 * sigma_y_2))));

          if (multi_prob == 0)
          {
            multi_prob = 0.01;
          }
          particles[i].weight *= multi_prob;
        }
      }
    }
    weight_normalizer += particles[i].weight;
  }

  // ===================================================================
  // Step 5: Normalize the weights of all particles since re-sampling
  // using probabilistic approach.
  // ===================================================================
  for (int i = 0; i < particles.size(); i++)
  {
    particles[i].weight /= weight_normalizer;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {

  vector<Particle> re_sampled_particles;

  // ===================================================================
  // Create a generator to be used for generating random
  // particle index and beta value
  // ===================================================================
  default_random_engine gen;

  // ================================
  //Generate random particle index
  // ================================
  uniform_int_distribution<int> particle_index(0, num_particles - 1);

  int current_index = particle_index(gen);

  double beta = 0.0;

  double max_weight_2 = 2.0 * *max_element(weights.begin(), weights.end());

  for (int i = 0; i < particles.size(); i++) {
    uniform_real_distribution<double> random_weight(0.0, max_weight_2);
    beta += random_weight(gen);

    while (beta > weights[current_index]) {
      beta -= weights[current_index];
      current_index = (current_index + 1) % num_particles;
    }
    re_sampled_particles.push_back(particles[current_index]);
  }
  particles = re_sampled_particles;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

void ParticleFilter::transform_to_map_coordinates(std::vector<LandmarkObs>& transformed_observations,
                                                  const std::vector<LandmarkObs>& observations,
                                                  int idx)
{
  for (int j = 0; j < observations.size(); j++)
  {
    LandmarkObs transformed_obs;
    transformed_obs.id = j;

    transformed_obs.x = particles[idx].x +
        (cos(particles[idx].theta) * observations[j].x) - (sin(particles[idx].theta) * observations[j].y);

    transformed_obs.y = particles[idx].y +
        (sin(particles[idx].theta) * observations[j].x) + (cos(particles[idx].theta) * observations[j].y);
    transformed_observations.push_back(transformed_obs);
  }
}

void ParticleFilter::get_landmarks_with_in_sensor_range(std::vector<LandmarkObs> &predicted_landmarks,
                                                        Map map_landmarks, double sensor_range,
                                                        int idx)
{
  for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
  {
    Map::single_landmark_s current_landmark = map_landmarks.landmark_list[j];
    if ((fabs((particles[idx].x - current_landmark.x_f)) <= sensor_range) &&
        (fabs((particles[idx].y - current_landmark.y_f)) <= sensor_range))
    {
      predicted_landmarks.push_back(LandmarkObs {current_landmark.id_i, current_landmark.x_f, current_landmark.y_f});
    }
  }
}