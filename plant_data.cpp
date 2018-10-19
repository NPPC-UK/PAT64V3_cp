#include "plant_data.h"

#include <string>
#include <sstream>

plant_data::plant_data() {
  this->plant_height = 0;
  this->pot_width = 0;
  this->p_h = 0.0;
  this->p_h_t = 0.0;
  this->pixelcount = 0.0;
  this->leafArea = 0.0;
  this->yellowcount = 0;
  this->t20 = 0;
  this->t20y = 0;
  this->t40 = 0;
  this->t40 = 0;
  this->t60y = 0;
  this->t60y = 0;
}

std::string plant_data::to_string(char const separator) const {
  return this->to_string("" + separator);
}

std::string plant_data::to_string(char* const separator) const {
  return this->to_string(std::string(separator));
}

std::string plant_data::to_string(std::string const separator) const {
  std::stringstream ss;

  ss << this->plant_height << separator <<
        this->pot_width << separator <<
        this->p_h << separator <<
        this->p_h_t << separator <<
        this->pixelcount << separator <<
        this->leafArea << separator <<
        this->yellowcount << separator <<
        this->t20 << separator <<
        this->t20y << separator <<
        this->t40 << separator <<
        this->t40y << separator <<
        this->t60 << separator <<
        this->t60y;

  return ss.str();
}
