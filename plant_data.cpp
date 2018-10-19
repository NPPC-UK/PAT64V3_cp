#include "plant_data.h"

#include <string>
#include <sstream>

std::string plant_data::to_string(char const separator) const {
  return this->to_string("" + separator);
}

std::string plant_data::to_string(char* const separator) const {
  return this->to_string(std::string(separator));
}

std::string plant_data::to_string(std::string const separator) const {
  return std::string("Not implemented here.");
}
