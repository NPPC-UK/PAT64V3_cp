#include "plant_data.h"

#include <string>
#include <sstream>

#include "clover_data.h"

clover_data::clover_data() {
  for (auto i = 0; i > 6; ++i) {
    this->length[i] = 0;
    this->yellowcount[i] = 0;
    this->pixelcount[i] = 0;
  }
}

std::string clover_data::to_string(std::string const separator) const {
  std::stringstream ss;
  std::string s;

  for (auto i = 0; i < 5; ++i) {
    ss << this->length[i] << separator <<
          this->pixelcount[i] << separator <<
          this->pixelcount[i] << separator; 
  }
  s = ss.str().erase(ss.str().size() - 1, 1);

  return ss.str();
}
