#ifndef CLOVER_DATA_WHEAT
#define CLOVER_DATA_WHEAT
#include <string>

#include "plant_data.h"

class clover_data : public plant_data {
  public:
  double pixelcount[6], yellowcount[6], length[6];

  std::string to_string(std::string const separator) const;

  clover_data();
};

#endif
