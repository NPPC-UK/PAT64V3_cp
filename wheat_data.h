#ifndef PLANT_DATA_WHEAT
#define PLNAT_DATA_WHEAT
#include <string>

#include "plant_data.h"

class wheat_data : public plant_data {
  public:
  int plant_height;
  int pot_width;
  double p_h;
  double p_h_t;
  double pixelcount;
  double leafArea;
  int yellowcount;
  int t20;
  int t20y;
  int t40;
  int t40y;
  int t60;
  int t60y;

  std::string to_string(std::string const separator) const;

  wheat_data();
};

#endif
