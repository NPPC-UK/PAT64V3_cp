#ifndef ANALYSIS
#define ANALYSIS

#include <memory>

#include "plant_data.h"

std::unique_ptr<plant_data> GetData(const char* filename);

#endif
