#ifndef PLANT_DATA
#define PLANT_DATA

#include <string>

#include <opencv2/opencv.hpp>

class plant_data {
  public:
  cv::Mat image;

  virtual std::string to_string(std::string const separator) const;
  std::string to_string(char* const separator) const;
  std::string to_string(char const separator) const;
}; 

#endif
