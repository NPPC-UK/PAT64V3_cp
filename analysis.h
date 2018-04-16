#ifndef ANALYSIS
#define ANALYSIS

#include <string>

#include <opencv2/opencv.hpp>

class plant_data {
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
  cv::Mat image;

  std::string to_string(std::string const separator) const;
  std::string to_string(char* const separator) const;
  std::string to_string(char const separator) const;

  plant_data();
}; 

plant_data GetData(const char* filename);

#endif
