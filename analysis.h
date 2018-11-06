#ifndef ANALYSIS
#define ANALYSIS

#include <memory>

#include "plant_data.h"

std::unique_ptr<plant_data> GetData(const char* filename);

/*
 * Return an image where all plant pixels have been set to 255 on 
 * all channels.  The pixels that are counted as plant pixels is 
 * implementation specific (analysis_{bb,wb}.cpp) and depends on 
 * the values of gthres and gbthres
 */
cv::Mat FindPlantPixels(cv::Mat img, double gthres, double gbthres);

cv::Point OnPlantTop(cv::Mat img, cv::Rect rect);
#endif
