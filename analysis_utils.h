#ifndef ANAL_UTILS
#define ANAL_UTILS
#include <opencv2/opencv.hpp>


int OnCountPixels(cv::Mat img, int pottop, int planttop);
cv::Mat RestoreImgFromTemp(cv::Mat temp, cv::Mat source);
#endif
