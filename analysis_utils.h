#ifndef ANAL_UTILS
#define ANAL_UTILS
#include <opencv2/opencv.hpp>

/**
 * Count the number of pixels with intensity <=210.
 *
 * Convert the input image to grayscale first if
 * necessary.  Count only pixels in the central 70% of columns 
 * and between rows pottop and planttop.
 */
int OnCountPixels(cv::Mat img, int pottop, int planttop);
cv::Mat RestoreImgFromTemp(cv::Mat temp, cv::Mat source);
#endif
