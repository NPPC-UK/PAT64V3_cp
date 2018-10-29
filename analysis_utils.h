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
int OnCountPixels(const cv::Mat img, unsigned int pottop, unsigned int planttop);

/**
 * Apply the mask to the source image.
 *
 * The mask is a grayscale (single channel) image.  For any 
 * pixel with value 0 in the mask, set the corresponding pixel
 * in the output to (255,255,255). For any pixel with non 0 
 * value in the mask, copy the value of the source pixel to the
 * output pixel.
 *
 * Assume that the source is a (B,G,R) image.
 *
 */
cv::Mat RestoreImgFromTemp(const cv::Mat mask, const cv::Mat source);
#endif
