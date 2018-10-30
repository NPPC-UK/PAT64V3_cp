#ifndef ANAL_UTILS
#define ANAL_UTILS
#include <opencv2/opencv.hpp>

/**
 * Defines Morphological opening and closing.  Use this instead of magic
 * number or bool.
 */
enum class MorphOp {
  Open = 0,
  Close = 1
};

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

/*
 * Finds the points delimiting a pot on the lemnatec system.
 *
 * I conjecture that the following is true
 *
 * Point 0 corresponds to the upper edge of the pot, in the centre
 * of that edge.
 *
 * Point 1 corresponds to the left most part of the pot
 * Point 2 corresponds to the right most part of the pot
 */
std::array<cv::Point, 3> FindLTPotLimits(const cv::Mat img);

/**
 * Perform Morphological opening or closing (depending on 'op') on the 
 * target img.
 *
 * The structuring element is square with dimensions 2*esize+1 for erosion
 * and 2*dsize+1 for dilation.  The erosion operation is performed etimes 
 * many times, the dilation operation is performed dtimes many times.
 */
cv::Mat OnMorphology(const cv::Mat img, 
                     int etimes, 
                     int dtimes, 
                     int esize, 
                     int dsize, 
                     MorphOp op);

/**
 * Invoke OnMorphology with identical arguments, except use MorphOp enum
 * instead of flag. 
 *
 * Allows compilation of code that is not aware of MorphOp Enum
 */
cv::Mat OnMorphology(const cv::Mat img, 
                     int etimes, 
                     int dtimes, 
                     int esize, 
                     int dsize, 
                     int flag);

#endif
