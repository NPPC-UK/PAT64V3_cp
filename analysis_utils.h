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
[[deprecated("Use 'cv::Mat OnMorphology(cv::Mat, int, int, int, int, MorphOp)' instead")]]
cv::Mat OnMorphology(const cv::Mat img, 
                     int etimes, 
                     int dtimes, 
                     int esize, 
                     int dsize, 
                     int flag);

/**
 * Return an image where any pixel that is 255 in img2 and where that 
 * pixel in img1 has intensity 154 or less is set to 255, 0 otherwise.
 */
cv::Mat CompareImagePixels(cv::Mat img1, cv::Mat img2);

/*
 * Suspected: Perform the Ruifrok color deconvolution algorithm
 * m_flag controls the three colour triplets
 */
cv::Mat* DeconvolutionMat(cv::Mat img, int m_flag);

/*
 * Find the side of the car. Perform morphological opening/closing
 * depending on the integer parameters, followed by thresholdind and finding
 * of contours.  The car is represented by those contours within certain 
 * restrictions in size and position.
 */
cv::Rect OnFindCarSide(cv::Mat img, 
                       int etimes, 
                       int dtimes, 
                       int esize, 
                       int dsize, 
                       int thres, 
                       MorphOp op);

/*
 * Perform action similar to 'RestoreImgFromTemp'. This function deals with 
 * RGB and grayscale images.
 *
 * Any pixel where the mask is > 250, copy the source pixel to the 
 * output pixel.  All other pixels are set to 255 on all channels in the case
 * of a RGB image and 0 in the case of a grayscale image.
 */
cv::Mat RemoveFrame(cv::Mat mask, cv::Mat source);
#endif
