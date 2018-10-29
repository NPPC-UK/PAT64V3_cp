#include "analysis_utils.h"

int OnCountPixels(const cv::Mat img, unsigned int pottop, unsigned int planttop) {
  cv::Mat result;

  if(img.channels()!=1)
    cvtColor(img, result, CV_BGR2GRAY);
  else
    result=img.clone();

  int i, j;
  int count=0;

  for(j=planttop; j<pottop; j++)
    for(i=result.cols*0.15; i<result.cols*0.85; i++)
    {
      if(*(result.data+j*result.step+i)<=210)
      {
        count=count+1;
      }
    }
  return count;
}

cv::Mat RestoreImgFromTemp(const cv::Mat temp, const cv::Mat source) {
  //temp is the template holding the information where leaf pixels are. source is the original input image
  auto output=source.clone();

  int i, j;
  for(i=0; i<temp.cols; i++)
    for(j=0; j<temp.rows; j++)
    {
      if(*(temp.data+j*temp.step+i)!=0)
      {
        *(output.data+i*source.channels()+j*source.step)=*(source.data+i*source.channels()+j*source.step);
        *(output.data+i*source.channels()+j*source.step+1)=*(source.data+i*source.channels()+j*source.step+1);
        *(output.data+i*source.channels()+j*source.step+2)=*(source.data+i*source.channels()+j*source.step+2);
      }
      else
      {
        *(output.data+i*source.channels()+j*source.step)=255;
        *(output.data+i*source.channels()+j*source.step+1)=255;
        *(output.data+i*source.channels()+j*source.step+2)=255;
      }
    }

  return output;
}

