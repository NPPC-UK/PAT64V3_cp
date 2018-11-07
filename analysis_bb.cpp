#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <regex>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>

#include "analysis.h"
#include "analysis_utils.h"
#include "wheat_data.h"


using namespace cv;
using namespace std;

Point* OnPlantTopTiller(Mat img, Mat temp, Rect rect, int width, int height, int thres);

std::unique_ptr<plant_data> GetData(const char* filename)
{
  auto p_data = std::make_unique<wheat_data>();

  Mat img;
  img=imread(filename);


  if(img.data)
  {
    Mat image=img.clone();

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);

    //Mat img0=img.clone();
    Mat img1=img.clone();

    TermCriteria termcrit=TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5,1);
    //pyrMeanShiftFiltering( img0, img0, 1, 1, 1,termcrit);


    /*#pragma omp parallel for
      for(int l=0; l<img.cols; l++)
      for(int k=0; k<img.rows; k++)
      {
      if(img.at<Vec3b>(k, l)[0]>img.at<Vec3b>(k, l)[1]*1.5)
      {
      img.at<Vec3b>(k, l)[0]=255;
      img.at<Vec3b>(k, l)[1]=255; 
      img.at<Vec3b>(k, l)[2]=255;
      }
      }*/

    Mat* pImg0=new Mat[3];
    pImg0=DeconvolutionMat(img, 3);//colour deconvolution to seperate frame and pot off. For top use 8, side use 3;

    Mat conHull;

    cvtColor(pImg0[2], conHull, CV_BGR2GRAY);//for top use pImg[0], side use pImg[2];

    Mat tmp;//frame and pot
    tmp=OnMorphology(conHull, 1, 1, 1, 1, MorphOp::Open);//2, 1, 1, 1, 1; 5, 5, 1, 1, 1

    threshold( tmp, tmp, 180, 255, THRESH_BINARY );//find pot

    //Rect rectA=OnFindCarSide(conHull, 5, 5, 4, 4, 215, 1);//220, normal use
    //Rect rectA=OnFindCarSide(img0, 2, 2, 5, 5, 60, 1);//220, for MS
    //Rect rectA=OnFindCarSide(conHull, 3, 3, 4, 4, 200, 1);

    Rect rectB;
    /*rectB.x = rectA.x- rectA.width*1.7;
      if(rectB.x<0)
      rectB.x=10;
      rectB.y = 10;//MS, 100
      rectB.width = rectA.width*4.65;
      if(rectB.x+rectB.width>conHull.cols)
      rectB.width=conHull.cols-2*rectB.x;
      rectB.height = rectA.y+rectA.height*0.07;//calculate height from the pot top*/
    //rectB.height = rectA.y+rectA.height-rectA.width-rectB.y;//calculate height from the pot bottom
    /*rectB.x=10;
      rectB.y=10;
      rectB.width=img.cols-20;
      rectB.height=img.rows-525-30;*///before 27/09/2017
    rectB.x=200;
    rectB.y=50;
    rectB.width=img.cols-400;
    rectB.height=img.rows-380-50;//after 27/09/2017

    Mat tn;
    tn=FindPlantPixels(img, 10, 1.0);//original 0.001, w2 is 0.02; 0.7, 0.02; VF1: 0.6, 0.000, w8: 0.6, 1; w11 v1: 0.85, 1.3;img, 10, 1.0

    cvtColor(tn, tn, CV_BGR2GRAY);

    threshold(tn, tn, 254, 255, CV_THRESH_BINARY_INV);
    //use adaptive thresholding to get bright pixels
    //cvtColor(img, img, CV_BGR2GRAY);
    //adaptiveThreshold(img, img, 255, CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY_INV,75,10);
    //img=img+tn;

    Mat out;
    out=RemoveFrame(tmp, img);//we do not have to remove frame if we can find plant pixel better
    out=tn;

    Mat drawing = Mat::zeros( image.size(), CV_8UC3 );

    Mat roi(out, rectB);
    Mat roi1(image, rectB);
    Mat roi2(drawing, rectB);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    findContours( roi, contours, hierarchy, CV_RETR_CCOMP , CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    out=0;

    double leafArea=0;
    double pixelcount=0;
    int idx = 0;

    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
      int area=0;
      area=contourArea(contours[idx], false);
      Rect trect;
      trect=boundingRect(contours[idx]);

      //if(trect.x+trect.width/2>roi.cols*0.20 && trect.x+trect.width/2<roi.cols*0.8 && trect.y+trect.height/2>rectB.x+rectB.height*0.3)
      //if(trect.x+trect.width/2>roi.cols*0.10 && trect.x+trect.width/2<roi.cols*0.9 && trect.y+trect.height/2>rectB.y+rectB.height*0.1)
      if(trect.x+trect.width/2>roi.cols*0.10 && trect.x+trect.width/2<roi.cols*0.9 && trect.y+trect.height/2>rectB.y+10)
        if(area>200)
        {
          Scalar color( 100, 100, 0 );
          drawContours( roi1, contours, idx, color, CV_FILLED, 8, hierarchy );
          drawContours( roi, contours, idx, color, CV_FILLED, 8, hierarchy );
          drawContours( roi2, contours, idx, color, CV_FILLED, 8, hierarchy );
          pixelcount=pixelcount+area;
        }
    }

    int planttopflag=0;
    auto top=OnPlantTop(out, rectB);//find top pixel of the plant
    if(top.y<0 ||top.x<0)
    {
      top.x=rectB.x+rectB.width/2;
      top.y=rectB.y+rectB.height;
    }

    if(top.y<img.rows*0.02)
      planttopflag=1;



    Point* tiller;
    tiller=OnPlantTopTiller(img1, out, rectB, 5, 20,5000);
    if(tiller[0].y<0 ||tiller[0].x<0)
    {
      tiller[0].x=rectB.x+rectB.width/2;
      tiller[0].y=rectB.y+rectB.height;
    }

    Point pend;
    Point tend;

    pend.x=image.cols*0.7;
    pend.y=rectB.y+rectB.height;

    tend.x=image.cols*0.7;
    tend.y=top.y;

    line(image, Point(rectB.x+rectB.width/2, rectB.y+rectB.height), pend, Scalar(0,255,0), 2, 8, 0);
    line(image, top, tend, Scalar(0,255,0), 2, 8, 0);
    line(image, Point(tend.x-10, tend.y), Point(pend.x-10, pend.y), Scalar(0,255,0), 2, 8, 0);

    tend.x=image.cols*0.3;
    tend.y=tiller[0].y;
    pend.x=image.cols*0.3;

    line(image, tiller[0], tend, Scalar(255,0,0), 2, 8, 0);
    line(image, Point(tend.x+10, tend.y), Point(pend.x+10, pend.y), Scalar(255,0,0), 2, 8, 0);

    int plant_height, pot_width, plant_height_t;
    plant_height=rectB.y+rectB.height-top.y;//plant height in pixel
    plant_height_t=rectB.y+rectB.height-tiller[0].y;//plant height in pixel
    //pot_width=rectA.width;//pot width in pixel
    pot_width=290;
    double p_h=double(plant_height)*(150/double(pot_width));//plant height in mm, for car 250, for pot 150
    double p_h_t=double(plant_height_t)*(150/double(pot_width));//plant height in mm, for car 250, for pot 150
    leafArea=pixelcount*(150/double(pot_width))*(150/double(pot_width));

    char s[200];
    // sprintf_s(s, 200, "%0.fmm", p_h );
    sprintf(s, "%0.fmm", p_h );
    putText(image, s, Point(image.cols*0.7+5,top.y+plant_height/2), 0, 2, Scalar(155,155,0), 3, 8,false);

    //sprintf_s(s, 200, "%0.fmm", p_h_t );
    sprintf(s, "%0.fmm", p_h_t );
    putText(image, s, Point(image.cols*0.25-5,tiller[0].y+plant_height_t/2), 0, 2, Scalar(155,155,0), 3, 8,false);

    //sprintf_s(s, 200, "%0.fsquare mm", leafArea );
    sprintf(s, "%0.fsquare mm", leafArea );
    putText(image, s, Point(image.cols*0.7+5,top.y+plant_height/2+70), 0, 2, Scalar(155,155,0), 3, 8,false);

    rectangle(image, rectB, Scalar(0, 0, 255), 1, 8, 0);
    //rectangle(image, rectA, Scalar(0, 0, 255), 1, 8, 0);

    p_data->plant_height = plant_height;
    p_data->pot_width = pot_width;
    p_data->p_h = p_h;
    p_data->p_h_t = p_h_t;
    p_data->pixelcount = pixelcount;
    p_data->leafArea = leafArea;

    int t20=0;
    int t20y=0;
    int t40=0;
    int t40y=0;
    int t60=0;
    int t60y=0;

    int yellowcount=0;
#pragma omp parallel for
    for(int l=rectB.x; l<rectB.x+rectB.width; l++)
      for(int k=top.y; k<top.y+(rectB.y+rectB.height-top.y)*0.2; k++)
      {
        if(drawing.at<Vec3b>(k, l)[0]>10 || drawing.at<Vec3b>(k, l)[1]>10 || drawing.at<Vec3b>(k, l)[2]>10)
        {
          if(img1.at<Vec3b>(k, l)[2]>img1.at<Vec3b>(k, l)[1]+10)
          {
            image.at<Vec3b>(k, l)[0]=0;
            image.at<Vec3b>(k, l)[1]=0; 
            image.at<Vec3b>(k, l)[2]=200;
            yellowcount=yellowcount+1;
          }
          else
          {
            image.at<Vec3b>(k, l)[0]=100;
            image.at<Vec3b>(k, l)[1]=100; 
            image.at<Vec3b>(k, l)[2]=0;
          }

        }
      }

    line(image, Point(500, top.y), Point(1954, top.y), Scalar(0,255,255), 2, 8, 0);

#pragma omp parallel for
    for(int l=rectB.x; l<rectB.x+rectB.width; l++)
      for(int k=top.y+(rectB.y+rectB.height-top.y)*0.2; k<top.y+(rectB.y+rectB.height-top.y)*0.4; k++)
      {
        if(drawing.at<Vec3b>(k, l)[0]>10 || drawing.at<Vec3b>(k, l)[1]>10 || drawing.at<Vec3b>(k, l)[2]>10)
        {
          t20=t20+1;

          if(img1.at<Vec3b>(k, l)[2]>img1.at<Vec3b>(k, l)[1]+10)
          {
            image.at<Vec3b>(k, l)[0]=0;
            image.at<Vec3b>(k, l)[1]=0; 
            image.at<Vec3b>(k, l)[2]=200;
            yellowcount=yellowcount+1;
            t20y=t20y+1;
          }
          else
          {
            image.at<Vec3b>(k, l)[0]=100;
            image.at<Vec3b>(k, l)[1]=100; 
            image.at<Vec3b>(k, l)[2]=0;
          }

        }
      }

    line(image, Point(500, top.y+(rectB.y+rectB.height-top.y)*0.2), Point(1954, top.y+(rectB.y+rectB.height-top.y)*0.2), Scalar(0,255,255), 2, 8, 0);

#pragma omp parallel for
    for(int l=rectB.x; l<rectB.x+rectB.width; l++)
      for(int k=top.y+(rectB.y+rectB.height-top.y)*0.4; k<top.y+(rectB.y+rectB.height-top.y)*0.6; k++)
      {
        if(drawing.at<Vec3b>(k, l)[0]>10 || drawing.at<Vec3b>(k, l)[1]>10 || drawing.at<Vec3b>(k, l)[2]>10)
        {
          t40=t40+1;

          if(img1.at<Vec3b>(k, l)[2]>img1.at<Vec3b>(k, l)[1]+10)
          {
            image.at<Vec3b>(k, l)[0]=0;
            image.at<Vec3b>(k, l)[1]=0; 
            image.at<Vec3b>(k, l)[2]=200;
            yellowcount=yellowcount+1;
            t40y=t40y+1;
          }
          else
          {
            image.at<Vec3b>(k, l)[0]=100;
            image.at<Vec3b>(k, l)[1]=100; 
            image.at<Vec3b>(k, l)[2]=0;
          }

        }
      }

    line(image, Point(500, top.y+(rectB.y+rectB.height-top.y)*0.4), Point(1954, top.y+(rectB.y+rectB.height-top.y)*0.4), Scalar(0,255,255), 2, 8, 0);

#pragma omp parallel for
    for(int l=rectB.x; l<rectB.x+rectB.width; l++)
      for(int k=top.y+(rectB.y+rectB.height-top.y)*0.6; k<rectB.y+rectB.height; k++)
      {
        if(drawing.at<Vec3b>(k, l)[0]>10 || drawing.at<Vec3b>(k, l)[1]>10 || drawing.at<Vec3b>(k, l)[2]>10)
        {
          t60=t60+1;

          if(img1.at<Vec3b>(k, l)[2]>img1.at<Vec3b>(k, l)[1]+10)
          {
            image.at<Vec3b>(k, l)[0]=0;
            image.at<Vec3b>(k, l)[1]=0; 
            image.at<Vec3b>(k, l)[2]=200;
            yellowcount=yellowcount+1;
            t60y=t60y+1;
          }
          else
          {
            image.at<Vec3b>(k, l)[0]=100;
            image.at<Vec3b>(k, l)[1]=100; 
            image.at<Vec3b>(k, l)[2]=0;
          }

        }
      }

    line(image, Point(500, top.y+(rectB.y+rectB.height-top.y)*0.6), Point(1954, top.y+(rectB.y+rectB.height-top.y)*0.6), Scalar(0,255,255), 2, 8, 0);


    p_data->yellowcount = yellowcount;
    p_data->t20 = t20;
    p_data->t20y = t20y;
    p_data->t40 = t40;
    p_data->t40y = t40y;
    p_data->t60 = t60;
    p_data->t60y = t60y;


    p_data->image = image;

    delete[] pImg0;
  }
  return p_data;
}

Point OnPlantTop(Mat img, Rect rect)
{
  auto top = cv::Point(-1, -1);//remember to delete after using
  Mat result;

  if(img.channels()!=1)
    cvtColor(img, result, CV_BGR2GRAY);
  else
    result=img.clone();

  int i, j;
  int count=0;

  for(j=rect.y; j<rect.y+rect.height; j++)
  {
    for(i=rect.x+rect.width*0.1; i<rect.x+rect.width*0.9; i++)//keep away from the region border more than 10% of the width
    {//original includes *(result.data+(j+25)*result.step+i)!=0 ||
      //check if the leaf top checker position is with scope
      int flag1=1;
      int flag2=1;

      if((i-10)<(rect.x+20))
        flag1=0;
      if((i+10)>(rect.x+rect.width-20))
        flag2=0;

      if(*(result.data+j*result.step+i)!=0 && count==0 && (*(result.data+(j+10)*result.step+i)!=0 
            ||(*(result.data+(j+10)*result.step+i+4)!=0 && flag2)||(*(result.data+(j+10)*result.step+i-4)!=0 && flag1)||(*(result.data+(j+10)*result.step+i+10)!=0 && flag2)
            ||(*(result.data+(j+10)*result.step+i-10)!=0) && flag1))//check if the left and right pixels below are leaf pixels
      {
        top.x=i;
        top.y=j;
        count=count+1;
        break;
      }
    }
  }
  return top;
}



Mat FindPlantPixels(Mat img, double gthres, double gbthres)
{
  Mat output;
  output=img.clone();

  double b, g, r;

  int i, j;
  if(img.channels()==3)
  {
    for(i=0; i<img.cols; i++)
      for(j=0; j<img.rows; j++)
      {
        b=double(*(output.data+i*output.channels()+j*output.step))/1.0;
        g=double(*(output.data+i*output.channels()+j*output.step+1))/1.0;
        r=double(*(output.data+i*output.channels()+j*output.step+2))/1.0;
        /*b=double(*(output.data+i*output.channels()+j*output.step))/255.0;
          g=double(*(output.data+i*output.channels()+j*output.step+1))/255.0;
          r=double(*(output.data+i*output.channels()+j*output.step+2))/255.0;*/

        //if(g<gthres && r>=b && g>=b)
        if((b>gthres && r> gthres && g>gthres && b*gbthres<g))
        {
          *(output.data+i*output.channels()+j*output.step)=*(output.data+i*output.channels()+j*output.step);
          *(output.data+i*output.channels()+j*output.step+1)=*(output.data+i*output.channels()+j*output.step+1);
          *(output.data+i*output.channels()+j*output.step+2)=*(output.data+i*output.channels()+j*output.step+2);
        }
        else
        {
          *(output.data+i*output.channels()+j*output.step)=255;
          *(output.data+i*output.channels()+j*output.step+1)=255;
          *(output.data+i*output.channels()+j*output.step+2)=255;
        }

        if(g<r && g<100 && r<100)
        {
          *(output.data+i*output.channels()+j*output.step)=255;
          *(output.data+i*output.channels()+j*output.step+1)=255;
          *(output.data+i*output.channels()+j*output.step+2)=255;
        }
      }
  }

  return output;
}

Point* OnPlantTopTiller(Mat img, Mat temp, Rect rect, int width, int height, int thres)
{
  Point* tops=new Point[4];
  Mat gray;
  if(img.channels()!=1)
    cvtColor(img, gray, CV_BGR2GRAY);

  int count=0;

  int i, j, flag;
  Mat drawing = Mat::zeros( img.size(), CV_8UC1 );
  for(i=rect.x+width; i<rect.x+rect.width-width; i=i+1)//scan image points according to their heights. No deeper than 2090 in Y direction to remove pot patches (358 is the average height of the pots)
    for(j=rect.y; j<rect.y+rect.height-height; j=j+1)
      if(temp.at<uchar>(j, i)>10) 
      {
        int th=0;
        int bh=0;
        flag=0;
        for(int k=i-width/2; k<=i+width/2; k++)
          for(int l=j-height/2; l<=j-1; l++)
          {
            if(temp.at<uchar>(l, k)>10)
              th=th+gray.at<uchar>(l, k);
            else
              flag=1;
          }
        for(int k=i-width/2; k<=i+width/2; k++)
          for(int l=j+1; l<=j+height/2; l++)
          {
            if(temp.at<uchar>(l, k)>10)
              bh=bh+gray.at<uchar>(l, k);
            else
              flag=1;
          }
        if(abs(bh-th)>=thres && flag==0)
          drawing.at<uchar>(j, i)=255;
      }

  drawing=OnMorphology(drawing, 2, 2, 1, 1, MorphOp::Open);

  for(j=rect.y; j<rect.y+rect.height; j++)
    for(i=rect.x+20; i<rect.x+rect.width-20; i++)//keep away from the region within 30 pixels to the central panel
    {//original includes *(result.data+(j+25)*result.step+i)!=0 ||
      //check if the leaf top checker position is with scope
      int flag1=1;
      int flag2=1;

      if((i-10)<(rect.x+20))
        flag1=0;
      if((i+10)>(rect.x+rect.width-20))
        flag2=0;

      if(*(drawing.data+j*drawing.step+i)!=0 && count==0 && (*(drawing.data+(j+20)*drawing.step+i)!=0 
            ||(*(drawing.data+(j+20)*drawing.step+i+0)!=0 && flag2)||(*(drawing.data+(j+20)*drawing.step+i-0)!=0 && flag1)))//check if the left and right pixels below are leaf pixels
      {
        tops[0].x=i;
        tops[0].y=j;
        count=count+1;
        break;
      }
    }
  if(count>0)
    return tops;
  else
  {
    tops[0].x=-1;
    tops[0].y=-1;
    return tops;
  }
  return tops;
}
