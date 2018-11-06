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


std::unique_ptr<plant_data> GetData(const char* filename)
{
  auto p_data = make_unique<wheat_data>();

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


#pragma omp parallel for
    for(int l=0; l<img.cols; l++)
      for(int k=0; k<img.rows; k++)
      {
        if(img.at<Vec3b>(k, l)[0]>img.at<Vec3b>(k, l)[1]*1.5)
        {
          img.at<Vec3b>(k, l)[0]=255;
          img.at<Vec3b>(k, l)[1]=255; 
          img.at<Vec3b>(k, l)[2]=255;
        }
      }

    Mat* pImg0=new Mat[3];
    pImg0=DeconvolutionMat(img, 3);//colour deconvolution to seperate frame and pot off. For top use 8, side use 3;

    Mat conHull;

    cvtColor(pImg0[2], conHull, CV_BGR2GRAY);//for top use pImg[0], side use pImg[2];

    Mat tmp;//frame and pot
    tmp=OnMorphology(conHull, 0, 0, 1, 1, 1);//2, 1, 1, 1, 1; 5, 5, 1, 1, 1

    threshold( tmp, tmp, 180, 255, THRESH_BINARY );//find pot

    //Rect rectA=OnFindCarSide(conHull, 5, 5, 4, 4, 215, 1);//220, normal use
    //Rect rectA=OnFindCarSide(img0, 2, 2, 5, 5, 60, 1);//220, for MS
    Rect rectA=OnFindCarSide(conHull, 9, 9, 1, 1, 180, 1);//200 for w11

    int rx, ry, rw, rh;


    //if(strcmp(s1.c_str(), "2015-04-27")==0)
    //{
    //  rx=817;
    //  ry=2045;
    //  rw=360;
    //  rh=345;
    //  fprintf(fp2, "%d", rx);
    //  fprintf(fp2, "\n");
    //  fprintf(fp2, "%d", ry);
    //  fprintf(fp2, "\n");
    //  fprintf(fp2, "%d", rw);
    //  fprintf(fp2, "\n");
    //  fprintf(fp2, "%d", rh);
    //  fprintf(fp2, "\n");
    //  fclose(fp2);
    //}

    //if(strcmp(s1.c_str(), "2015-04-01")==0)
    //{
    //  rx=740;
    //  ry=1880;
    //  rw=510;
    //  rh=500;
    //  fprintf(fp2, "%d", rx);
    //  fprintf(fp2, "\n");
    //  fprintf(fp2, "%d", ry);
    //  fprintf(fp2, "\n");
    //  fprintf(fp2, "%d", rw);
    //  fprintf(fp2, "\n");
    //  fprintf(fp2, "%d", rh);
    //  fprintf(fp2, "\n");
    //  fclose(fp2);
    //}


    //string sss;
    //sss.assign(outputpath);
    //sss=sss.substr(0, sss.rfind("W8-"));
    //sss.append("\\carsize.txt");

    //char buffer1[256]; 
    //int k1[4], ki1=0;
    //fstream outFile1;  
    //outFile1.open(sss.c_str(),ios::in);  
    //cout<<sss.c_str()<<"--- all file is as follows:---"<<endl;  
    //while(!outFile1.eof())  
    //{  
    //  outFile1.getline(buffer1,256,'\n');//getline(char *,int,char) 表示该行字符达到256个或遇到换行就结束  
    //  k1[ki1]=atoi(buffer1);
    //  cout<<k1[ki1]<<endl;  
    //  ki1=ki1+1;
    //}  

    //outFile1.close();  

    rectA.x=740;
    rectA.y=2310;
    rectA.width=510;
    rectA.height=300;

    Rect rectB;
    rectB.x = rectA.x- rectA.width*1.7;
    if(rectB.x<0)
      rectB.x=10;
    rectB.y = 10;//MS, 100
    rectB.width = rectA.width*4.65;
    if(rectB.x+rectB.width>conHull.cols)
      rectB.width=conHull.cols-2*rectB.x;
    rectB.height = rectA.y+float(rectA.height)*0.07-10;//calculate height from the pot top
    rectB.height = rectA.y+rectA.height-rectA.width-rectB.y;//calculate height from the pot bottom

    rectB.height = 2300;

    Mat tn;
    tn=FindPlantPixels(img, 0.85, 1.3);//original 0.001, w2 is 0.02; 0.7, 0.02; VF1: 0.6, 0.000, w11 0.85, b*1.3<g

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

    Point pend;
    Point tend;

    pend.x=image.cols*0.7;
    pend.y=rectB.y+rectB.height;

    tend.x=image.cols*0.7;
    tend.y=top.y;

    line(image, Point(rectB.x+rectB.width/2, rectB.y+rectB.height), pend, Scalar(0,255,0), 2, 8, 0);
    line(image, top, tend, Scalar(0,255,0), 2, 8, 0);

    line(image, Point(tend.x-10, tend.y), Point(pend.x-10, pend.y), Scalar(0,255,0), 2, 8, 0);

    int plant_height, pot_width;
    plant_height=rectB.y+rectB.height-top.y;//plant height in pixel
    pot_width=rectA.width;//pot width in pixel
    double p_h=double(plant_height)*(250/double(pot_width));//plant height in mm
    leafArea=pixelcount*(250/double(pot_width))*(250/double(pot_width));

    char s[200];
    //sprintf_s(s, 200, "%5.fmm", p_h );
    sprintf(s, "%5.fmm", p_h );
    putText(image, s, Point(image.cols*0.7+5,top.y+plant_height/2), 0, 2, Scalar(155,155,0), 3, 8,false);

    //sprintf_s(s, 200, "%5.fmm", leafArea );
    sprintf(s, "%5.fmm", leafArea );
    putText(image, s, Point(image.cols*0.7+5,top.y+plant_height/2+70), 0, 2, Scalar(155,155,0), 3, 8,false);

    //sprintf_s(s, 200, "2");
    sprintf(s, "2");
    putText(image, s, Point(image.cols*0.7+303+8,top.y+plant_height/2+45), 0, 1, Scalar(155,155,0), 3, 8,false);

    rectangle(image, rectB, Scalar(0, 0, 255), 1, 8, 0);
    rectangle(image, rectA, Scalar(0, 0, 255), 1, 8, 0);


    //if(strcmp(s1.c_str(), "2015-04-27")==0)
    //{
    //  fprintf(fp1, "%d", plant_height);
    //  fprintf(fp1, "\n");
    //  fprintf(fp1, "%d", pot_width);
    //  fprintf(fp1, "\n");
    //  fclose(fp1);
    //}

    int t00=0;
    int t00y=0;
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
          t00=t00+1;

          if(img1.at<Vec3b>(k, l)[2]>img1.at<Vec3b>(k, l)[1]+10)
          {
            image.at<Vec3b>(k, l)[0]=0;
            image.at<Vec3b>(k, l)[1]=0; 
            image.at<Vec3b>(k, l)[2]=200;
            yellowcount=yellowcount+1;
            t00y=t00y+1;
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
      for(int k=top.y+(rectB.y+rectB.height-top.y)*0.2; 
          k<top.y+(rectB.y+rectB.height-top.y)*0.4; k++)
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

    line(image, 
        Point(500, top.y+(rectB.y+rectB.height-top.y)*0.2),
        Point(1954, top.y+(rectB.y+rectB.height-top.y)*0.2), 
        Scalar(0,255,255), 2, 8, 0);

#pragma omp parallel for
    for(int l=rectB.x; l<rectB.x+rectB.width; l++)
      for(int k=top.y+(rectB.y+rectB.height-top.y);
          k<top.y+(rectB.y+rectB.height-top.y)*0.6; 
          k++)
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

    line(image, 
        Point(500, top.y+(rectB.y+rectB.height-top.y)*0.4), 
        Point(1954, top.y+(rectB.y+rectB.height-top.y)*0.4), 
        Scalar(0,255,255), 2, 8, 0);

#pragma omp parallel for
    for(int l=rectB.x; l<rectB.x+rectB.width; l++)
      for(int k=top.y+(rectB.y+rectB.height-top.y)*0.6; 
          k<rectB.y+rectB.height; 
          k++)
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

    line(image, 
        Point(500, top.y+(rectB.y+rectB.height-top.y)*0.6), 
        Point(1954, top.y+(rectB.y+rectB.height-top.y)*0.6), 
        Scalar(0,255,255), 2, 8, 0);

    p_data->plant_height = plant_height;
    p_data->pot_width = pot_width;
    p_data->p_h = p_h;
    p_data->pixelcount = pixelcount;
    p_data->leafArea = leafArea;
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
  auto top = Point(-1, -1);//remember to delete after using
  Mat result;

  if(img.channels()!=1)
    cvtColor(img, result, CV_BGR2GRAY);
  else
    result=img.clone();

  int i, j;
  int count=0;

  for(j=rect.y; j<rect.y+rect.height; j++)
    for(i=rect.x+rect.width*0.1; i<rect.x+rect.width*0.9; i++)//keep away from the region border more than 10% of the width
    {//original includes *(result.data+(j+25)*result.step+i)!=0 ||
      //check if the leaf top checker position is with scope
      int flag1=1;
      int flag2=1;

      if((i-10)<(rect.x+20))
        flag1=0;
      if((i+10)>(rect.x+rect.width-20))
        flag2=0;

      if(*(result.data+j*result.step+i)!=0 && count==0 && (*(result.data+(j+15)*result.step+i)!=0 
            ||(*(result.data+(j+15)*result.step+i+6)!=0 && flag2)||(*(result.data+(j+15)*result.step+i-6)!=0 && flag1)||(*(result.data+(j+15)*result.step+i+15)!=0 && flag2)
            ||(*(result.data+(j+15)*result.step+i-15)!=0)||(*(result.data+(j+15)*result.step+i+25)!=0 && flag2)
            ||(*(result.data+(j+15)*result.step+i-25)!=0) && flag1))//check if the left and right pixels below are leaf pixels
      {
        top.x=i;
        top.y=j;
        count=count+1;
        break;
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
        b=double(*(output.data+i*output.channels()+j*output.step))/255.0;
        g=double(*(output.data+i*output.channels()+j*output.step+1))/255.0;
        r=double(*(output.data+i*output.channels()+j*output.step+2))/255.0;

        //if(g<gthres && r>=b && g>=b)
        if(g<gthres && r< gthres && b<gthres && b*gbthres<g)
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
      }
  }

  return output;
}
