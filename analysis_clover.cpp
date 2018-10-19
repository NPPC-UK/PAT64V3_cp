#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <regex>
#include <string>

#include <opencv2/opencv.hpp>

#include "analysis.h"
#include "format_string.h"
#include "clover_data.h"

using namespace cv;
using namespace std;

Mat* DeconvolutionMat(Mat img, int m_flag);
Mat CompareImagePixels(Mat img1, Mat img2);
Point* OnPlantTop(Mat img, Rect rect);
Mat OnMorphology(Mat img, int etimes, int dtimes, int esize, int dsize, int flag);
Rect OnFindCarSide(Mat img, int etimes , int dtimes , int esize , int dsize , int thres , int flag);
Point* OnPotPoints(Mat img);
Mat RestoreImgFromTemp(Mat temp, Mat source);
int OnCountPixels(Mat img, int pottop, int planttop);
Mat RemoveFrame(Mat temp, Mat source);
Mat FindPlantPixels(Mat img, double gthres, double gbthres);

plant_data GetData(const char* filename)
{
  clover_data p_data;

  string s, s1, s2;

  Mat img;
  img=imread(filename);


  if(img.data)
  {
    string s3;
    s1.append("-11");
    s2.append("-21");
    s3.append("-31");

    Mat image=img.clone();
    int oflag=0;//overlapping flag

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);

    Mat img0=img.clone();
    Mat img1=img.clone();

#pragma omp parallel for
    for(int l=0; l<img0.cols; l++)
      for(int k=0; k<img0.rows; k++)
      {
        if((img0.at<Vec3b>(k, l)[1]*0.95<=img0.at<Vec3b>(k, l)[0] || img0.at<Vec3b>(k, l)[1]*1.3<=img0.at<Vec3b>(k, l)[2])
            || (img0.at<Vec3b>(k, l)[0]<40 && img0.at<Vec3b>(k, l)[1]<40 && img0.at<Vec3b>(k, l)[2]<40)
            || (img0.at<Vec3b>(k, l)[0]>250 || img0.at<Vec3b>(k, l)[1]>250 || img0.at<Vec3b>(k, l)[2]>250))
        {
          img0.at<Vec3b>(k, l)[0]=255;
          img0.at<Vec3b>(k, l)[1]=255; 
          img0.at<Vec3b>(k, l)[2]=255;
        }
      }

    Mat tf = Mat::zeros( img0.size(), CV_8UC3 );

#pragma omp parallel for
    for(int l=0; l<img0.cols; l++)
      for(int k=0; k<img0.rows; k++)
      {
        if(img.at<Vec3b>(k, l)[0]>165 && img.at<Vec3b>(k, l)[1]>165 && img.at<Vec3b>(k, l)[2]>165)
        {
          tf.at<Vec3b>(k, l)[0]=255;
          tf.at<Vec3b>(k, l)[1]=255; 
          tf.at<Vec3b>(k, l)[2]=255;
        }
      }

    int frame_y=0;
    int mini_fy=10000;

    for(int l=1500; l<2000; l++)
      for(int k=0; k<900; k++)
      {
        if(tf.at<Vec3b>(k, l)[0]>250 && tf.at<Vec3b>(k, l)[1]>250 && tf.at<Vec3b>(k, l)[2]>250)
        {
          if(k<mini_fy)
            mini_fy=k;
        }
      }

    Mat tmp;
    tmp=img0;

    cvtColor(tmp, tmp, CV_BGR2GRAY);

    threshold(tmp, tmp, 254, 255, CV_THRESH_BINARY_INV);

    tmp=OnMorphology(tmp, 1, 1, 1, 1, 1);
    tmp=OnMorphology(tmp, 2, 2, 1, 1, 0);

    vector<vector<Point>> p[4];
    vector<vector <Point>> ocon;

    vector<vector<Point> > oc;

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;


    int dx, dy;
    dx=0;
    dy=mini_fy-660;

    rectangle(tmp, Rect(0, 1350+dy, 6000, 4), Scalar(0,0,0), -1);
    rectangle(tmp, Rect(0, 2100+dy, 6000, 4), Scalar(0,0,0), -1);
    rectangle(tmp, Rect(2985+dx, 500, 4, 2500), Scalar(0,0,0), -1);


    int idx = 0;
    Mat drawing = Mat::zeros( tmp.size(), CV_8UC3 );
    Mat tp = Mat::zeros( tmp.size(), CV_8UC1 );

    Rect rect;
    rect.x=370+dx;
    rect.y=700+dy;
    rect.width=20;
    rect.height=570;

    rectangle(drawing, rect, Scalar(51, 0, 255), -1, 8, 0);

    rect.x=5590+dx;
    rect.y=700+dy;
    rectangle(drawing, rect, Scalar(51, 0, 255), -1, 8, 0);

    rect.x=370+dx;
    rect.y=1450+dy;
    rectangle(drawing, rect, Scalar(51, 0, 255), -1, 8, 0);

    rect.x=5590+dx;
    rect.y=1450+dy;
    rectangle(drawing, rect, Scalar(51, 0, 255), -1, 8, 0);

    rect.x=370+dx;
    rect.y=2180+dy;
    rectangle(drawing, rect, Scalar(51, 0, 255), -1, 8, 0);

    rect.x=5590+dx;
    rect.y=2180+dy;
    rectangle(drawing, rect, Scalar(51, 0, 255), -1, 8, 0);

    rect.x=330+dx;
    rect.y=670+dy;
    rect.width=5300;
    rect.height=530;

    for(int i=rect.x; i<rect.x+rect.width; i++)//define ROIs only containing plants
      for(int j=rect.y; j<rect.y+rect.height; j++)
      {
        tp.at<uchar>(j, i)=tmp.at<uchar>(j, i);
      }


    rect.y=1430+dy;

    for(int i=rect.x; i<rect.x+rect.width; i++)
      for(int j=rect.y; j<rect.y+rect.height; j++)
      {
        tp.at<uchar>(j, i)=tmp.at<uchar>(j, i);
      }


    rect.y=2170+dy;

    for(int i=rect.x; i<rect.x+rect.width; i++)
      for(int j=rect.y; j<rect.y+rect.height; j++)
      {
        tp.at<uchar>(j, i)=tmp.at<uchar>(j, i);
      }

    findContours( tp, contours, hierarchy, CV_RETR_CCOMP , CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );//find contours from ROIs

    Point2f centre[6];
    centre[0].x=600+dx;
    centre[0].y=985+dy;

    centre[1].x=5360+dx;
    centre[1].y=985+dy;

    centre[2].x=600+dx;
    centre[2].y=1735+dy;

    centre[3].x=5360+dx;
    centre[3].y=1735+dy;

    centre[4].x=600+dx;
    centre[4].y=2465+dy;

    centre[5].x=5360+dx;
    centre[5].y=2465+dy;

    Mat inter, olap;
    inter=Mat::zeros( tmp.size(), CV_8UC3 );
    olap=Mat::zeros( tmp.size(), CV_8UC3 );

    int tc=0; //total number of plant blocks
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
      int area=0;
      area=contourArea(contours[idx], false);
      if(area>100)
      {
        Scalar color( 100, 100, 0 );
        /// Get the moments
        Moments mu;
        mu = moments( contours[idx], false ); 
        /// Get the mass centers:
        Point2f mc;
        mc = Point2f( mu.m10/mu.m00 , mu.m01/mu.m00 );

        double mindis=20000;
        double xx;
        double yy;
        int fx=0;
        int fy=0;

        int pos=0;
        for(int i=0; i<6; i++)
        {
          xx=mc.x-centre[i].x;
          yy=mc.y-centre[i].y;

          if(fabs(xx)<50)
            fx=1;
          if(fabs(yy)<50)
            fy=1;
          //find the closest centre position
          double dis=sqrt(xx*xx+yy*yy);
          if(dis<mindis)
          {
            mindis=dis;
            pos=i;
          }
        }

        int cdis=0;
        for(int i=0; i<6; i++)
        {
          double dis=pointPolygonTest(contours[idx], centre[i], true);//decide if centre is within this contour

          if(fabs(dis)<40)
          {
            cdis=cdis+1;
          }
        }

        if(cdis<2 && mindis<250)
          drawContours( inter, contours, idx, color, CV_FILLED, 8, hierarchy );
        else
        {
          if(cdis>=2)
            oflag=1;
          drawContours( olap, contours, idx, color, CV_FILLED, 8, hierarchy );
        }

      }
    }

    inter=inter+olap;

    Scalar tt;
    tt=sum(inter);

    contours.clear();
    hierarchy.clear();
    tc=0;
    idx=0;

    cvtColor(inter, inter, CV_BGR2GRAY);
    threshold(inter, inter,50, 255, CV_THRESH_BINARY);

    if(tt.val[0]!=0)
    {
      findContours( inter, contours, hierarchy, CV_RETR_CCOMP , CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

      for( ; idx >= 0; idx = hierarchy[idx][0] )
      {
        int area=0;
        area=contourArea(contours[idx], false);
        if(area>200)
        {
          Scalar color[6];
          color[0]= Scalar( 255, 50, 50 );
          color[1]= Scalar( 102, 255, 255 );
          color[2]= Scalar( 255, 204, 255 );
          color[3]= Scalar( 255, 225, 0 );
          color[4]= Scalar( 200, 104, 255 );
          color[5]= Scalar( 155, 205, 0 );

          /// Get the moments
          Moments mu;
          mu = moments( contours[idx], false ); 
          /// Get the mass centers:
          Point2f mc;
          mc = Point2f( mu.m10/mu.m00 , mu.m01/mu.m00 );


          double mindis=20000;
          int pos=0;
          for(int i=0; i<6; i++)
          {
            double dis=sqrt((mc.x-centre[i].x)*(mc.x-centre[i].x)+(mc.y-centre[i].y)*(mc.y-centre[i].y));
            if(dis<mindis)
            {
              mindis=dis;
              pos=i;
            }
          }

          if(mindis<5000)
          {
            drawContours( drawing, contours, idx, color[pos], CV_FILLED, 8, hierarchy );
            p_data.pixelcount[pos] += area;
          }
        }
      }

      for(int i=0; i<drawing.cols; i++)
        for(int j=0; j<drawing.rows; j++)
        {
          if(drawing.at<Vec3b>(j, i)[1]==50 &&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[1]*0.95)&&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[0]*1.5)&& img.at<Vec3b>(j, i)[2]>50)
          {
            p_data.yellowcount[0]++;
            drawing.at<Vec3b>(j,i)[0]=0;
            drawing.at<Vec3b>(j,i)[1]=15;
            drawing.at<Vec3b>(j,i)[2]=255;
          }
          if(drawing.at<Vec3b>(j, i)[1]==255 &&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[1]*0.95)&&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[0]*1.5)&& img.at<Vec3b>(j, i)[2]>50)
          {
            p_data.yellowcount[1]++;
            drawing.at<Vec3b>(j,i)[0]=0;
            drawing.at<Vec3b>(j,i)[1]=15;
            drawing.at<Vec3b>(j,i)[2]=255;
          }
          if(drawing.at<Vec3b>(j, i)[1]==204 &&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[1]*0.95)&&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[0]*1.5)&& img.at<Vec3b>(j, i)[2]>50)
          {
            p_data.yellowcount[2]++;
            drawing.at<Vec3b>(j,i)[0]=0;
            drawing.at<Vec3b>(j,i)[1]=15;
            drawing.at<Vec3b>(j,i)[2]=255;
          }
          if(drawing.at<Vec3b>(j, i)[1]==225 &&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[1]*0.95)&&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[0]*1.5)&& img.at<Vec3b>(j, i)[2]>50)
          {
            p_data.yellowcount[3]++;
            drawing.at<Vec3b>(j,i)[0]=0;
            drawing.at<Vec3b>(j,i)[1]=15;
            drawing.at<Vec3b>(j,i)[2]=255;
          }
          if(drawing.at<Vec3b>(j, i)[1]==104 &&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[1]*0.95)&&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[0]*1.5)&& img.at<Vec3b>(j, i)[2]>50)
          {
            p_data.yellowcount[4]++;
            drawing.at<Vec3b>(j,i)[0]=0;
            drawing.at<Vec3b>(j,i)[1]=15;
            drawing.at<Vec3b>(j,i)[2]=255;
          }
          if(drawing.at<Vec3b>(j, i)[1]==205 &&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[1]*0.95)&&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[0]*1.5)&& img.at<Vec3b>(j, i)[2]>50)
          {
            p_data.yellowcount[5]++;
            drawing.at<Vec3b>(j,i)[0]=0;
            drawing.at<Vec3b>(j,i)[1]=15;
            drawing.at<Vec3b>(j,i)[2]=255;
          }
        }


      Point top[6];

      for(int i=2985+dx; i>=370+dx; i=i-1)//top left
        for(int j=700+dy; j<=1270+dy; j=j+1)
        {
          if(drawing.at<Vec3b>(j, i)[1]==50 &&(drawing.at<Vec3b>(j-20, i-50)[1]>10||drawing.at<Vec3b>(j+20, i-50)[1]>10))
          {
            for(int k=-100; k<=100; k++)
            {
              if(drawing.at<Vec3b>(j+k, i-60)[1]==50)
              {
                top[0].x=i;
                top[0].y=j;

                line(drawing, top[0], Point(i, 700+dy), Scalar(0,255,0), 2, 8, 0);
                //line(drawing, Point(370+dx, 1120+dy), Point(160+dx, 700+dy), Scalar(0,255,0), 2, 8, 0);
                line(drawing, Point(370+dx, 700+dy), Point(i, 700+dy), Scalar(0,255,0), 2, 8, 0);
                p_data.length[0]=float(i-370-dx)*0.19;

                s = stringf("%0.fmm", float(i-370-dx)*0.19);
                putText(drawing, s, Point(370+dx+(i-370-dx)/2, 680+dy), 0, 2, Scalar(155,155,0), 3, 8,false);

                j=1271+dx;
                i=369+dy;

                break;
              }
            }
          }
        }

      for(int i=2985+dx; i<=5610+dx; i=i+1)//top right
        for(int j=700+dy; j<=1270+dy; j=j+1)
        {
          if(drawing.at<Vec3b>(j, i)[1]==255 &&(drawing.at<Vec3b>(j-20, i+50)[1]>10||drawing.at<Vec3b>(j+20, i+50)[1]>10))
          {
            for(int k=-100; k<=100; k++)
            {
              if(drawing.at<Vec3b>(j+k, i+60)[1]==255)
              {
                top[1].x=i;
                top[1].y=j;

                line(drawing, top[1], Point(i, 700+dy), Scalar(0,255,0), 2, 8, 0);
                //line(drawing, Point(5230+dx, 1070+dy), Point(5230+dx, 700+dy), Scalar(0,255,0), 2, 8, 0);
                line(drawing, Point(5610+dx, 700+dy), Point(i, 700+dy), Scalar(0,255,0), 2, 8, 0);
                p_data.length[1]=float(5610+dx-i)*0.19;

                s = stringf("%0.fmm", float(5610+dx-i)*0.19);
                putText(drawing, s, Point(5610+dx-(5610+dx-i)/2, 680+dy), 0, 2, Scalar(155,155,0), 3, 8,false);

                i=5611+dx;
                j=1271+dy;

                break;
              }
            }
          }
        }
      for(int i=2985+dx; i>=370+dx; i=i-1)//bottom left
        for(int j=1450+dy; j<=2020+dy; j=j+1)
        {
          if(drawing.at<Vec3b>(j, i)[1]==204 &&(drawing.at<Vec3b>(j-20, i-50)[1]>10||drawing.at<Vec3b>(j+20, i-50)[1]>10))
          {
            for(int k=-100; k<=100; k++)
            {
              if(drawing.at<Vec3b>(j+k, i-60)[1]==204)
              {
                top[2].x=i;
                top[2].y=j;

                line(drawing, top[2], Point(i, 1450+dy), Scalar(0,255,0), 2, 8, 0);
                //line(drawing, Point(160+dx, 2850+dy), Point(160+dx, 2340+dy), Scalar(0,255,0), 2, 8, 0);
                line(drawing, Point(i, 1450+dy), Point(370+dx, 1450+dy), Scalar(0,255,0), 2, 8, 0);
                p_data.length[2]=float(i-370-dx)*0.19;

                s = stringf("%0.fmm", float(i-370-dx)*0.19);
                putText(drawing, s, Point(370+dx+(i-370-dx)/2, 1430+dy), 0, 2, Scalar(155,155,0), 3, 8,false);

                i=369+dx;
                j=2021+dy;

                break;
              }
            }
          }
        }
      for(int i=2985+dx; i<=5610+dx; i=i+1)//bottom right
        for(int j=1450+dy; j<=2020+dy; j=j+1)
        {
          if(drawing.at<Vec3b>(j, i)[1]==225 &&(drawing.at<Vec3b>(j-20, i+50)[1]>10||drawing.at<Vec3b>(j+20, i+50)[1]>10))
          {
            for(int k=-100; k<=100; k++)
            {
              if(drawing.at<Vec3b>(j+k, i+60)[1]==225)
              {
                top[3].x=i;
                top[3].y=j;

                line(drawing, top[3], Point(i, 1450+dy), Scalar(0,255,0), 2, 8, 0);
                //line(drawing, Point(5230+dx, 2850+dy), Point(5230+dx, 2310+dy), Scalar(0,255,0), 2, 8, 0);
                line(drawing, Point(i, 1450+dy), Point(5610+dx, 1450+dy), Scalar(0,255,0), 2, 8, 0);
                p_data.length[3]=float(5610+dx-i)*0.19;

                s = stringf(s, 200, "%0.fmm", float(5610+dx-i)*0.19);
                putText(drawing, s, Point(5610+dx-(5610+dx-i)/2, 1430+dy), 0, 2, Scalar(155,155,0), 3, 8,false);

                i=5611+dx;
                j=2021+dy;

                break;
              }
            }
          }
        }
      for(int i=2985+dx; i>=370+dx; i=i-1)//bottom left
        for(int j=2180+dy; j<=2750+dy; j=j+1)
        {
          if(drawing.at<Vec3b>(j, i)[1]==104 &&(drawing.at<Vec3b>(j-20, i-50)[1]>10||drawing.at<Vec3b>(j+20, i-50)[1]>10))
          {
            for(int k=-100; k<=100; k++)
            {
              if(drawing.at<Vec3b>(j+k, i-60)[1]==104)
              {
                top[4].x=i;
                top[4].y=j;

                line(drawing, top[4], Point(i, 2180+dy), Scalar(0,255,0), 2, 8, 0);
                //line(drawing, Point(160+dx, 2850+dy), Point(160+dx, 2340+dy), Scalar(0,255,0), 2, 8, 0);
                line(drawing, Point(i, 2180+dy), Point(370+dx, 2180+dy), Scalar(0,255,0), 2, 8, 0);
                p_data.length[4]=float(i-370-dx)*0.19;

                s = stringf("%0.fmm", float(i-370-dx)*0.19);
                putText(drawing, s, Point(370+dx+(i-370-dx)/2, 2160+dy), 0, 2, Scalar(155,155,0), 3, 8,false);

                i=369+dx;
                j=2751+dy;

                break;
              }
            }
          }
        }
      for(int i=2985+dx; i<=5610+dx; i=i+1)//bottom right
        for(int j=2180+dy; j<=2750+dy; j=j+1)
        {
          if(drawing.at<Vec3b>(j, i)[1]==205 &&(drawing.at<Vec3b>(j-20, i+50)[1]>10||drawing.at<Vec3b>(j+20, i+50)[1]>10))
          {
            for(int k=-100; k<=100; k++)
            {
              if(drawing.at<Vec3b>(j+k, i+60)[1]==205)
              {
                top[5].x=i;
                top[5].y=j;

                line(drawing, top[5], Point(i, 2180+dy), Scalar(0,255,0), 2, 8, 0);
                //line(drawing, Point(5230+dx, 2850+dy), Point(5230+dx, 2310+dy), Scalar(0,255,0), 2, 8, 0);
                line(drawing, Point(i, 2180+dy), Point(5610+dx, 2180+dy), Scalar(0,255,0), 2, 8, 0);
                p_data.length[5]=float(5610+dx-i)*0.19;

                s = stringf("%0.fmm", float(5610+dx-i)*0.19);
                putText(drawing, s, Point(5610+dx-(5610+dx-i)/2, 2160+dy), 0, 2, Scalar(155,155,0), 3, 8,false);

                i=5611+dx;
                j=2751+dy;

                break;
              }
            }
          }
        }

      putText(drawing, s1, Point(2800+dx, 550+dy), 0, 2, Scalar(255,255,0), 6, 8,false);
      putText(drawing, s2, Point(2800+dx, 1300+dy), 0, 2, Scalar(255,255,0), 6, 8,false);
      putText(drawing, s3, Point(2800+dx, 2030+dy), 0, 2, Scalar(255,255,0), 6, 8,false);

      Point org;
      org.x = 20;
      org.y = img0.rows*0.1;

      img0=img*0.3+drawing*0.7;

      p_data.image = img0;

      //fprintf(fp, "%s,%0.f,%0.f,%0.f,%0.f,%0.f,%0.f,%s,%0.f,%0.f,%0.f,%0.f,%0.f,%0.f,%s,%0.f,%0.f,%0.f,%0.f,%0.f,%0.f\n", s1, length[0], pixelcount[0], yellowcount[0], length[1], pixelcount[1], yellowcount[1], s2, length[2], pixelcount[2], yellowcount[2], length[3], pixelcount[3], yellowcount[3],s3, length[4], pixelcount[4], yellowcount[4], length[5], pixelcount[5], yellowcount[5]);

      return p_data;
    }
  }
}

int OnCountPixels(Mat img, int pottop, int planttop)
{
  Mat result;

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

Mat RestoreImgFromTemp(Mat temp, Mat source)
{
  //temp is the template holding the information where leaf pixels are. source is the original input image
  Mat output;
  output=source.clone();

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

Point* OnPotPoints(Mat img)
{
  Point* pots=new Point[4];//remember to delete after using
  Mat result;

  if(img.channels()!=1)
    cvtColor(img, result, CV_BGR2GRAY);
  else
    result=img.clone();

  int i, j;
  int count=0;

  int y=10000;
  int maxx=0;
  int minx=10000;
  int miny=0;
  int maxy=0;

  for(i=result.cols*0.40; i<result.cols*0.6; i=i+5)
    for(j=result.rows*0.5; j<result.rows-20; j++)
    {
      if(abs(*(result.data+j*result.step+i)-*(result.data+(j-1)*result.step+i))>10)
      {
        if(j<y)
          y=j;
        break;
      }
    }

  pots[0].x=result.cols/2;
  if(y+10<result.rows)
    pots[0].y=y+10;
  else
    pots[0].y=y;

#pragma omp parallel for
  for(j=pots[0].y; j<result.rows-20; j++)
    for(i=result.cols*0.20; i<result.cols*0.8; i=i+1)

    {
      if(abs(*(result.data+j*result.step+i)-*(result.data+j*result.step+i+1))>10)
      {
        if(i<minx)
        {
          minx=i+1;
          miny=j;
        }
        if(i>maxx)
        {
          maxx=i-1;
          maxy=j;
        }
      }
    }

  pots[1].x=minx;
  pots[1].y=miny;

  pots[2].x=maxx;
  pots[2].y=maxy;


  return pots;
}

Mat OnMorphology(Mat img, int etimes, int dtimes, int esize, int dsize, int flag)
{
  Mat result;

  if(img.channels()!=1)
    cvtColor(img, result, CV_BGR2GRAY);
  else
    result=img.clone();


  Mat delement, eelement,melement;

  /*int morph_elem = 0;
    int morph_size = 3;
    int morph_operator = 0;
    int const max_operator = 4;
    int const max_elem = 2;
    int const max_kernel_size = 21;

    melement=getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size));

    morphologyEx( result, result, 2, melement );//opening to remove holes*/

  int dilation_elem = 0;
  int dilation_size = dsize;
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  delement = getStructuringElement( dilation_type,
      Size( 2*dilation_size + 1, 2*dilation_size+1 ),
      Point( dilation_size, dilation_size ) );



  int erosion_elem = 0;
  int erosion_size = esize;
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
  eelement = getStructuringElement( erosion_type,
      Size( 2*erosion_size + 1, 2*erosion_size+1 ),
      Point( erosion_size, erosion_size ) );


  if(flag==0)//erode before dilate
  {
    /// Apply the erosion operation
    for(int i=0; i<etimes; i++)
      erode( result, result, eelement);
    /// Apply the dilation operation
    for(int i=0; i<dtimes; i++)
      dilate( result, result, delement );
  }
  else//dilate before erode
  {
    /// Apply the dilation operation
    for(int i=0; i<dtimes; i++)
      dilate( result, result, delement );
    /// Apply the erosion operation
    for(int i=0; i<etimes; i++)
      erode( result, result, eelement);
  }
  return result;
}

Point* OnPlantTop(Mat img, Rect rect)
{
  Point* tops=new Point[4];//remember to delete after using
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
}

Mat CompareImagePixels(Mat img1, Mat img2)
{
  //extract plant pixels from image
  //img1 is the full image, img2 is the image without leaves
  Mat result;
  cvtColor(img1, img1, CV_BGR2GRAY);
  cvtColor(img2, img2, CV_BGR2GRAY);
  //adaptiveThreshold(img1, img1, 255, CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,75,10);
  result=img1.clone();

  int i, j;

#pragma omp parallel for
  for(i=0; i<img1.cols; i++)
    for(j=0; j<img1.rows; j++)
    {
      if(*(img2.data+j*img2.step+i)>254 && (*(img1.data+j*img1.step+i)-*(img2.data+j*img2.step+i))<-100)
        *(result.data+j*result.step+i)=255;
      else
        *(result.data+j*result.step+i)=0;
    }

  return result;
}

Mat* DeconvolutionMat(Mat img, int m_flag)
{
  double leng, A, V, C;
  double log255=log(255.0);
  int i,j;
  double* MODx = new double[3];
  double* MODy = new double[3];
  double* MODz = new double[3];
  double* cosx = new double[3];
  double* cosy = new double[3];
  double* cosz  = new double[3];
  double* len = new double[3];
  double* q = new double[9];
  int** rLUT = new int*[3];
  int** gLUT = new int*[3];
  int** bLUT = new int*[3];
  for(i=0;i<3;i++)
  {
    rLUT[i] = new int[256];
    gLUT[i] = new int[256];
    bLUT[i] = new int[256];
  }
  int width=img.cols;
  int height=img.rows;

  Mat* outputimages;
  outputimages=new Mat[3];
  for(i=0; i<3; i++)
  {
    outputimages[i]=img.clone();
  }

  if (m_flag==0)
  {
    // GL Haem matrix
    MODx[0]= 0.644211; //0.650;
    MODy[0]= 0.716556; //0.704;
    MODz[0]= 0.266844; //0.286;
    // GL Eos matrix
    MODx[1]= 0.092789; //0.072;
    MODy[1]= 0.954111; //0.990;
    MODz[1]= 0.283111; //0.105;
    // Zero matrix
    MODx[2]= 0.0;
    MODy[2]= 0.0;
    MODz[2]= 0.0;
  }

  if (m_flag==1)
  {
    // 3,3-diamino-benzidine tetrahydrochloride
    // Haem matrix
    MODx[0]= 0.650;
    MODy[0]= 0.704;
    MODz[0]= 0.286;
    // DAB matrix
    MODx[1]= 0.268;
    MODy[1]= 0.570;
    MODz[1]= 0.776;
    // Zero matrix
    MODx[2]= 0.0;
    MODy[2]= 0.0;
    MODz[2]= 0.0;
  }

  if (m_flag==2)
  {
    //fast red
    MODx[0]= 0.21393921;
    MODy[0]= 0.85112669;
    MODz[0]= 0.47794022;
    // fast blue
    MODx[1]= 0.74890292;
    MODy[1]= 0.60624161;
    MODz[1]= 0.26731082;
    // dab
    MODx[2]= 0.268;
    MODy[2]= 0.570;
    MODz[2]= 0.776;
  }

  if (m_flag==3)
  {
    // MG matrix (GL)
    MODx[0]= 0.98003;
    MODy[0]= 0.144316;
    MODz[0]= 0.133146;
    // DAB matrix
    MODx[1]= 0.268;
    MODy[1]= 0.570;
    MODz[1]= 0.776;
    // Zero matrix
    MODx[2]= 0.0;
    MODy[2]= 0.0;
    MODz[2]= 0.0;
  }


  if (m_flag==4)
  {
    // Haem matrix
    MODx[0]= 0.650;
    MODy[0]= 0.704;
    MODz[0]= 0.286;
    // Eos matrix
    MODx[1]= 0.072;
    MODy[1]= 0.990;
    MODz[1]= 0.105;
    // DAB matrix
    MODx[2]= 0.268;
    MODy[2]= 0.570;
    MODz[2]= 0.776;
  }


  if (m_flag==5)
  {
    // 3-amino-9-ethylcarbazole
    // Haem matrix
    MODx[0]= 0.650;
    MODy[0]= 0.704;
    MODz[0]= 0.286;
    // AEC matrix
    MODx[1]= 0.2743;
    MODy[1]= 0.6796;
    MODz[1]= 0.6803;
    // Zero matrix
    MODx[2]= 0.0;
    MODy[2]= 0.0;
    MODz[2]= 0.0;
  }


  if (m_flag==6)
  {
    //Azocarmine and Aniline Blue (AZAN)
    // GL Blue matrix
    MODx[0]= .853033;
    MODy[0]= .508733;
    MODz[0]= .112656;
    // GL Red matrix
    MODx[1]= 0.070933;
    MODy[1]= 0.977311;
    MODz[1]= 0.198067;
    // Orange matrix (not set yet, currently zero)
    MODx[2]= 0.0;
    MODy[2]= 0.0;
    MODz[2]= 0.0;
  }


  if (m_flag==7)
  {
    // GL Alcian Blue matrix
    MODx[0]= 0.874622;
    MODy[0]= 0.457711;
    MODz[0]= 0.158256;
    // GL Haematox after PAS matrix
    MODx[1]= 0.552556;
    MODy[1]= 0.7544;
    MODz[1]= 0.353744;
    // Zero matrix
    MODx[2]= 0.0;
    MODy[2]= 0.0;
    MODz[2]= 0.0;
  }


  if (m_flag==8)
  {
    // GL Haem matrix
    MODx[0]= 0.644211; //0.650;
    MODy[0]= 0.716556; //0.704;
    MODz[0]= 0.266844; //0.286;
    // GL PAS matrix
    MODx[1]= 0.175411;
    MODy[1]= 0.972178;
    MODz[1]= 0.154589;
    // Zero matrix
    MODx[2]= 0.0;
    MODy[2]= 0.0;
    MODz[2]= 0.0;
  }


  if (m_flag==9)
  {
    //R
    MODx[0]= 0.0;
    MODy[0]= 1.0;
    MODz[0]= 1.0;
    //G
    MODx[1]= 1.0;
    MODy[1]= 0.0;
    MODz[1]= 1.0;
    //B
    MODx[2]= 1.0;
    MODy[2]= 1.0;
    MODz[2]= 0.0;
  }


  if (m_flag==10)
  {
    //C
    MODx[0]= 1.0;
    MODy[0]= 0.0;
    MODz[0]= 0.0;
    //M
    MODx[1]= 0.0;
    MODy[1]= 1.0;
    MODz[1]= 0.0;
    //Y
    MODx[2]= 0.0;
    MODy[2]= 0.0;
    MODz[2]= 1.0;
  }

  if (m_flag==11)
  {
    // MG matrix (GL)
    MODx[0]= 0.98003;
    MODy[0]= 0.144316;
    MODz[0]= 0.133146;
    // DAB matrix
    MODx[1]= 0.268;
    MODy[1]= 0.570;
    MODz[1]= 0.776;
    // Zero matrix
    MODx[2]= 0.99;
    MODy[2]= 0.0;
    MODz[2]= 0.99;
  }


  // start
  for (i=0; i<3; i++)
  {
    //normalise vector length
    cosx[i]=cosy[i]=cosz[i]=0.0;
    len[i]=sqrt(MODx[i]*MODx[i] + MODy[i]*MODy[i] + MODz[i]*MODz[i]);
    if (len[i] != 0.0)
    {
      cosx[i]= MODx[i]/len[i];
      cosy[i]= MODy[i]/len[i];
      cosz[i]= MODz[i]/len[i];
    }
  }


  // translation matrix
  if (cosx[1]==0.0)
  { //2nd colour is unspecified
    if (cosy[1]==0.0)
    {
      if (cosz[1]==0.0)
      {
        cosx[1]=cosz[0];
        cosy[1]=cosx[0];
        cosz[1]=cosy[0];
      }
    }
  }

  if (cosx[2]==0.0)
  { // 3rd colour is unspecified
    if (cosy[2]==0.0)
    {
      if (cosz[2]==0.0)
      {
        if ((cosx[0]*cosx[0] + cosx[1]*cosx[1])> 1)
        {
          //MessageBox("Colour [3] has a negative R component.");
          cosx[2]=0.0;
        }
        else 
        {
          cosx[2]=sqrt(1.0-(cosx[0]*cosx[0])-(cosx[1]*cosx[1]));
        }

        if ((cosy[0]*cosy[0] + cosy[1]*cosy[1])> 1)
        {
          //MessageBox("Colour [3] has a negative G component.");
          cosy[2]=0.0;
        }
        else 
        {
          cosy[2]=sqrt(1.0-(cosy[0]*cosy[0])-(cosy[1]*cosy[1]));
        }

        if ((cosz[0]*cosz[0] + cosz[1]*cosz[1])> 1)
        {
          //MessageBox("Colour [3] has a negative B component.");
          cosz[2]=0.0;
        }
        else 
        {
          cosz[2]=sqrt(1.0-(cosz[0]*cosz[0])-(cosz[1]*cosz[1]));
        }
      }
    }
  }

  leng=sqrt(cosx[2]*cosx[2] + cosy[2]*cosy[2] + cosz[2]*cosz[2]);

  cosx[2]= cosx[2]/leng;
  cosy[2]= cosy[2]/leng;
  cosz[2]= cosz[2]/leng;

  //matrix inversion
  A = cosy[1] - cosx[1] * cosy[0] / cosx[0];
  V = cosz[1] - cosx[1] * cosz[0] / cosx[0];
  C = cosz[2] - cosy[2] * V/A + cosx[2] * (V/A * cosy[0] / cosx[0] - cosz[0] / cosx[0]);
  q[2] = (-cosx[2] / cosx[0] - cosx[2] / A * cosx[1] / cosx[0] * cosy[0] / cosx[0] + cosy[2] / A * cosx[1] / cosx[0]) / C;
  q[1] = -q[2] * V / A - cosx[1] / (cosx[0] * A);
  q[0] = 1.0 / cosx[0] - q[1] * cosy[0] / cosx[0] - q[2] * cosz[0] / cosx[0];
  q[5] = (-cosy[2] / A + cosx[2] / A * cosy[0] / cosx[0]) / C;
  q[4] = -q[5] * V / A + 1.0 / A;
  q[3] = -q[4] * cosy[0] / cosx[0] - q[5] * cosz[0] / cosx[0];
  q[8] = 1.0 / C;
  q[7] = -q[8] * V / A;
  q[6] = -q[7] * cosy[0] / cosx[0] - q[8] * cosz[0] / cosx[0];


  // initialize 3 output colour stacks
  for (i=0; i<3; i++)
  {
    for (j=0; j<256; j++) 
    { //LUT[1]
      //if (cosx[i] < 0)
      //	rLUT[255-j]=(byte)(255.0 + (double)j * cosx[i]);
      //else
      rLUT[i][255-j]=int(255.0 - double(j) * cosx[i]);

      //if (cosy[i] < 0)
      //	gLUT[255-j]=(byte)(255.0 + (double)j * cosy[i]);
      //else
      gLUT[i][255-j]=int(255.0 - double(j) * cosy[i]);

      //if (cosz[i] < 0)
      //	bLUT[255-j]=(byte)(255.0 + (double)j * cosz[i]);
      ///else
      bLUT[i][255-j]=int(255.0 - double(j) * cosz[i]);
    }
  }

  // translate ------------------

  int imagesize = width * height;

#pragma omp parallel for
  for (j=0;j<imagesize;j++)
  {
    // log transform the RGB data
    int R = *((uchar*)(img.data)+j*3+2);
    int G = *((uchar*)(img.data)+j*3+1);
    int B = *((uchar*)(img.data)+j*3);

    double Rlog = -((255.0*log((double(R)+1)/255.0))/log255);
    double Glog = -((255.0*log((double(G)+1)/255.0))/log255);
    double Blog = -((255.0*log((double(B)+1)/255.0))/log255);

    for (i=0; i<3; i++)
    {
      // rescale to match original paper values
      double Rscaled = Rlog * q[i*3];
      double Gscaled = Glog * q[i*3+1];
      double Bscaled = Blog * q[i*3+2];
      double output = exp(-((Rscaled + Gscaled + Bscaled) - 255.0) * log255 / 255.0);

      if(output>255) 
        output=255;

      *(outputimages[i].data+j*3+2)=char(*(rLUT[i]+int(floor(output+.5))));
      *(outputimages[i].data+j*3+1)=char(*(gLUT[i]+int(floor(output+.5))));
      *(outputimages[i].data+j*3)=char(*(bLUT[i]+int(floor(output+.5))));
    }

  }

  return outputimages;
}

Mat RemoveFrame(Mat temp, Mat source)
{
  //temp is the template holding the information where frame is. source is the original input image
  Mat output;
  output=source.clone();

  int i, j;
  if(source.channels()==3)
  {
    for(i=0; i<temp.cols; i++)
      for(j=0; j<temp.rows; j++)
      {
        if(*(temp.data+j*temp.step+i)>250)
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
  }

  if(source.channels()==1)
  {
    for(i=0; i<temp.cols; i++)
      for(j=0; j<temp.rows; j++)
      {
        if(*(temp.data+j*temp.step+i)>250)
        {
          *(output.data+i*source.channels()+j*source.step)=*(source.data+i*source.channels()+j*source.step);
        }
        else
        {
          *(output.data+i*source.channels()+j*source.step)=0;
        }
      }
  }

  return output;
}

Rect OnFindCarSide(Mat img, int etimes , int dtimes , int esize , int dsize , int thres , int flag)
{
  Mat conHull;

  if(img.channels()!=1)
    cvtColor(img, conHull, CV_BGR2GRAY);
  else
    conHull=img.clone();

  conHull=OnMorphology(conHull, etimes, dtimes, esize, dsize, flag);// dilate and erode on frame and pot to remove small areas

  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  vector<Point> side;

  Mat threshold_output;
  threshold( conHull, threshold_output, thres, 255, THRESH_BINARY );
  findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );



  /*Rect rect;
    for( int i = 0; i< contours.size(); i++ )
    {
    if(contourArea(contours[i], false)>20)
    {
    Rect trect;
    trect=boundingRect(contours[i]);

    if(trect.width<trect.height*1.5 && trect.height<trect.width*1.5)//contour width is less than the 1.5 times of height
    if(trect.x>conHull.cols*0.25 && trect.x<conHull.cols*0.7 && trect.y>conHull.rows*0.3 && trect.width<conHull.cols*0.4 && trect.height<conHull.rows*0.3)//if the rect value is reasonable
    {
    for(int j=0; j<contours[i].size(); j++)
    side.push_back(contours[i][j]);
    }

    }
    }*///MS, MS application

  Rect rect;
  for( int i = 0; i< contours.size(); i++ )
  {
    if(contours[i].size()>20)//100 is a bit of big
    {
      Rect trect;
      trect=boundingRect(contours[i]);
      if(trect.x>conHull.cols*0.25 && trect.x<conHull.cols*0.7 && trect.y>conHull.rows*0.3 && trect.width<conHull.cols*0.4 && trect.height<conHull.rows*0.3)//if the rect value is reasonable
      {
        for(int j=0; j<contours[i].size(); j++)
          side.push_back(contours[i][j]);
      }

    }
  }

  rect=boundingRect(side);
  return rect;
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
