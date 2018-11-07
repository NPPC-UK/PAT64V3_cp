#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <regex>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>

#include "analysis.h"
#include "analysis_utils.h"
#include "format_string.h"
#include "clover_data.h"

using namespace cv;
using namespace std;

std::unique_ptr<plant_data> GetData(const char* filename)
{
  auto p_data = std::make_unique<clover_data>();

  string s, s1, s2;

  Mat img;
  img=imread(filename);

  if(!img.data) {
    throw std::invalid_argument(stringf("The image: '%s' contains no data.",
          filename));
  }
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

  tmp=OnMorphology(tmp, 1, 1, 1, 1, MorphOp::Close);
  tmp=OnMorphology(tmp, 2, 2, 1, 1, MorphOp::Open);

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

  if(tt.val[0]==0) {
    throw std::domain_error("Blue component of image is 0.");
  }

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
        p_data->pixelcount[pos] += area;
      }
    }
  }

  for(int i=0; i<drawing.cols; i++)
    for(int j=0; j<drawing.rows; j++)
    {
      if(drawing.at<Vec3b>(j, i)[1]==50 &&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[1]*0.95)&&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[0]*1.5)&& img.at<Vec3b>(j, i)[2]>50)
      {
        p_data->yellowcount[0]++;
        drawing.at<Vec3b>(j,i)[0]=0;
        drawing.at<Vec3b>(j,i)[1]=15;
        drawing.at<Vec3b>(j,i)[2]=255;
      }
      if(drawing.at<Vec3b>(j, i)[1]==255 &&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[1]*0.95)&&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[0]*1.5)&& img.at<Vec3b>(j, i)[2]>50)
      {
        p_data->yellowcount[1]++;
        drawing.at<Vec3b>(j,i)[0]=0;
        drawing.at<Vec3b>(j,i)[1]=15;
        drawing.at<Vec3b>(j,i)[2]=255;
      }
      if(drawing.at<Vec3b>(j, i)[1]==204 &&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[1]*0.95)&&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[0]*1.5)&& img.at<Vec3b>(j, i)[2]>50)
      {
        p_data->yellowcount[2]++;
        drawing.at<Vec3b>(j,i)[0]=0;
        drawing.at<Vec3b>(j,i)[1]=15;
        drawing.at<Vec3b>(j,i)[2]=255;
      }
      if(drawing.at<Vec3b>(j, i)[1]==225 &&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[1]*0.95)&&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[0]*1.5)&& img.at<Vec3b>(j, i)[2]>50)
      {
        p_data->yellowcount[3]++;
        drawing.at<Vec3b>(j,i)[0]=0;
        drawing.at<Vec3b>(j,i)[1]=15;
        drawing.at<Vec3b>(j,i)[2]=255;
      }
      if(drawing.at<Vec3b>(j, i)[1]==104 &&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[1]*0.95)&&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[0]*1.5)&& img.at<Vec3b>(j, i)[2]>50)
      {
        p_data->yellowcount[4]++;
        drawing.at<Vec3b>(j,i)[0]=0;
        drawing.at<Vec3b>(j,i)[1]=15;
        drawing.at<Vec3b>(j,i)[2]=255;
      }
      if(drawing.at<Vec3b>(j, i)[1]==205 &&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[1]*0.95)&&(img.at<Vec3b>(j, i)[2]>img.at<Vec3b>(j, i)[0]*1.5)&& img.at<Vec3b>(j, i)[2]>50)
      {
        p_data->yellowcount[5]++;
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
            p_data->length[0]=float(i-370-dx)*0.19;

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
            p_data->length[1]=float(5610+dx-i)*0.19;

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
            p_data->length[2]=float(i-370-dx)*0.19;

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
            p_data->length[3]=float(5610+dx-i)*0.19;

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
            p_data->length[4]=float(i-370-dx)*0.19;

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
            p_data->length[5]=float(5610+dx-i)*0.19;

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

  p_data->image = img0;

  //fprintf(fp, "%s,%0.f,%0.f,%0.f,%0.f,%0.f,%0.f,%s,%0.f,%0.f,%0.f,%0.f,%0.f,%0.f,%s,%0.f,%0.f,%0.f,%0.f,%0.f,%0.f\n", s1, length[0], pixelcount[0], yellowcount[0], length[1], pixelcount[1], yellowcount[1], s2, length[2], pixelcount[2], yellowcount[2], length[3], pixelcount[3], yellowcount[3],s3, length[4], pixelcount[4], yellowcount[4], length[5], pixelcount[5], yellowcount[5]);

  return p_data;
}
