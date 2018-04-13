#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <regex>
#include <string>

#include <opencv2/opencv.hpp>

#include "analysis.h"


using namespace cv;
using namespace std;

void GetData(const char* filename, const char* outputpath, string s1, string s2)
{
  FILE *fp;
  FILE *fp1;
  FILE *fp2;
  Mat* DeconvolutionMat(Mat img, int m_flag);
  Mat CompareImagePixels(Mat img1, Mat img2);
  Point* OnPlantTop(Mat img, Rect rect);
  Mat OnMorphology(Mat img, int etimes, int dtimes, int esize, int dsize, int flag);
  Rect OnFindCarSide(Mat img, int etimes , int dtimes , int esize , int dsize , int thres , int flag);
  Point* OnPotPoints(Mat img);
  Mat RestoreImgFromTemp(Mat temp, Mat source);
  int OnCountPixels(Mat img, int pottop, int planttop);
  void GetData(char* filename, char* outputpath, string s1, string s2);
  Mat RemoveFrame(Mat temp, Mat source);
  Mat FindPlantPixels(Mat img, double gthres, double gbthres);



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


    if(strcmp(s1.c_str(), "2015-04-27")==0)
    {
      rx=817;
      ry=2045;
      rw=360;
      rh=345;
      fprintf(fp2, "%d", rx);
      fprintf(fp2, "\n");
      fprintf(fp2, "%d", ry);
      fprintf(fp2, "\n");
      fprintf(fp2, "%d", rw);
      fprintf(fp2, "\n");
      fprintf(fp2, "%d", rh);
      fprintf(fp2, "\n");
      fclose(fp2);
    }

    if(strcmp(s1.c_str(), "2015-04-01")==0)
    {
      rx=740;
      ry=1880;
      rw=510;
      rh=500;
      fprintf(fp2, "%d", rx);
      fprintf(fp2, "\n");
      fprintf(fp2, "%d", ry);
      fprintf(fp2, "\n");
      fprintf(fp2, "%d", rw);
      fprintf(fp2, "\n");
      fprintf(fp2, "%d", rh);
      fprintf(fp2, "\n");
      fclose(fp2);
    }


    string sss;
    sss.assign(outputpath);
    sss=sss.substr(0, sss.rfind("W8-"));
    sss.append("\\carsize.txt");

    char buffer1[256]; 
    int k1[4], ki1=0;
    fstream outFile1;  
    outFile1.open(sss.c_str(),ios::in);  
    cout<<sss.c_str()<<"--- all file is as follows:---"<<endl;  
    while(!outFile1.eof())  
    {  
      outFile1.getline(buffer1,256,'\n');//getline(char *,int,char) 表示该行字符达到256个或遇到换行就结束  
      k1[ki1]=atoi(buffer1);
      cout<<k1[ki1]<<endl;  
      ki1=ki1+1;
    }  

    outFile1.close();  

    rectA.x=k1[0];;
    rectA.y=k1[1];
    rectA.width=k1[2];
    rectA.height=k1[3];

    Rect rectB;
    rectB.x = rectA.x- rectA.width*1.7;
    if(rectB.x<0)
      rectB.x=10;
    rectB.y = 10;//MS, 100
    rectB.width = rectA.width*4.65;
    if(rectB.x+rectB.width>conHull.cols)
      rectB.width=conHull.cols-2*rectB.x;
    rectB.height = rectA.y+float(rectA.height)*0.07-10;//calculate height from the pot top
    //rectB.height = rectA.y+rectA.height-rectA.width-rectB.y;//calculate height from the pot bottom

    Mat tn;
    tn=FindPlantPixels(img, 0.85, 1.3);//original 0.001, w2 is 0.02; 0.7, 0.02; VF1: 0.6, 0.000, w11 0.85, b*1.3<g

    cvtColor(tn, tn, CV_BGR2GRAY);

    threshold(tn, tn, 254, 255, CV_THRESH_BINARY_INV);
    //use adaptive thresholding to get bright pixels
    //cvtColor(img, img, CV_BGR2GRAY);
    //adaptiveThreshold(img, img, 255, CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY_INV,75,10);
    //img=img+tn;

    Mat out;
    //out=RemoveFrame(tmp, img);//we do not have to remove frame if we can find plant pixel better
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
      if(trect.x+trect.width/2>roi.cols*0.10 && trect.x+trect.width/2<roi.cols*0.9 && trect.y+trect.height/2>rectB.y+rectB.height*0.1)
        if(area>70)
        {
          Scalar color( 100, 100, 0 );
          drawContours( roi1, contours, idx, color, CV_FILLED, 8, hierarchy );
          drawContours( roi, contours, idx, color, CV_FILLED, 8, hierarchy );
          drawContours( roi2, contours, idx, color, CV_FILLED, 8, hierarchy );
          pixelcount=pixelcount+area;
        }
    }


    int planttopflag=0;
    Point* tops=new Point[4];
    tops=OnPlantTop(out, rectB);//find top pixel of the plant
    if(tops[0].y<0 ||tops[0].x<0)
    {
      tops[0].x=rectB.x+rectB.width/2;
      tops[0].y=rectB.y+rectB.height;
    }

    if(tops[0].y<img.rows*0.02)
      planttopflag=1;

    Point pend;
    Point tend;

    pend.x=image.cols*0.7;
    pend.y=rectB.y+rectB.height;

    tend.x=image.cols*0.7;
    tend.y=tops[0].y;

    line(image, Point(rectB.x+rectB.width/2, rectB.y+rectB.height), pend, Scalar(0,255,0), 2, 8, 0);
    line(image, tops[0], tend, Scalar(0,255,0), 2, 8, 0);

    line(image, Point(tend.x-10, tend.y), Point(pend.x-10, pend.y), Scalar(0,255,0), 2, 8, 0);

    int plant_height, pot_width;
    plant_height=rectB.y+rectB.height-tops[0].y;//plant height in pixel
    pot_width=rectA.width;//pot width in pixel
    double p_h=double(plant_height)*(250/double(pot_width));//plant height in mm
    leafArea=pixelcount*(250/double(pot_width))*(250/double(pot_width));

    char s[200];
    //sprintf_s(s, 200, "%5.fmm", p_h );
    sprintf(s, "%5.fmm", p_h );
    putText(image, s, Point(image.cols*0.7+5,tops[0].y+plant_height/2), 0, 2, Scalar(155,155,0), 3, 8,false);

    //sprintf_s(s, 200, "%5.fmm", leafArea );
    sprintf(s, "%5.fmm", leafArea );
    putText(image, s, Point(image.cols*0.7+5,tops[0].y+plant_height/2+70), 0, 2, Scalar(155,155,0), 3, 8,false);

    //sprintf_s(s, 200, "2");
    sprintf(s, "2");
    putText(image, s, Point(image.cols*0.7+303+8,tops[0].y+plant_height/2+45), 0, 1, Scalar(155,155,0), 3, 8,false);

    rectangle(image, rectB, Scalar(0, 0, 255), 1, 8, 0);
    rectangle(image, rectA, Scalar(0, 0, 255), 1, 8, 0);


    if(strcmp(s1.c_str(), "2015-04-27")==0)
    {
      fprintf(fp1, "%d", plant_height);
      fprintf(fp1, "\n");
      fprintf(fp1, "%d", pot_width);
      fprintf(fp1, "\n");
      fclose(fp1);
    }

    string ss;
    ss.assign(outputpath);
    ss=ss.substr(0, ss.rfind("W8-"));
    //ss.append("\\");
    ss.append(s2);
    ss.append(".txt");

    char buffer[256]; 
    int k[3], ki=0;
    fstream outFile;  
    outFile.open(ss.c_str(),ios::in);  
    cout<<ss.c_str()<<"--- all file is as follows:---"<<endl;  
    while(!outFile.eof())  
    {  
      outFile.getline(buffer,256,'\n');//getline(char *,int,char) 表示该行字符达到256个或遇到换行就结束  
      k[ki]=atoi(buffer);
      cout<<k[ki]<<endl;  
      ki=ki+1;
    }  
    outFile.close();  

    int t00=0;
    int t00y=0;
    int t20=0;
    int t20y=0;
    int t40=0;
    int t40y=0;
    int t60=0;
    int t60y=0;

    int finalheight=k[0];
    int finalpotwidth=k[1];
    int potsidestartingpoint=rectA.y+float(rectA.height)*0.07;

    finalheight=float(finalheight)*float(pot_width)/(float(finalpotwidth)+0.0000001);

    int yellowcount=0;
#pragma omp parallel for
    for(int l=rectB.x; l<rectB.x+rectB.width; l++)
      for(int k=(potsidestartingpoint-finalheight<0?0:potsidestartingpoint-finalheight); k<potsidestartingpoint-finalheight*0.75; k++)
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

    line(image, Point(500, potsidestartingpoint-finalheight), Point(1954, potsidestartingpoint-finalheight), Scalar(0,255,255), 2, 8, 0);

#pragma omp parallel for
    for(int l=rectB.x; l<rectB.x+rectB.width; l++)
      for(int k=potsidestartingpoint-finalheight*0.75; k<potsidestartingpoint-finalheight*0.5; k++)
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

    line(image, Point(500, potsidestartingpoint-finalheight*0.75), Point(1954, potsidestartingpoint-finalheight*0.75), Scalar(0,255,255), 2, 8, 0);

#pragma omp parallel for
    for(int l=rectB.x; l<rectB.x+rectB.width; l++)
      for(int k=potsidestartingpoint-finalheight*0.5; k<potsidestartingpoint-finalheight*0.25; k++)
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

    line(image, Point(500, potsidestartingpoint-finalheight*0.5), Point(1954, potsidestartingpoint-finalheight*0.5), Scalar(0,255,255), 2, 8, 0);

#pragma omp parallel for
    for(int l=rectB.x; l<rectB.x+rectB.width; l++)
      for(int k=potsidestartingpoint-finalheight*0.25; k<potsidestartingpoint; k++)
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

    line(image, Point(500, potsidestartingpoint-finalheight*0.25), Point(1954, potsidestartingpoint-finalheight*0.25), Scalar(0,255,255), 2, 8, 0);

    fprintf(fp, "%d,%d,%0.f,%0.f,%0.f,%d,", plant_height, pot_width, p_h, pixelcount, leafArea, yellowcount);
    fprintf(fp, "%d, %d, %d, %d, %d, %d, %d, %d", t00, t00y, t20, t20y, t40, t40y, t60, t60y);

    imwrite(outputpath, image);

    delete[] pImg0;
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
