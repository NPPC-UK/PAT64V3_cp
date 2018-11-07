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

std::array<cv::Point, 3> FindLTPotLimits(const cv::Mat img) {
  std::array<cv::Point, 3> pots; 
  cv::Mat result;

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

  /* This for loop sets the variable y to the value of the lowest row
   * where the intensity changes by more than 10 from one row to the
   * next for the first time in that column.
   *
   * In this image '.' represents intensity 0, '#' represents intensity
   * 20.  Rows are labeled:
   *
   * 0: .....
   * 1: ...#.
   * 2: .###.
   * 3: ####.
   * 4: #####
   * 5: #####
   * 6: #####
   *
   * In this case the variable y is set to 4.  The lowest row in which any 
   * column changes intensity by more than 10.
   *
   * y seems to represent a boundary on where pot pixels are likely to be found
   */
  // In the central 20% of columns in steps of 5
  for(i=result.cols*0.40; i<result.cols*0.6; i=i+5)
    // In the bottom half of the image - 20
    for(j=result.rows*0.5; j<result.rows-20; j++)
    {
      // Difference between This pixel and  the same pixel 
      // in the previous row > 10
      if(abs(*(result.data+j*result.step+i)-*(result.data+(j-1)*result.step+i))>10)
      {

        if(j<y) {
          y=j;
        }

        break;
      }
    }

  /* 
   * The point pots[0] is a point in the center column of the image, at the
   * position of the variable y (see above) with some buffer if possible
   */
  pots[0].x=result.cols/2;
  if(y+10<result.rows)
    pots[0].y=y+10;
  else
    pots[0].y=y;

#pragma omp parallel for
  /*
   * Sets minx to the left most column where there is a sudden change in
   * intensity, for any row below y (see above).  miny is set to the 
   * corresponding row.
   *
   * maxx/maxy are analogous
   */
  // For every row in the image from y to nearly the bottom
  for(j=pots[0].y; j<result.rows-20; j++)
    // For every central column (The edges are likely to contain fluff)
    for(i=result.cols*0.20; i<result.cols*0.8; i++)

    {
      // If Intensity changes from this column to the next 
      // (in the current row)
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


  // Returning std::array by value.  Should use RVO.
  return pots;
}

cv::Mat OnMorphology(const cv::Mat img, 
                     int etimes, 
                     int dtimes, 
                     int esize, 
                     int dsize, 
                     int flag) {
  if (flag == 0)
    return OnMorphology(img, etimes, dtimes, esize, dsize, MorphOp::Open);
  else
    return OnMorphology(img, etimes, dtimes, esize, dsize, MorphOp::Close);
}

cv::Mat OnMorphology(const cv::Mat img, 
                     int etimes, 
                     int dtimes, 
                     int esize, 
                     int dsize, 
                     MorphOp op) {
  cv::Mat result;

  if(img.channels()!=1)
    cvtColor(img, result, CV_BGR2GRAY);
  else
    result=img.clone();


  cv::Mat delement, eelement,melement;

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
  if( dilation_elem == 0 ){ dilation_type = cv::MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = cv::MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = cv::MORPH_ELLIPSE; }

  delement = cv::getStructuringElement( dilation_type,
      cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
      cv::Point( dilation_size, dilation_size ) );



  int erosion_elem = 0;
  int erosion_size = esize;
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = cv::MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = cv::MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = cv::MORPH_ELLIPSE; }
  eelement = cv::getStructuringElement( erosion_type,
      cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
      cv::Point( erosion_size, erosion_size ) );


  if(op == MorphOp::Open)//erode before dilate
  {
    /// Apply the erosion operation
    for(int i=0; i<etimes; i++)
      erode( result, result, eelement);
    /// Apply the dilation operation
    for(int i=0; i<dtimes; i++)
      dilate( result, result, delement );
  }
  else if (op == MorphOp::Close)//dilate before erode
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


cv::Mat CompareImagePixels(cv::Mat img1, cv::Mat img2) {
  //extract plant pixels from image
  //img1 is the full image, img2 is the image without leaves
  cvtColor(img1, img1, CV_BGR2GRAY);
  cvtColor(img2, img2, CV_BGR2GRAY);
  //adaptiveThreshold(img1, img1, 255, CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,75,10);
  auto result=img1.clone();

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

cv::Mat* DeconvolutionMat(cv::Mat img, int m_flag)
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

  auto outputimages=new cv::Mat[3];
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

  /* 
   * TODO: Remove debugging
   */
  cv::imwrite("./debug_deconvolution0.png", outputimages[0]);
  cv::imwrite("./debug_deconvolution1.png", outputimages[1]);
  cv::imwrite("./debug_deconvolution2.png", outputimages[2]);
  return outputimages;
}

cv::Rect OnFindCarSide(cv::Mat img,
                       int etimes,
                       int dtimes,
                       int esize,
                       int dsize,
                       int thres,
                       int flag)
{
  cv::Mat conHull;

  if(img.channels()!=1)
    cvtColor(img, conHull, CV_BGR2GRAY);
  else
    conHull=img.clone();

  if (flag == 0)
    conHull=OnMorphology(conHull, etimes, dtimes, esize, dsize, MorphOp::Open);// dilate and erode on frame and pot to remove small areas
  else
    conHull=OnMorphology(conHull, etimes, dtimes, esize, dsize, MorphOp::Close);// dilate and erode on frame and pot to remove small areas


  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  std::vector<cv::Point> side;
  cv::Mat threshold_output;

  threshold(conHull, 
            threshold_output, 
            thres, 
            255, 
            cv::THRESH_BINARY);
  findContours(threshold_output, 
               contours, 
               hierarchy, 
               CV_RETR_TREE, 
               CV_CHAIN_APPROX_SIMPLE, 
               cv::Point(0, 0));

  cv::Rect rect;
  // Iterate over contours
  for( int i = 0; i< contours.size(); i++ )
  {
    if(contours[i].size()>20)//100 is a bit of big
    {
      cv::Rect trect;
      trect=boundingRect(contours[i]);
      // CHeck that the contour has sensible size and position in the image. If
      // yes, this represents the side of the pot
      if(trect.x>conHull.cols*0.25 && 
         trect.x<conHull.cols*0.7 && 
         trect.y>conHull.rows*0.3 && 
         trect.width<conHull.cols*0.4 && 
         trect.height<conHull.rows*0.3)
      {
        for(int j=0; j<contours[i].size(); j++)
          side.push_back(contours[i][j]);
      }
    }
  }

  rect=boundingRect(side);
  return rect;
}

cv::Mat RemoveFrame(cv::Mat mask, cv::Mat source)
{
  auto output = source.clone();

  int i, j;
  if(source.channels()==3)
  {
    for(i=0; i<mask.cols; i++)
      for(j=0; j<mask.rows; j++)
      {
        if(*(mask.data+j*mask.step+i)>250)
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
    for(i=0; i<mask.cols; i++)
      for(j=0; j<mask.rows; j++)
      {
        if(*(mask.data+j*mask.step+i)>250)
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
