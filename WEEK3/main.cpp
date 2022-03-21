#include<iostream>
#include<string>
#include<sstream>

using namespace std;

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/types_c.h>

using  namespace cv;

void drawLine(Mat src, Mat dst){
    vector<Vec4i>lines;
	HoughLinesP(src, lines, 1, CV_PI / 180, 50, 0, 50);
    for(auto x : lines){
        line( dst, Point(x[0],x[1]), Point(x[2], x[3]), Scalar(0,255,0), 2, LINE_AA);
    }
}
int main(int argc, const char** argv){
    Mat source = imread("../../highway.jpg");
    Mat gray = imread("../../highway.jpg", IMREAD_GRAYSCALE);
    Mat equ_dst;

    equalizeHist(gray,equ_dst);

    vector<int> blurKernel={5,7,1};

    for(auto x : blurKernel){
        medianBlur(equ_dst, equ_dst,x);
    }

    Mat threshold_1,lap;
    threshold(equ_dst, threshold_1, 245, 255, THRESH_BINARY);

    Mat kernelOptimal=getStructuringElement(MORPH_RECT,Size(5,5));

    for (int i=0;i<2;i++){
        morphologyEx(threshold_1,threshold_1,MORPH_CLOSE,kernelOptimal);
    }
        
    Laplacian(threshold_1,lap,CV_8U,3);
    drawLine(lap, source);
	imshow("Result", source);
    
    waitKey(0);
    system("pause");
    return 0;
}


