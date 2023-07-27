/**
 * @file watershed.cpp
 * @author luan yeming
 * @brief  新手教材(应用)
 * @version 0.1
 * @date 2023-07-25
 * @note https://github.com/martin90102/rookie_book
 * @copyright Copyright (c) 2023
 *
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

/**
 * @brief 分水岭算法
 *
 * @param image
 * @param dst
 */
void watershed_seg(Mat &image, Mat &dst)
{
    Mat gray, binary;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);
    imshow("gray", gray);

    // 将灰度图像转换为二值图像
    threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // 形态学操作
    Mat dist;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_OPEN, kernel);
    distanceTransform(binary, dist, DIST_L2, 3);
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);

    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    int max = *max_element(dist.begin<float>(), dist.end<float>());

    // 制作marker
    Mat markers;
    connectedComponents(dist_8u, markers, 8, CV_32S);
    markers = markers + 1; // 保证背景为0

    // 分水岭算法
    watershed(image, markers);
    // 显示结果
    Mat mark;
    markers.convertTo(mark, CV_8U);
    equalizeHist(mark, mark);
    imshow("output", mark);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "give correct arguments in format below!" << endl;
        cout << "Usage: " << argv[0] << " <image>" << endl;
        return -1;
    }
    string path = argv[1];
    Mat image = imread(path);
    if (image.empty())
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }
    Mat dst = image.clone();
    watershed_seg(image, dst);
    waitKey(0);
    return 0;
}