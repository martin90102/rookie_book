/**
 * @file img_segment.cpp
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
 * @brief 区域生长算法
 *
 * @param image 输入图像
 * @param dst 输出图像
 */
void region_growing(Mat &image, Mat &dst)
{
    Mat gray, binary;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    imshow("gray", gray);
    // 进行高斯滤波
    GaussianBlur(gray, gray, Size(3, 3), 0, 0);
    // 随机生成一个种子坐标
    Point seed = Point(rand() % image.cols, rand() % image.rows);
    // 区域生长
    floodFill(gray, seed, Scalar(255, 255, 255), NULL, Scalar(10, 10, 10), Scalar(10, 10, 10), 4);
    imshow("binary", gray);
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
    region_growing(image, dst);
    waitKey(0);
    return 0;
}