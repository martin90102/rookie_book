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
 * @brief 二值分割算法（大律法为例）
 *
 * @param image 输入图像
 * @param dst 输出图像
 *
 */
void threshold_demo(Mat &image, Mat &dst)
{
    Mat gray, binary;
    // 转换为灰度图
    cvtColor(image, gray, COLOR_BGR2GRAY);
    // 显示原图
    imshow("gray", gray);
    // 大律法
    threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
    // 显示操作之后的图像
    imshow("binary", binary);
    dst = binary;
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
    threshold_demo(image, dst);
    waitKey(0);
    return 0;
}