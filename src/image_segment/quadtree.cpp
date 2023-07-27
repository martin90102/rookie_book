/**
 * @file quadtree.cpp
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
 * @brief 设R代表整个正方形图像区域，P代表逻辑谓词。基本分裂合并算法步骤如下：
            (1)对任一个区域，如果H(Ri)=FALSE就将其分裂成不重叠的四等份；
            (2)对相邻的两个区域Ri和Rj，它们也可以大小不同（即不在同一层），如果条件H(Ri∪Rj)=TRUE满足，就将它们合并起来。
            (3)如果进一步的分裂或合并都不可能，则结束。
 * @param image
 * @param dst
 */
void quadtree_seg(Mat &image, Mat &dst)
{
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    imshow("gray", gray);
    // 定义过程
    function<void(int, int, int, int)> quadtree = [&](int x, int y, int width, int height)
    {
        Mat region = gray(Rect(x, y, width, height));
        Mat mean, stddev;
        meanStdDev(region, mean, stddev);
        if (stddev.at<double>(0, 0) > 10)
        {
            // 如果不满足H(Ri)就分裂成四个子区域
            int half_width = width / 2;
            int half_height = height / 2;
            quadtree(x, y, half_width, half_height);
            quadtree(x + half_width, y, half_width, half_height);
            quadtree(x, y + half_height, half_width, half_height);
            quadtree(x + half_width, y + half_height, half_width, half_height);
        }
        else
        {
            // 否则就将该区域填充为white
            rectangle(dst, Rect(x, y, width - 1, height - 1), Scalar(255, 255, 255), -1);
        }
    };

    quadtree(0, 0, gray.cols, gray.rows);
    imshow("output", dst);
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
    quadtree_seg(image, dst);
    waitKey(0);
    return 0;
}