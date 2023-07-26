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
 * @brief 边缘检测分割
 *
 * @param image
 * @param dst
 */
void edge_seg(Mat &image, Mat &dst)
{
    Mat gray, binary;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    imshow("gray", gray);
    // 梯度算法处理
    Mat dst0;
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Sobel(gray, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT); //  求X方向梯度
    Sobel(gray, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT); //  求Y方向梯度
    convertScaleAbs(grad_x, abs_grad_x);                        // 转换为CV_8U
    convertScaleAbs(grad_y, abs_grad_y);                        // 转换为CV_8U
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst0);     // 图像融合
    imshow("grad", dst0);

    // Roberts算子
    Mat dst1, dst2;
    Mat roberts_cross_x = (Mat_<char>(2, 2) << 0, 0, -1, 1);
    Mat roberts_cross_y = (Mat_<char>(2, 2) << 0, -1, 0, 1);
    filter2D(gray, dst1, -1, roberts_cross_x);
    filter2D(gray, dst2, -1, roberts_cross_y);
    imshow("roberts_x", dst1);
    imshow("roberts_y", dst2);

    // Sobel算子
    Mat dst3, dst4;
    Sobel(gray, dst3, -1, 1, 0);
    Sobel(gray, dst4, -1, 0, 1);
    imshow("sobel_x", dst3);
    imshow("sobel_y", dst4);

    // Prewitt算子
    Mat dst5, dst6;
    Mat prewitt_x = (Mat_<char>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    Mat prewitt_y = (Mat_<char>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
    filter2D(gray, dst5, -1, prewitt_x);
    filter2D(gray, dst6, -1, prewitt_y);
    imshow("prewitt_x", dst5);
    imshow("prewitt_y", dst6);

    // Laplace算子
    Mat dst7;
    Laplacian(gray, dst7, -1);
    imshow("laplace", dst7);

    // Kirsh算子
    Mat dst8, dst9;
    Mat kirsh_x = (Mat_<char>(3, 3) << 5, 5, 5, -3, 0, -3, -3, -3, -3);
    Mat kirsh_y = (Mat_<char>(3, 3) << 5, -3, -3, 5, 0, -3, 5, -3, -3);
    filter2D(gray, dst8, -1, kirsh_x);
    filter2D(gray, dst9, -1, kirsh_y);
    imshow("kirsh_x", dst8);
    imshow("kirsh_y", dst9);

    // Canny算子
    Canny(gray, dst, 100, 200);
    imshow("canny", dst);
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
    edge_seg(image, dst);
    waitKey(0);
    return 0;
}