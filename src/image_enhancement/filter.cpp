/**
 * @file filter.cpp
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

void filters(Mat &image, Mat &dst)
{
    cvtColor(image, image, COLOR_BGR2GRAY);
    imshow("origin", image);

    // 添加高斯噪声的结果
    // Mat gauss_, pepper_;
    // auto GaussianNoise = [](Mat &image, Mat &outputImage, double mean, double stddev)
    // {
    //     cv::Mat noise = cv::Mat(image.size(), image.type());
    //     cv::randn(noise, mean, stddev); // create white noise
    //     outputImage = image + noise;
    // };
    // GaussianNoise(image, gauss_, 10, 10);
    // imshow("GaussianNoise", gauss_);

    // // 添加椒盐噪声的结果
    // auto SaltPepperNoise = [](Mat &image, Mat &outputImage, double pa, double pb)
    // {
    //     cv::Mat noise = cv::Mat(image.size(), image.type());
    //     randu(noise, 0, 255);
    //     outputImage = image.clone();

    //     for (int i = 0; i < image.rows; i++)
    //     {
    //         for (int j = 0; j < image.cols; j++)
    //         {
    //             if (noise.at<uchar>(i, j) < pa * 255)
    //             {
    //                 outputImage.at<uchar>(i, j) = 0;
    //             }
    //             else if (noise.at<uchar>(i, j) > pb * 255)
    //             {
    //                 outputImage.at<uchar>(i, j) = 255;
    //             }
    //         }
    //     }
    // };
    // SaltPepperNoise(image, pepper_, 0.01, 0.99);
    // // normalize(pepper_, pepper_, 0, 255, NORM_MINMAX, CV_8UC1);
    // imshow("SaltPepperNoise", pepper_);

    // 高斯滤波
    GaussianBlur(image, dst, Size(7, 7), 0, 0);
    imshow("GaussianBlur", dst);
    // 中值滤波
    medianBlur(image, dst, 7);
    imshow("medianBlur", dst);

    // 双边滤波
    bilateralFilter(image, dst, 9, 10, 10);
    imshow("bilateralFilter", dst);

    // 引导滤波(演示保边效果)
    auto Guided_filter = [](Mat &image, int win_size)
    {
        // 初始化滤波窗口
        Mat I = image.clone();
        Mat p = image.clone();

        Mat mean_I, mean_p, mean_Ip, mean_II, var_I, a, b, mean_a, mean_b, q;
        // 计算均值
        boxFilter(I, mean_I, CV_64F, Size(win_size, win_size));
        boxFilter(p, mean_p, CV_64F, Size(win_size, win_size));
        boxFilter(I.mul(p), mean_Ip, CV_64F, Size(win_size, win_size));
        boxFilter(I.mul(I), mean_II, CV_64F, Size(win_size, win_size));
        // 计算方差
        var_I = mean_II - mean_I.mul(mean_I);
        // 计算a,b
        a = (mean_Ip - mean_I.mul(mean_p)) / (var_I + 0.001);
        b = mean_p - a.mul(mean_I);
        // 计算均值
        boxFilter(a, mean_a, CV_64F, Size(win_size, win_size));
        boxFilter(b, mean_b, CV_64F, Size(win_size, win_size));
        // 计算q
        q = mean_a.mul(I);
        q += mean_b;
        return q;
    };
    image.convertTo(image, CV_64F);
    Mat guided = Guided_filter(image, 7);
    normalize(guided, guided, 0, 255, NORM_MINMAX, CV_8UC1);
    imshow("Guided_filter", guided);
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
    filters(image, dst);
    waitKey(0);
    return 0;
}