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
 * @brief 小波分割（拉普拉斯小波举例）
 *
 * @param image
 * @param dst
 */
void wavelet_seg(Mat &image, Mat &dst)
{
    // 拉普拉斯分解
    auto laplace_decompose = [&](Mat &src, int s, Mat &wave)
    {
        Mat full_src(src.rows, src.cols, CV_32FC1);
        Mat dst = src.clone();
        dst.convertTo(dst, CV_32FC1);
        for (int m = 0; m < s; m++)
        {
            dst.convertTo(dst, CV_32FC1);
            Mat wave_src(dst.rows, dst.cols, CV_32FC1);
            // 列变换，detail=mean-original
            for (int i = 0; i < wave_src.rows; i++)
            {
                for (int j = 0; j < wave_src.cols / 2; j++)
                {
                    wave_src.at<float>(i, j) = (dst.at<float>(i, 2 * j) + dst.at<float>(i, 2 * j + 1)) / 2;
                    wave_src.at<float>(i, j + wave_src.cols / 2) = wave_src.at<float>(i, j) - dst.at<float>(i, 2 * j);
                }
            }
            Mat temp = wave_src.clone();
            for (int i = 0; i < wave_src.rows / 2; i++)
            {
                for (int j = 0; j < wave_src.cols / 2; j++)
                {
                    wave_src.at<float>(i, j) = (temp.at<float>(2 * i, j) + temp.at<float>(2 * i + 1, j)) / 2;
                    wave_src.at<float>(i + wave_src.rows / 2, j) = wave_src.at<float>(i, j) - temp.at<float>(2 * i, j);
                }
            }
            dst.release();
            dst = wave_src(Rect(0, 0, wave_src.cols / 2, wave_src.rows / 2));
            wave_src.copyTo(full_src(Rect(0, 0, wave_src.cols, wave_src.rows)));
        }
        wave = full_src.clone();
    };

    // 小波重构
    auto wave_recover = [&](Mat &full_scale, Mat &original, int level)
    {
        for (int m = 0; m < level; m++)
        {
            Mat temp = full_scale(Rect(0, 0, full_scale.cols / pow(2, level - m - 1),
                                       full_scale.rows / pow(2, level - m - 1)));
            Mat recover_src(temp.rows, temp.cols, CV_32FC1);
            for (int i = 0; i < recover_src.rows; i++)
            {
                for (int j = 0; j < recover_src.cols / 2; j++)
                {
                    if (i % 2 == 0)
                        recover_src.at<float>(i, j) = temp.at<float>(i / 2, j) -
                                                      temp.at<float>(i / 2 + recover_src.rows / 2, j);
                    else
                        recover_src.at<float>(i, j) = temp.at<float>(i / 2, j) + temp.at<float>(i, j / 2 + temp.cols / 2);
                }
            }
            recover_src.copyTo(temp);
        }
        original = full_scale.clone();
        original.convertTo(original, CV_8UC1);
    };

    // 小波操作
    auto wave_operate = [&](Mat &full_scale, int level)
    {
        Mat temp = full_scale(Rect(0, 0, full_scale.cols / 4, full_scale.rows / 4));
        temp = temp(Rect(0, 0, temp.cols / 2, temp.rows / 2));
        Mat temp2 = temp.clone();
        for (int i = 0; i < temp2.rows; i++)
            for (int j = 0; j < temp2.cols; j++)
                temp2.at<float>(i, j) -= 20;
        temp2.copyTo(temp);
        for (int i = temp.rows / 2; i < temp.rows; i++)
        {
            for (int j = 0; j < temp.cols / 2; j++)
            {
                if (temp.at<float>(i, j) > 0)
                    temp.at<float>(i, j) += 5;
                if (temp.at<float>(i, j) < 0)
                    temp.at<float>(i, j) -= 5;
            }
        }
    };

    // 小波分解
    auto waveletDecompose = [&](Mat &_src, Mat &_lowFilter, Mat &_highFilter)
    {
        assert(_src.rows == 1 && _lowFilter.rows == 1 && _highFilter.rows == 1);
        assert(_src.cols >= _lowFilter.cols && _src.cols >= _highFilter.cols);
        Mat src = Mat_<float>(_src);
        int D = src.cols;
        Mat lowFilter = Mat_<float>(_lowFilter);
        Mat highFilter = Mat_<float>(_highFilter);

        Mat dst1 = Mat::zeros(1, D, src.type());
        Mat dst2 = Mat::zeros(1, D, src.type());
        filter2D(src, dst1, -1, lowFilter);
        filter2D(src, dst2, -1, highFilter);
        Mat downDst1 = Mat::zeros(1, D / 2, src.type());
        Mat downDst2 = Mat::zeros(1, D / 2, src.type());
        resize(dst1, downDst1, downDst1.size());
        resize(dst2, downDst2, downDst2.size());
        for (int i = 0; i < D / 2; i++)
        {
            src.at<float>(0, i) = downDst1.at<float>(0, i);
            src.at<float>(0, i + D / 2) = downDst2.at<float>(0, i);
        }
        return src;
    };

    Mat gray, res;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    imshow("gray", gray);
    Mat wave;
    // 拉普拉斯分解
    laplace_decompose(gray, 3, wave);
    // 小波操作
    wave_operate(wave, 3);
    // 重构
    wave_recover(wave, res, 3);
    imshow("output", res);
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
    wavelet_seg(image, dst);
    waitKey(0);
    return 0;
}