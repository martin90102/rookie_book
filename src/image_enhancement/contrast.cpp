/**
 * @file contrast.cpp
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
#include <opencv2/xphoto.hpp>
using namespace std;
using namespace cv;

// 计算暗通道
Mat getDarkChannel(const Mat &src, int size)
{
    Mat rgbmin = Mat::zeros(src.size(), CV_8UC1);
    Mat temp;
    for (int i = 0; i < src.channels(); i++)
    {
        extractChannel(src, temp, i);
        if (i == 0)
            temp.copyTo(rgbmin);
        min(rgbmin, temp, rgbmin);
    }
    Mat kernel = getStructuringElement(MORPH_RECT, Size(size, size));
    erode(rgbmin, rgbmin, kernel);
    return rgbmin;
}

// 估计大气光
int getAirlight(const Mat &src, const Mat &dark)
{
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(dark, &minVal, &maxVal, &minLoc, &maxLoc);
    int maxIntensity = 0;
    for (int i = 0; i < src.channels(); i++)
    {
        maxIntensity = max((int)src.at<Vec3b>(maxLoc)[i], maxIntensity);
    }
    return maxIntensity;
}

// 估计传输图
Mat getTransmission(const Mat &src, int airlight, int size, double w)
{
    Mat transmission = Mat::ones(src.size(), CV_64F);
    Mat temp;
    for (int i = 0; i < src.channels(); i++)
    {
        temp = src / (float)airlight;
        extractChannel(temp, temp, i);
        temp = getDarkChannel(temp, size);
        transmission = min(transmission, temp);
    }
    transmission = transmission * w;
    return transmission;
}

// 恢复场景辐射
Mat getRadiance(const Mat &src, int airlight, const Mat &transmission, double t0)
{
    Mat radiance = Mat::zeros(src.size(), src.type());
    for (int i = 0; i < src.channels(); i++)
    {
        Mat temp;
        extractChannel(src, temp, i);
        temp = (temp - airlight) / max(transmission, t0) + airlight;
        temp.copyTo(radiance);
    }
    normalize(radiance, radiance, 0, 255, NORM_MINMAX, CV_8UC1);
    return radiance;
}


Mat dcpEnhance(const Mat &src, int size = 15, double w = 0.01, double t0 = 0.1)
{
    // 转换为64F
    Mat temp;
    src.convertTo(temp, CV_64F);
    // 计算暗通道
    Mat dark = getDarkChannel(temp, size);
    int airlight = getAirlight(temp, dark);
    Mat transmission = getTransmission(temp, airlight, size, w);
    Mat radiance = getRadiance(temp, airlight, transmission, t0);
    return radiance;
}

void contrast(Mat &image, Mat &dst)
{
    // 自动对比度和色阶
    Mat auto_contrast;
    cvtColor(image, auto_contrast, COLOR_BGR2YCrCb);
    vector<Mat> channels;
    split(auto_contrast, channels);
    for(auto &channel: channels)
        equalizeHist(channel, channel);
    merge(channels, auto_contrast);
    cvtColor(auto_contrast, auto_contrast, COLOR_YCrCb2BGR);
    imshow("equalize_plus", auto_contrast);

    Mat gray;
    imshow("origin", image);
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // 直方图均衡化
    Mat equalize_hist;
    equalizeHist(gray, equalize_hist);
    imshow("equalize_hist", equalize_hist);

    // 自适应直方图均衡化
    Mat clahe;
    Ptr<CLAHE> clahe_ptr = createCLAHE();
    clahe_ptr->setClipLimit(4);
    clahe_ptr->setTilesGridSize(Size(8, 8));
    clahe_ptr->apply(gray, clahe);
    imshow("clahe", clahe);

    // DCP增强
    Mat dcp;
    dcp = dcpEnhance(image);
    imshow("dcp", dcp);
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
    contrast(image, dst);
    waitKey(0);
    return 0;
}