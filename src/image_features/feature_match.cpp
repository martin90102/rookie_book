/**
 * @file feature_match.cpp
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
#include <opencv2/xfeatures2d.hpp>
using namespace std;
using namespace cv;

void match(Mat &image)
{   
    Mat copy_ = image.clone();
    // 旋转45度
    Mat affine_matrix = getRotationMatrix2D(Point2f(image.cols / 2, image.rows / 2), 45, 1);
    warpAffine(image, image, affine_matrix, image.size());

    // orb的匹配效果
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    orb->detectAndCompute(image, Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(copy_, Mat(), keypoints2, descriptors2);
    
    // BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptors1, descriptors2, matches);

    // 利用RANSAC算法剔除误匹配
    vector<DMatch> good_matches;
    vector<Point2f> points1, points2;
    for (auto m : matches)
    {
        points1.push_back(keypoints1[m.queryIdx].pt);
        points2.push_back(keypoints2[m.trainIdx].pt);
    }
    Mat mask;
    Mat H = findHomography(points1, points2, RANSAC, 3, mask);
    for (int i = 0; i < mask.rows; ++i)
    {
        if (mask.at<uchar>(i, 0) == 1)
        {
            good_matches.push_back(matches[i]);
        }
    }
    // 画出匹配结果
    Mat dst;
    drawMatches(image, keypoints1, copy_, keypoints2, good_matches, dst);
    imshow("match", dst);

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
    match(image);
    waitKey(0);
    return 0;
}