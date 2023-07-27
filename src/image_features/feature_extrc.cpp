/**
 * @file feature_extrc.cpp
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

const int HALF_PATCH_SIZE = 15;		///<上面这个大小的一半，或者说是半径
static float IC_Angle(const Mat& image, Point2f pt,  const vector<int> & u_max)
{
	//图像的矩，前者是按照图像块的y坐标加权，后者是按照图像块的x坐标加权
    int m_01 = 0, m_10 = 0;

	//获得这个特征点所在的图像块的中心点坐标灰度值的指针center
    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

	//这条v=0中心线的计算需要特殊对待
    //由于是中心行+若干行对，所以PATCH_SIZE应该是个奇数
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
		//注意这里的center下标u可以是负的！中心水平线上的像素按x坐标（也就是u坐标）加权
        m_10 += u * center[u];

	//这里的step1表示这个图像一行包含的字节总数。
    int step = (int)image.step1();
	//注意这里是以v=0中心线为对称轴，然后对称地每成对的两行之间进行遍历，这样处理加快了计算速度
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
		//本来m_01应该是一列一列地计算的，但是由于对称以及坐标x,y正负的原因，可以一次计算两行
        int v_sum = 0;
		// 获取某行像素横坐标的最大范围，注意这里的图像块是圆形的！
        int d = u_max[v];
		//在坐标范围内挨个像素遍历，实际是一次遍历2个
        // 假设每次处理的两个点坐标，中心线下方为(x,y),中心线上方为(x,-y) 
        // 对于某次待处理的两个点：m_10 = Σ x*I(x,y) =  x*I(x,y) + x*I(x,-y) = x*(I(x,y) + I(x,-y))
        // 对于某次待处理的两个点：m_01 = Σ y*I(x,y) =  y*I(x,y) - y*I(x,-y) = y*(I(x,y) - I(x,-y))
        for (int u = -d; u <= d; ++u)
        {
			//得到需要进行加运算和减运算的像素灰度值
			//val_plus：在中心线下方x=u时的的像素灰度值
            //val_minus：在中心线上方x=u时的像素灰度值
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
			//在v（y轴）上，2行所有像素灰度值之差
            v_sum += (val_plus - val_minus);
			//u轴（也就是x轴）方向上用u坐标加权和（u坐标也有正负符号），相当于同时计算两行
            m_10 += u * (val_plus + val_minus);
        }
        //将这一行上的和按照y坐标加权
        m_01 += v * v_sum;
    }
    // 计算角度，精度为0.3度
    return fastAtan2((float)m_01, (float)m_10);
}

void SIFT_features(Mat &image, Mat &dst)
{
    // 如果不是灰度图，转换为灰度图
    if (image.channels() != 1)
    {
        cvtColor(image, image, COLOR_BGR2GRAY);
    }
    // 提取角点
    vector<Point2f> keypoints_;
    cv::goodFeaturesToTrack(image, keypoints_, 1000, 0.01, 10);
    // 转换为keypoint
    vector<KeyPoint> keypoints;
    for (auto kp : keypoints_)
    {
        keypoints.push_back(KeyPoint(kp, 1.0));
    }
    // 计算梯度和角度
    GaussianBlur(image, image, Size(5, 5), 1, 1);
    Mat grad_x, grad_y;
    // 自定义kernel
        // 计算梯度
    Mat kernel_x = (Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1); // x方向
    Mat kernel_y = (Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1); // y方向
    filter2D(image, grad_x, CV_32F, kernel_x);
    filter2D(image, grad_y, CV_32F, kernel_y);
    Mat grad;
    cv::magnitude(grad_x, grad_y, grad);
        // 计算角度
    Mat angle = Mat::zeros(image.size(), CV_32F);
    phase(grad_y, grad_x, angle);
    // 计算每个关键点主方向（灰度质心法）
    for(auto kp: keypoints)
    {
        // 计算关键点周围的梯度和角度
        vector<int> u_max(HALF_PATCH_SIZE + 2);
        // 计算每一行的最大值
        for (int v = -HALF_PATCH_SIZE; v <= HALF_PATCH_SIZE; ++v)
        {
            // 计算每一行的最大值
            int vmax = 0;
            for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
            {
                int val_plus = image.at<uchar>(cvRound(kp.pt.y + v), cvRound(kp.pt.x + u));
                int val_minus = image.at<uchar>(cvRound(kp.pt.y - v), cvRound(kp.pt.x - u));
                vmax = max(vmax, max(val_plus, val_minus));
            }
            u_max[v + HALF_PATCH_SIZE + 1] = vmax;
        }
        // 计算主方向
        kp.angle = IC_Angle(image, kp.pt, u_max);
    }
    // 画出关键点
    drawKeypoints(image, keypoints, dst, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("sift", dst);
}

void features(Mat &image, Mat &dst)
{
    // harris
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    dst = Mat::zeros(image.size(), CV_32FC1);
    Mat conners;
    cornerHarris(gray, conners, 2, 3, 0.04, BORDER_DEFAULT);
    Mat conners_norm;
    normalize(conners, conners_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    Mat conners_norm_scaled;
    convertScaleAbs(conners_norm, conners_norm_scaled);
    // 画出响应前100的角点
    // 写大顶堆对角点进行排序（比较策略为响应高的在前）
    auto cmp = [](pair<double, Point> a, pair<double, Point> b)
    { return a.first < b.first; };
    priority_queue<pair<double, Point>, vector<pair<double, Point>>, decltype(cmp)> q(cmp);
    for (int i = 0; i < conners_norm.rows; i++)
    {
        for (int j = 0; j < conners_norm.cols; j++)
        {
            q.push(make_pair(conners_norm.at<float>(i, j), Point(j, i)));
        }
    }
    // 取前1000个,转换为keypoint，画出来
    vector<KeyPoint> keypoints_harris;
    for (int i = 0; i < 1000; i++)
    {
        keypoints_harris.push_back(KeyPoint(q.top().second.x, q.top().second.y, 1.0));
        q.pop();
    }
    drawKeypoints(image, keypoints_harris, dst, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("harris", dst);
    waitKey(0);

    // orb
    Mat orb_dst;
    Ptr<Feature2D> orb = ORB::create(1000);
    vector<KeyPoint> keypoints_orb;
    orb->detect(image, keypoints_orb);
    drawKeypoints(image, keypoints_orb, orb_dst, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("orb", orb_dst);
    waitKey(0);

    // kaze
    Mat kaze_dst;
    Ptr<Feature2D> kaze = KAZE::create();
    vector<KeyPoint> keypoints_kaze;
    kaze->detect(image, keypoints_kaze);
    drawKeypoints(image, keypoints_kaze, kaze_dst, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("kaze", kaze_dst);
    waitKey(0);

    // akaze
    Mat akaze_dst;
    Ptr<Feature2D> akaze = AKAZE::create();
    vector<KeyPoint> keypoints_akaze;
    akaze->detect(image, keypoints_akaze);
    drawKeypoints(image, keypoints_akaze, akaze_dst, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("akaze", akaze_dst);
    waitKey(0);

    // sift
    Mat sift_dst;
    SIFT_features(image, sift_dst);
    waitKey(0);
    destroyAllWindows();
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
    features(image, dst);
    
    return 0;
}