#pragma once

#include "common_include.h"

#define PI 3.1415926

static cv::Mat Euler2Rotation_Matrix(const std::vector<double> &Euler_Angle)
{
    double roll(Euler_Angle[0]);
    double pitch(Euler_Angle[1]);
    double yaw(Euler_Angle[2]);

    double cos_y = (abs(cos(yaw)) < 1e-3 ? 0 : cos(yaw));
    double cos_p = (abs(cos(pitch)) < 1e-3 ? 0 : cos(pitch));
    double cos_r = (abs(cos(roll)) < 1e-3 ? 0 : cos(roll));
    double sin_y = (abs(sin(yaw)) < 1e-3 ? 0 : sin(yaw));
    double sin_p = (abs(sin(pitch)) < 1e-3 ? 0 : sin(pitch));
    double sin_r = (abs(sin(roll)) < 1e-3 ? 0 : sin(roll));

    cv::Mat R_x = (cv::Mat_<double>(3, 3) << 1, 0, 0,
                   0, cos_r, -sin_r,
                   0, sin_r, cos_r);
    cv::Mat R_y = (cv::Mat_<double>(3, 3) << cos_p, 0, sin_p,
                   0, 1, 0,
                   -sin_p, 0, cos_p);
    cv::Mat R_z = (cv::Mat_<double>(3, 3) << cos_y, -sin_y, 0,
                   sin_y, cos_y, 0,
                   0, 0, 1);
    return R_z * R_y * R_x;
}

static cv::Mat inv(cv::Mat T)
{
    cv::Mat R = T.rowRange(0, 3).colRange(0, 3);
    cv::Mat t = T.rowRange(0, 3).col(3);
    cv::Mat res(3, 4, CV_64F);
    R = R.t();
    t = -R * t;
    R.copyTo(res.rowRange(0, 3).colRange(0, 3));
    t.copyTo(res.rowRange(0, 3).col(3));
    return res;
}

static cv::Mat multiply(const cv::Mat &M1,const cv::Mat &M2)
{
    cv::Mat res(3, 4, CV_64F);
    cv::Mat R = M1.rowRange(0, 3).colRange(0, 3);
    cv::Mat t = M1.rowRange(0, 3).col(3);
    cv::Mat R_ = M2.rowRange(0, 3).colRange(0, 3);
    cv::Mat t_ = M2.rowRange(0, 3).col(3);
    R = R * R_;
    t = R * t_ + t;
    R.copyTo(res.rowRange(0, 3).colRange(0, 3));
    t.copyTo(res.rowRange(0, 3).col(3));
    return res;
}

static cv::Vec3d multiply(const cv::Mat &M,const cv::Vec3d &v)
{
    cv::Mat res(3, 1, CV_64F);
    cv::Mat R = M.rowRange(0, 3).colRange(0, 3);
    cv::Mat t = M.rowRange(0, 3).col(3);
    res = R * cv::Mat(v) + t;
    return cv::Vec3d(res.at<double>(0, 0),
                     res.at<double>(1, 0),
                     res.at<double>(2, 0));
}

class Camera
{
public:
    Camera(){};
    double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0, baseline_ = 0;
    cv::Mat pose_;
    cv::Mat pose_inv_;

    Camera(double fx, double fy, double cx, double cy, double baseline, cv::Mat pose):
        fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose)
    {
        pose_inv_ = inv(pose_);
    }

    cv::Mat get_pose() const
    {
        return pose_;
    }

    cv::Mat get_K() const
    {
        cv::Mat K = (cv::Mat_<double>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
        return K;
    }

    cv::Vec3d w2c(const cv::Vec3d &p_w, const cv::Mat &T_c_w);
    cv::Vec3d c2w(const cv::Vec3d &p_c, const cv::Mat &T_c_w);

    cv::Vec2d c2p(const cv::Vec3d &p_c);
    cv::Vec3d p2c(const cv::Vec2d &p_p, const double depth = 1);

    cv::Vec3d p2w(const cv::Vec2d &p_p, const cv::Mat &T_c_w, const double depth = 1);
    cv::Vec2d w2p(const cv::Vec3d &p_w, const cv::Mat &T_c_w);

    
};