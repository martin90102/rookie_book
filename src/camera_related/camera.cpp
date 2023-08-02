#include "camera.h"

cv::Vec3d Camera::w2c(const cv::Vec3d &p_w, const cv::Mat &T_c_w)
{
    cv::Mat temp = multiply(pose_, T_c_w) ;
    return multiply(temp,p_w);
}

cv::Vec3d Camera::c2w(const cv::Vec3d &p_c, const cv::Mat &T_c_w)
{
    cv::Mat temp =multiply(inv(T_c_w) , inv(pose_));
    return multiply(temp,p_c);
}

cv::Vec2d Camera::c2p(const cv::Vec3d &p_c)
{
    return cv::Vec2d(fx_ * p_c[0] / p_c[2] + cx_,
                     fy_ * p_c[1] / p_c[2] + cy_);
}

cv::Vec3d Camera::p2c(const cv::Vec2d &p_p, const double depth)
{
    return cv::Vec3d((p_p[0] - cx_) * depth / fx_,
                     (p_p[1] - cy_) * depth / fy_,
                     depth);
}

cv::Vec3d Camera::p2w(const cv::Vec2d &p_p, const cv::Mat &T_c_w, const double depth)
{
    return c2w(p2c(p_p, depth), T_c_w);
}

cv::Vec2d Camera::w2p(const cv::Vec3d &p_w, const cv::Mat &T_c_w)
{
    return c2p(w2c(p_w, T_c_w));
}

int main(int argc, char **argv)
{
    cv::Mat ini = (cv::Mat_<double>(3, 4) << 1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0);
    Camera camera(520.9, 521.0, 325.1, 249.7, 0.01, ini);
    cv::Vec3d p_w(1, 2, 1);
    std::vector<double> ypr(3,5/180.0*PI); 
    // make the angle like 5 degree each axis?
    //calculate the rotation matrix
    cv::Mat R = Euler2Rotation_Matrix(ypr);
    cv::Mat T_c_w;
    cv::hconcat(R, cv::Mat(cv::Vec3d(1, 1, 1)), T_c_w);
    cv::Vec3d p_c = camera.w2c(p_w, T_c_w);
    std::cout << "p_c = " << p_c << std::endl;
    cv::Vec2d p_p = camera.c2p(p_c);
    std::cout << "p_p = " << p_p << std::endl;
    cv::Vec3d p_c2 = camera.p2c(p_p);
    std::cout << "p_c2 = " << p_c2 << std::endl;
    cv::Vec3d p_w2 = camera.c2w(p_c2, T_c_w);
    std::cout << "p_w2 = " << p_w2 << std::endl;
    cv::Vec2d p_p2 = camera.w2p(p_w, T_c_w);
    std::cout << "p_p2 = " << p_p2 << std::endl;
    cv::Vec3d p_w3 = camera.p2w(p_p2, T_c_w);
    std::cout << "p_w3 = " << p_w3 << std::endl;
    return 0;
}