#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <unordered_map>
#include "Eigen/Dense"
#include "opencv2/core/eigen.hpp"
#include <fstream>
#include <iomanip>
#include <yaml-cpp/yaml.h>
#include <filesystem>
// Faster replacement for std::unordered_map used for extracting unique values
#include "robin_hood.h"

double degToRad(double deg) {
    return deg * (M_PI / 180);
}

double radToDeg(double rad) {
    return rad * (180 / M_PI);
}

cv::Mat EdgeDetect(cv::Mat &img, int lowTresh, int highTresh, int kernelSize) {
    cv::Mat ret;
    cv::GaussianBlur(img, ret, {3, 3}, 0);
    Canny(ret, ret, lowTresh, highTresh, kernelSize);
    return ret;
}

struct HoughLinesReturn {
    Eigen::MatrixXd accumulator;
    Eigen::ArrayXd rhos;
    Eigen::ArrayXd thetas;
};

HoughLinesReturn HoughLines(const cv::Mat &img, double angle_step = 0.1) {

    int thetas_size = round(M_PI / degToRad(angle_step));
    int width = img.size().width;
    int height = img.size().height;
    int img_diag = round(sqrt(width * width + height * height));

    Eigen::ArrayXd thetas = Eigen::ArrayXd::LinSpaced(thetas_size, -M_PI_2, M_PI_2);
    Eigen::ArrayXd rhos = Eigen::ArrayXd::LinSpaced(img_diag * 2, -img_diag, img_diag);
    Eigen::MatrixXd mat_eigen;
    cv::cv2eigen(img, mat_eigen);

    std::vector<double> edge_x;
    std::vector<double> edge_y;
    for (int i = 0; i < mat_eigen.rows(); i++)
        for (int j = 0; j < mat_eigen.cols(); j++)
            if (mat_eigen(i, j) > 5) {
                edge_x.push_back(i);
                edge_y.push_back(j);
            }

    Eigen::VectorXd idxs_x = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(edge_x.data(), edge_x.size());
    Eigen::VectorXd idxs_y = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(edge_y.data(), edge_y.size());

    // rho = x * Cos(theta) + y * Sin(Theta)
    Eigen::MatrixXd x_CosTheta = idxs_x * thetas.cos().matrix().transpose();
    Eigen::MatrixXd y_SinTheta = idxs_y * thetas.sin().matrix().transpose();
    Eigen::MatrixXi rhosmat = ((x_CosTheta + y_SinTheta).array().round() + img_diag).cast<int>();

    Eigen::MatrixXd accumulator = Eigen::MatrixXd::Zero(2 * img_diag, thetas_size);
    for (int i = 0; i < thetas_size; i++) {
        const auto &curr_rho_col = rhosmat.col(i);
        robin_hood::unordered_map<int, int> unique_rhos;
        for (int j = 0; j < curr_rho_col.size(); j++)
            unique_rhos[curr_rho_col(j)]++;

        for (auto &[unique_rho, count]: unique_rhos)
            accumulator(unique_rho, i) = count;
    }

    return {accumulator, rhos, thetas};
}

int main(int argc, char **argv) {

    if (argc < 2) {
        std::cout << "Please provide a map.yaml file.";
    }
    std::filesystem::path map_yaml_path(argv[1]);
    YAML::Node map_yaml_node;
    try {
        map_yaml_node = YAML::LoadFile(absolute(map_yaml_path));
    } catch (const YAML::BadFile &e) {
        std::cerr << e.msg << std::endl;
        return 1;
    }
    // Load the image
    std::filesystem::path map_path = map_yaml_path.replace_filename(map_yaml_node["image"].as<std::string>());
    cv::Mat src = imread(cv::samples::findFile(map_path), cv::IMREAD_GRAYSCALE);

    // Run edge detection
    cv::Mat dst = EdgeDetect(src, 230, 250, 3);

    constexpr int INITIAL_ROTATION = -10;
    // Rotate image by a small angle for more accuracy when running Hough
    cv::Point2f pc(dst.cols / 2., dst.rows / 2.);
    cv::Mat r = cv::getRotationMatrix2D(pc, INITIAL_ROTATION, 1.0);
    cv::warpAffine(dst, dst, r, src.size());

    HoughLinesReturn ret = HoughLines(dst);
    const auto &acc = ret.accumulator;

    // Squaring the accumulator, so we can sum up the columns
    // By doing this we are left with only intensities for each angle between -90 and 90 degrees
    Eigen::MatrixXd acc_sqr = acc.array().pow(2);
    Eigen::ArrayXd acc_column_sum = acc_sqr.colwise().sum();
    Eigen::ArrayXd acc_column_sum_copy = acc_column_sum;

    // Because there are two main axes seperated by 90 degrees that should be the strongest in the images
    // If we shift the array by 90 degrees or half of the array since it represents values between -90 and 90
    // and add up original and shifted values, we will get positive interference on the main axes
    for (int i = 0; i < acc_column_sum.size(); i++) {
        int second = (i + (acc_column_sum.size() / 2)) % acc_column_sum.size();
        acc_column_sum(i) += acc_column_sum_copy(second);
    }
    int idx;
    acc_column_sum.maxCoeff(&idx);
    double angle_correction = ret.thetas[idx] - degToRad(INITIAL_ROTATION);
    std::cout << "Rotation correction needed: " << -angle_correction
              << " rad (" << -radToDeg(angle_correction) << " deg)\n";

    double initial_map_rotation = map_yaml_node["origin"].as<std::vector<double>>()[2];
    double corrected_rotation = initial_map_rotation - angle_correction;
    std::cout << "New map origin rotation: " << corrected_rotation << " rad (" << radToDeg(corrected_rotation)
              << " deg)\n";
    return 0;
}