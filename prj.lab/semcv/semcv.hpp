#pragma once
#ifndef SEMCV
#define SEMCV

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

struct PixelDistributionStats {
    int count{};
    double mean{};
    double variance{};
    double stddev{};
    double minimum{};
    double maximum{};
};

// lab_01
std::string strid_from_mat(const cv::Mat& img, int n = 4);

std::vector<std::filesystem::path> get_list_of_file_paths(const std::filesystem::path& path_lst);

cv::Mat gen_gray_bars(int width = 768, int height = 30, int stripe_w = 3);

cv::Mat gamma_correct(const cv::Mat& img, double gamma);

//lab_02
constexpr int IMG_SIZE = 256;
constexpr int SQ_SIZE = 209;
constexpr int RADIUS = 83;
cv::Mat gen_tgtimg00(int lev0, int lev1, int lev2);

cv::Mat add_noise_gau(const cv::Mat& img, int sigma);

PixelDistributionStats calc_distribution_stats(const cv::Mat& img, const cv::Mat& mask);

cv::Mat draw_histogram_8u(const cv::Mat& img, const cv::Scalar& background_color = cv::Scalar(224, 224, 224), const cv::Scalar& bar_color = cv::Scalar(32, 32, 32));
#endif