#pragma once
#ifndef SEMCV
#define SEMCV

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

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

struct PixelDistributionStats {
    int count{};
    double mean{};
    double variance{};
    double stddev{};
    int minimum{};
    int maximum{};
};

PixelDistributionStats calc_distribution_stats(const cv::Mat& img, const cv::Mat& mask);

cv::Mat draw_histogram_8u(const cv::Mat& img, const cv::Scalar& background_color = cv::Scalar(224, 224, 224), const cv::Scalar& bar_color = cv::Scalar(32, 32, 32));

// lab_03
cv::Mat to_grayscale(const cv::Mat& img);

cv::Mat global_binarization(const cv::Mat& img, const int threshold, const int maxVal = 255);

cv::Mat overlay_mask(const cv::Mat& img, const cv::Mat& mask, const cv::Scalar& mask_color = cv::Scalar(0, 255, 0), const double alpha = 0.5);


struct BinaryClassificationMetrics {
    int TP{0};  // True Positive
    int FP{0};  // False Positive
    int FN{0};  // False Negative
    int TN{0};  // True Negative

    double TPR() const { // True positive rate
        if ((TP + FN > 0)) return static_cast<double>(TP) / (TP + FN);
        return 0.0;
    }

    double FPR() const { // False positive rate
        if ((FP + TN > 0)) return static_cast<double>(FP) / (FP + TN);
        return 0.0;
    }

    double Precision() const {
        if ((TP + FP > 0)) return static_cast<double>(TP) / (TP + FP);
        return 0.0;
    }

    double IoU() const {
        if ((TP + FP + FN > 0)) return static_cast<double>(TP) / (TP + FP + FN);
        return 0.0;
    }

    double Accuracy() const {
        const int total = TP + FP + FN + TN;
        if (total > 0) return static_cast<double>(TP + TN) / total;
        return 0.0;
    }
};

BinaryClassificationMetrics calc_binary_metrics(const cv::Mat& predicted_mask, const cv::Mat& mask);

// lab_04
cv::Mat create_segmentation_mask(const cv::Mat& img);

cv::Mat overlay_segmentation(const cv::Mat& img, const cv::Mat& mask);

#endif