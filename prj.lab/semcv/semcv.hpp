#pragma once
#ifndef SEMCV
#define SEMCV

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

std::string strid_from_mat(const cv::Mat& img, const int n = 4);

std::vector<std::filesystem::path> get_list_of_file_paths(const std::filesystem::path& path_lst);

cv::Mat gen_img(const int width = 768, const int height = 30, const int stripe_w = 3);

cv::Mat gamma_correct(const cv::Mat& img, const double gamma);
#endif