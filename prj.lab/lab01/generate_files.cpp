#include <opencv2/opencv.hpp>
#include <semcv/semcv.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>


std::string getProjectRootDir() {
    std::filesystem::path src_path(__FILE__);
    std::filesystem::path parent = src_path.parent_path();
    return parent.string();
}

int main() {
    std::string project_dir = getProjectRootDir();
    std::string output_dir = project_dir + "/test_images/";

    std::filesystem::create_directories(output_dir);
    std::cout << "Output directory: " << output_dir << std::endl;

    std::vector<std::pair<int, int>> specs = {
        {CV_8U,  1}, {CV_8U,  3}, {CV_8U,  4},
        {CV_8S,  1}, {CV_8S,  3},
        {CV_16U, 1}, {CV_16U, 3}, {CV_16U, 4},
        {CV_16S, 1}, {CV_16S, 3},
        {CV_32S, 1}, {CV_32S, 3},
        {CV_32F, 1}, {CV_32F, 3},
        {CV_64F, 1}, {CV_64F, 3}
    };

    std::vector<std::string> filenames;

    for (auto [depth, channels] : specs) {
        constexpr int H = 64;
        constexpr int W = 64;
        cv::Mat img;
        switch (depth) {
            case CV_8U:  { cv::Mat tmp(H, W, CV_MAKETYPE(CV_8U, channels), cv::Scalar(128)); img = tmp; } break;
            case CV_8S:  { cv::Mat tmp(H, W, CV_MAKETYPE(CV_8S, channels), cv::Scalar(-64)); img = tmp; } break;
            case CV_16U: { cv::Mat tmp(H, W, CV_MAKETYPE(CV_16U, channels), cv::Scalar(32768)); img = tmp; } break;
            case CV_16S: { cv::Mat tmp(H, W, CV_MAKETYPE(CV_16S, channels), cv::Scalar(-16384)); img = tmp; } break;
            case CV_32S: { cv::Mat tmp(H, W, CV_MAKETYPE(CV_32S, channels), cv::Scalar(1000000)); img = tmp; } break;
            case CV_32F: { cv::Mat tmp(H, W, CV_MAKETYPE(CV_32F, channels), cv::Scalar(0.5f)); img = tmp; } break;
            case CV_64F: { cv::Mat tmp(H, W, CV_MAKETYPE(CV_64F, channels), cv::Scalar(0.75)); img = tmp; } break;
            default: continue;
        }

        std::string strid = strid_from_mat(img);

        std::vector<std::string> exts;
        if (depth == CV_8U && (channels == 1 || channels == 3)) {
            exts = {"jpg", "png", "tiff"};
        } else if (depth == CV_8U || depth == CV_16U) {
            exts = {"png", "tiff"};
        } else {
            exts = {"tiff"};
        }

        for (const auto& ext : exts) {
            std::string filename = strid + "." + ext;
            std::string path = output_dir + filename;

            if (cv::imwrite(path, img)) {
                std::cout << "Saved: " << filename << std::endl;
                filenames.push_back(filename);
            } else {
                std::cerr << "Failed to save: " << filename << std::endl;
            }
        }
    }

    std::ofstream lst(output_dir + "task01.lst");
    if (lst.is_open()) {
        for (const auto& name : filenames) {
            lst << name << "\n";
        }
        lst.close();
        std::cout << "\n task01.lst written (" << filenames.size() << " files)." << std::endl;
    } else {
        std::cerr << "Could not open task01.lst for writing!" << std::endl;
        return 1;
    }

    return 0;
}