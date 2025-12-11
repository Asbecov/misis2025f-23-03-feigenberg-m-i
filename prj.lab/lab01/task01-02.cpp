#include <iostream>
#include <opencv2/opencv.hpp>
#include <semcv/semcv.hpp>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: task01-02 <out_png_path>" << std::endl;
        return 2;
    }
    const std::vector<double> gammas = {1.8, 2.0, 2.2, 2.4, 2.6};
    const cv::Mat orig = gen_img();
    std::vector<cv::Mat> rows;
    for (const double& g : gammas) {
        cv::Mat corrected = gamma_correct(orig, g);
        cv::Mat pair;
        cv::hconcat(orig, corrected, pair);
        rows.push_back(pair);
    }
    cv::Mat collage;
    cv::vconcat(rows, collage);
    if (!cv::imwrite(argv[1], collage)) {
        std::cerr << "Failed to write " << argv[1] << std::endl;
        return 3;
    }
    std::cout << "Saved: " << argv[1] << std::endl;
    return 0;
}