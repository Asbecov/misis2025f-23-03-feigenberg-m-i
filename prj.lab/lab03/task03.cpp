#include <opencv2/opencv.hpp>
#include <semcv/semcv.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

void threshold_binarization(const std::string& input_path, const std::string& output_dir, double threshold = 127.0) {
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return;
    }

    cv::Mat gray = to_grayscale(img);

    std::string gray_path = output_dir + "/grayscale.png";
    cv::imwrite(gray_path, gray);
    std::cout << "Saved grayscale to " << gray_path << std::endl;

    cv::Mat hist_img = draw_histogram_8u(gray);
    std::string hist_path = output_dir + "/histogram.png";
    cv::imwrite(hist_path, hist_img);
    std::cout << "Saved histogram to " << hist_path << std::endl;

    cv::Mat binary = global_binarization(gray, threshold);
    std::string binary_path = output_dir + "/binary_mask.png";
    cv::imwrite(binary_path, binary);
    std::cout << "Saved binary mask to " << binary_path << std::endl;

    cv::Mat overlay = overlay_mask(img, binary);
    std::string overlay_path = output_dir + "/overlay.png";
    cv::imwrite(overlay_path, overlay);
    std::cout << "Saved overlay to " << overlay_path << std::endl;

}

void quality_assessment(const std::string& img_path, const std::string& mask_path, const std::string& output_dir) {
    const cv::Mat img = cv::imread(img_path);
    const cv::Mat mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);

    if (img.empty() || mask.empty()) {
        std::cerr << "Failed to load images" << std::endl;
        return;
    }

    cv::Mat gray = to_grayscale(img);

    double best_iou = 0.0;
    int best_threshold = 0;

    std::string stats_path = output_dir + "/threshold_analysis.csv";
    std::ofstream ofs(stats_path);

    ofs << "threshold,TP,FP,FN,TN,TPR,FPR,Precision,IoU,Accuracy" << std::endl;

    for (int t = 0; t <= 255; t += 5) {
        cv::Mat binary = global_binarization(gray, t);
        BinaryClassificationMetrics metrics = calc_binary_metrics(binary, mask);

        if (metrics.IoU() > best_iou) best_threshold = t;

        ofs << std::fixed << std::setprecision(2)
            << t << ","
            << metrics.TP << "," << metrics.FP << "," << metrics.FN << "," << metrics.TN << ","
            << metrics.TPR() << "," << metrics.FPR() << ","
            << metrics.Precision() << "," << metrics.IoU() << ","
            << metrics.Accuracy() << std::endl;
    }

    ofs.close();
    std::cout << "Saved threshold analysis to " << stats_path << std::endl;

    std::cout << "Best threshold: " << best_threshold << " (IoU = " << best_iou << ")" << std::endl;

    cv::Mat best_binary = global_binarization(gray, best_threshold);
    std::string best_binary_path = output_dir + "/best_binary.png";
    cv::imwrite(best_binary_path, best_binary);

    cv::Mat best_overlay = overlay_mask(img, best_binary, cv::Scalar(0, 255, 0), 0.5);
    std::string best_overlay_path = output_dir + "/best_overlay.png";
    cv::imwrite(best_overlay_path, best_overlay);

    const BinaryClassificationMetrics best_metrics = calc_binary_metrics(best_binary, mask);
    std::cout << std::endl << "Best Threshold Metrics " << std::endl;
    std::cout << "Threshold: " << best_threshold << std::endl;
    std::cout << "TP: " << best_metrics.TP << ", FP: " << best_metrics.FP
              << ", FN: " << best_metrics.FN << ", TN: " << best_metrics.TN << std::endl;
    std::cout << "TPR (Recall): " << best_metrics.TPR() << std::endl;
    std::cout << "FPR: " << best_metrics.FPR() << std::endl;
    std::cout << "Precision: " << best_metrics.Precision() << std::endl;
    std::cout << "IoU: " << best_metrics.IoU() << std::endl;
    std::cout << "Accuracy: " << best_metrics.Accuracy() << std::endl;
}

void generate_masks(const std::string& input_path, const std::string& output_dir) {
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return;
    }

    cv::Mat gray = to_grayscale(img);

    cv::Mat mask1 = global_binarization(gray, 127.0);
    std::string mask1_path = output_dir + "/mask_127.png";
    cv::imwrite(mask1_path, mask1);

    cv::Mat mask2;
    cv::threshold(gray, mask2, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    std::string mask2_path = output_dir + "/mask_otsu.png";
    cv::imwrite(mask2_path, mask2);

    cv::Mat mask3;
    cv::adaptiveThreshold(gray, mask3, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
    std::string mask3_path = output_dir + "/mask_adaptive.png";
    cv::imwrite(mask3_path, mask3);

    std::cout << "Generated masks:" << std::endl
                << "  - " << mask1_path << " (threshold 127)" << std::endl
                << "  - " << mask2_path << " (Otsu)" << std::endl
                << "  - " << mask3_path << " (adaptive)" << std::endl;
}

int main(const int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage:" << std::endl
                  << "  Threshold binarization: " << argv[0] << " --thresh_bin <input_image> <output_dir> [threshold]" << std::endl
                  << "  Quality assessment: " << argv[0] << " --assess_quality <input_image> <mask> <output_dir>" << std::endl
                  << "  Generate masks: " << argv[0] << " --gen_masks <input_image> <output_dir>" << std::endl;
        return 1;
    }

    if (const std::string mode = argv[1]; mode == "--thresh_bin") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " --thresh_bin <input_image> <output_dir> [threshold]" << std::endl;
            return 1;
        }
        const int threshold = (argc >= 5) ? std::stod(argv[4]) : 127.0;
        threshold_binarization(argv[2], argv[3], threshold);
    } else if (mode == "--assess_quality") {
        if (argc < 5) {
            std::cerr << "Usage: " << argv[0] << " --assess_quality <input_image> <mask> <output_dir>" << std::endl;
            return 1;
        }
        quality_assessment(argv[2], argv[3], argv[4]);
    } else if (mode == "--gen_masks") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " --gen_masks <input_image> <output_dir>" << std::endl;
            return 1;
        }
        generate_masks(argv[2], argv[3]);
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }

    return 0;
}