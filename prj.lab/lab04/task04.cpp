#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <semcv/semcv.hpp>

void create_segment_masks(const std::string& input_path, const std::string& output_dir) {
    const std::vector<std::filesystem::path> path_list = get_list_of_file_paths(input_path);

    for (auto& path : path_list) {
        const cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << path.filename() << std::endl;
            continue;
        }

        const cv::Mat mask = create_segmentation_mask(img);
        const cv::Mat overlay = overlay_segmentation(img, mask);

        double maxVal;
        cv::minMaxLoc(mask, nullptr, &maxVal);
        cv::Mat mask8u;
        mask.convertTo(mask8u, CV_8U, 255.0 / maxVal);

        const std::string mask_out_path = output_dir + "/" + path.stem().string() + "_mask.png";
        const std::string overlay_out_path = output_dir + "/" + path.stem().string() + "_overlay.png";

        cv::imwrite(mask_out_path, mask8u);
        cv::imwrite(overlay_out_path, overlay);

        std::cout << "Saved mask and overlay to " << output_dir << std::endl;
    }
}

void assess_segmentation_quality(const std::string& input_path, const std::string& input_mask_path, const std::string& output_dir) {
    const std::vector<std::filesystem::path> mask_path_list = get_list_of_file_paths(input_mask_path);

    std::vector<cv::Mat> gt_masks;
    for (auto& path : mask_path_list) {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        gt_masks.emplace_back(img);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << path.filename() << std::endl;
            continue;
        }
    }

    const cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return;
    }

    std::string stats_path = output_dir + "/threshold_analysis.csv";
    std::ofstream ofs(stats_path);

    ofs << "FG Threshold,mean iou,mean_precision,mean_recall,mean_accuracy" << std::endl;

    double best_iou = 0;
    double best_thresh = 0;
    SegmentationMetrics best_metrics{};

    for (int i = 0; i < 256; i++) {
        const double threshold = i / 255.0;
        const cv::Mat segmentation_mask = create_segmentation_mask(img, threshold);
        const SegmentationMetrics metrics = calc_segmentation_metrics(segmentation_mask, gt_masks, threshold);

        if (metrics.mean_iou > best_iou) {
            best_iou = metrics.mean_iou;
            best_thresh = threshold;
            best_metrics = metrics;
        }

        ofs << std::fixed << std::setprecision(2)
            << threshold << ","
            << metrics.mean_iou << ","
            << metrics.mean_precision << ","
            << metrics.mean_recall << ","
            << metrics.mean_accuracy << std::endl;
    }

    ofs.close();
    std::cout << "Saved threshold analysis to " << stats_path << std::endl;

    std::cout << "Best threshold: " << best_thresh << " (IoU = " << best_iou << ")" << std::endl;

    const cv::Mat best_segmentation = create_segmentation_mask(img, best_thresh);
    const std::string best_segmentation_path = output_dir + "/best_segmentation.png";
    double maxVal;
    cv::minMaxLoc(best_segmentation, nullptr, &maxVal);
    cv::Mat mask8u;
    best_segmentation.convertTo(mask8u, CV_8U, 255.0 / maxVal);
    cv::imwrite(best_segmentation_path, mask8u);

    const cv::Mat best_overlay = overlay_segmentation(img, best_segmentation);
    const std::string best_overlay_path = output_dir + "/best_overlay.png";
    cv::imwrite(best_overlay_path, best_overlay);

    std::cout << std::endl << "Best Segmentation Metrics " << std::endl;
    std::cout << "Mean IoU: " << best_metrics.mean_iou << std::endl;
    std::cout << "Mean Precision: " << best_metrics.mean_precision << std::endl;
    std::cout << "Mean Recall: " << best_metrics.mean_recall << std::endl;
    std::cout << "Mean Accuracy: " << best_metrics.mean_accuracy << std::endl;
}

int main(const int argc, const char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << std::endl
                  << "Create segmentation: " << argv[0] << "--create input_path output_dir" << std::endl
                  << "Asses segmentation quality: " << argv[0] << "--assess input_path input_mask_path output_dir" << std::endl;
        return 1;
    }

    if (const std::string mode = argv[1]; mode == "--create") {
        create_segment_masks(argv[2], argv[3]);
    }
    else if (mode == "--assess") {
        assess_segmentation_quality(argv[2], argv[3], argv[4]);
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }
}