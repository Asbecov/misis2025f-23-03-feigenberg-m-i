#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "semcv/semcv.hpp"

std::vector<cv::Rect> read_bboxes_from_file(const std::string &filename) {
    std::vector<cv::Rect> bboxes;
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Не удалось открыть файл: " << filename << std::endl;
        return bboxes;
    }

    int x, y, width, height;

    while (ifs >> x >> y >> width >> height) {
        bboxes.push_back(cv::Rect(x, y, width, height));
    }

    return bboxes;
}

void create_detection_results(const std::string& input_lst_path, const std::string& output_dir) {
    const std::vector<std::filesystem::path> path_list = get_list_of_file_paths(input_lst_path);

    for (const auto& path : path_list) {
        const cv::Mat img = cv::imread(path.string());
        if (img.empty()) {
            std::cerr << "Failed to load image: " << path.filename() << std::endl;
            continue;
        }

        const std::vector<double> scales = {2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5};
        const std::vector<Detection> detections = detect(img, scales);

        const cv::Mat overlay = visualize_detection(img, detections);

        const std::string base_name = path.stem().string();
        const std::string overlay_out_path = output_dir + "/" + base_name + "_detect_overlay.png";
        const std::string boxes_out_path = output_dir + "/" + base_name + "_detections.txt";

        cv::imwrite(overlay_out_path, overlay);

        std::cout << "Saved detection results for "
                  << path.filename()
                  << " to " << output_dir << std::endl;
    }
}

void assess_detection_results(const std::string& input_img_path, const std::string& input_gt_detections_path, const std::string& output_dir) {
    const cv::Mat img = cv::imread(input_img_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << input_img_path << std::endl;
        return;
    }

    const std::vector<double> scales = {2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5};
    const std::vector<Detection> detections = detect(img, scales);

    const std::vector<cv::Rect> bboxes = read_bboxes_from_file(input_gt_detections_path);
    const double iou = calc_mean_iou_by_bbox(detections, bboxes);

    const cv::Mat overlay = visualize_detection_with_gt(img, detections, bboxes);

    const std::string overlay_out_path = output_dir + "/gt_detect_overlay.png";

    cv::imwrite(overlay_out_path, overlay);
    std::cout << "Saved gt overlay for "
              << input_img_path
              << " ( IoU = " << iou
              << " )" << std::endl;
}


int main(const int argc, const char * const argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << std::endl
                    << "Detection: " << argv[0] << " --detect <input_lst_path> <output_dir>"
                    << "Assess results: " << argv[0] << "--assess <input_img_path> <input_gt_detections_path> <output_dir>"
                    << std::endl;
        return 1;
    }

    const std::string mode = argv[1];

    if (mode == "--detect") {
        create_detection_results(argv[2], argv[3]);
    } else if ( mode == "--assess") {
        assess_detection_results(argv[2], argv[3], argv[4]);
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }

    return 0;
}