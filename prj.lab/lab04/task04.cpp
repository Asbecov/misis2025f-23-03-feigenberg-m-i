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

int main(const int argc, const char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input_path output_dir" << std::endl;
        return 1;
    }

    create_segment_masks(argv[1], argv[2]);
}