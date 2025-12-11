#include <iostream>
#include <opencv2/opencv.hpp>

#include <semcv/semcv.hpp>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path to .lst file>" << std::endl;
        return 2;
    }
    const std::vector<std::filesystem::path> list = get_list_of_file_paths(argv[1]);
    for (std::filesystem::path const &path : list) {
        const std::string file_name = path.filename().string();
        const cv::Mat img = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            std::cout << file_name << "\t" << "bad, cannot open" << std::endl;
            continue;
        }
        const std::string actual = strid_from_mat(img);
        const std::string stem = path.stem().string();
        if (actual == stem) {
            std::cout << file_name << "\tgood" << std::endl;
        }
        else {
            std::cout << file_name << "\tbad, should be " << actual << std::endl;
        }
    }
    return 0;
}