#include <fstream>
#include <opencv2/opencv.hpp>

#include <semcv/semcv.hpp>

static std::string depth_to_type(const int& depth) {
    switch (depth) {
        case CV_8U: return "uint08";
        case CV_8S: return "sint08";
        case CV_16U: return "uint16";
        case CV_16S: return "sint16";
        case CV_32S: return "sint32";
        case CV_32F: return "real32";
        case CV_64F: return "real64";
        default: return "unknown";
    }
}

std::string strid_from_mat(const cv::Mat& img, const int n) {
    const int w = img.cols;
    const int h = img.rows;
    const int c = img.channels();
    const int depth = img.depth();

    std::ostringstream ss;
    ss << std::setw(n) << std::setfill('0') << w
        << "x"
        << std::setw(n) << std::setfill('0') << h
        << "."
        << c
        << depth_to_type(depth);
    return ss.str();
}

std::vector<std::filesystem::path> get_list_of_file_paths(const std::filesystem::path& path_lst) {
    std::vector<std::filesystem::path> out;
    std::ifstream ifs(path_lst);
    if (!ifs) return out;
    std::string line;
    const std::filesystem::path base = path_lst.parent_path();
    while (std::getline(ifs, line)) {
        const std::size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        const std::size_t end = line.find_last_not_of(" \t\r\n");
        const std::string s = line.substr(start, end-start+1);
        std::filesystem::path p(s);
        if (p.is_relative()) p = base / p;
        out.push_back(std::filesystem::absolute(p));
    }
    return out;
}

