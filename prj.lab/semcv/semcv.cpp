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

cv::Mat gen_gray_bars(const int width, const int height, const int stripe_w) {
    cv::Mat img(height, width, CV_8UC1, cv::Scalar(0));
    const int num_stripes = width / stripe_w;
    for (int sx = 0; sx < num_stripes; ++sx) {
        const double t = sx / (num_stripes - 1.0);
        const int val = std::round(t * 255);
        const int x0 = sx * stripe_w;
        const int x1 = std::min(width, x0 + stripe_w);
        const cv::Point point1(x0, 0);
        const cv::Point point2(x1 - 1, height - 1);
        cv::rectangle(img, point1, point2, cv::Scalar(val), cv::FILLED);
    }
    return img;
}

cv::Mat gamma_correct(const cv::Mat& img, const double gamma) {
    cv::Mat lut(1, 256, CV_8UC1);
    uchar* lut_ptr = lut.ptr();
    for (int i = 0; i < 256; ++i) {
        const double value = i / 255.0;
        const double out = std::pow(value, 1.0 / gamma);
        const uchar iv = cv::saturate_cast<uchar>(std::clamp(std::round(out * 255), 0.0 , 255.0));
        lut_ptr[i]= iv;
    }
    cv::Mat res;
    cv::LUT(img, lut, res);
    return res;
}

cv::Mat gen_tgtimg00(int lev0, int lev1, int lev2) {
    cv::Mat img(IMG_SIZE, IMG_SIZE, CV_8UC1, cv::Scalar(lev0));
    constexpr int square_x = (256 - SQ_SIZE) / 2;
    constexpr int square_y = (256 - SQ_SIZE) / 2;
    const cv::Point center(IMG_SIZE / 2, IMG_SIZE / 2);
    cv::rectangle(img, cv::Rect(square_x, square_y, SQ_SIZE, SQ_SIZE), cv::Scalar(lev1), cv::FILLED);
    cv::circle(img, center, RADIUS, cv::Scalar(lev2), cv::FILLED);
    return img;
}

cv::Mat add_noise_gau(const cv::Mat &img, int sigma) {
    if (sigma == 0) return img.clone();

    cv::Mat noise(img.rows, img.cols, CV_32FC1);
    cv::randn(noise, 0.0, sigma);

    cv::Mat img32f;
    img.convertTo(img32f, CV_32F);

    cv::Mat noisy32f = img32f + noise;

    cv::Mat result;
    noisy32f.convertTo(result, CV_8U);

    return result;
}

cv::Mat draw_histogram_8u(const cv::Mat& img, const cv::Scalar& background_color, const cv::Scalar& bar_color) {
    constexpr int hist_size = 256;
    float range[] = {0.f, 256.f};
    const float* ranges[] = { range };
    cv::Mat hist;
    cv::calcHist(&img, 1, nullptr, cv::Mat(), hist, 1, &hist_size, ranges, true, false);

    double max_val = 0.0;
    cv::minMaxLoc(hist, nullptr, &max_val);
    cv::Mat normal_hist;
    if (max_val > 0) {
        normal_hist = hist * (250.0 / max_val);
    } else {
        normal_hist = cv::Mat::zeros(hist.size(), hist.type());
    }

    cv::Mat canvas(hist_size, hist_size, CV_8UC3, cv::Scalar(background_color));
    for (int i = 0; i < hist_size; ++i) {
        const int h = cvRound(normal_hist.at<float>(i));
        if (h <= 0) continue;
        cv::rectangle(canvas,
                      cv::Point(i, 255),
                      cv::Point(i, 255 - std::min(h, 250)),
                      bar_color,
                      cv::FILLED);
    }
    return canvas;
}

PixelDistributionStats calc_distribution_stats(const cv::Mat& img, const cv::Mat& mask) {
    PixelDistributionStats s{};
    double sum = 0.0, sum2 = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    int count = 0;

    const int rows = img.rows, cols = img.cols;
    for (int y = 0; y < rows; ++y) {
        const uchar* img_column_ptr = img.ptr(y);
        const uchar* mask_column_ptr = mask.empty() ? nullptr : mask.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x) {
            if (mask_column_ptr && mask_column_ptr[x] == 0) continue;
            const double value = img_column_ptr[x];
            sum += value;
            sum2 += value * value;
            if (value < min) min = value;
            if (value > max) max = value;
            ++count;
        }
    }

    if (count == 0) {
        s.count = 0;
        s.mean = s.variance = s.stddev = s.minimum = s.maximum = std::numeric_limits<double>::quiet_NaN();
        return s;
    }

    const double mean = sum / count;
    const double variance = std::max(0.0, sum2 / count - mean * mean); // дисперсия (несмещённая можно по желанию)
    const double stddev = std::sqrt(variance);

    s.count = count;
    s.mean = mean;
    s.variance = variance;
    s.stddev = stddev;
    s.minimum = min;
    s.maximum = max;
    return s;
}

cv::Mat to_grayscale(const cv::Mat& img) {
    if (img.channels() == 1) return img;

    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    } else {
        std::vector<cv::Mat> channels;
        cv::split(img, channels);
        gray = channels[0].clone();
    }
    return gray;
}

cv::Mat global_binarization(const cv::Mat &img, const int threshold, const int maxVal) {
    CV_Assert(img.type() == CV_8UC1);

    cv::Mat binary;
    cv::threshold(img, binary, threshold, maxVal, cv::THRESH_BINARY);
    return binary;
}


cv::Mat overlay_mask(const cv::Mat& img, const cv::Mat& mask, const cv::Scalar& mask_color, const double alpha) {
    CV_Assert(img.size() == mask.size());
    CV_Assert(mask.type() == CV_8UC1);

    cv::Mat img_color;
    if (img.channels() == 1) {
        cv::cvtColor(img, img_color, cv::COLOR_GRAY2BGR);
    } else {
        img_color = img.clone();
    }

    cv::Mat colored_mask;
    cv::cvtColor(mask, colored_mask, cv::COLOR_GRAY2BGR);
    colored_mask.setTo(mask_color, mask > 0);

    cv::Mat result;
    cv::addWeighted(img_color, 1.0 - alpha, colored_mask, alpha, 0.0, result);

    return result;
}

BinaryClassificationMetrics calc_binary_metrics(const cv::Mat& predicted_mask, const cv::Mat& mask) {
    CV_Assert(predicted_mask.size() == mask.size());
    CV_Assert(predicted_mask.type() == CV_8UC1);
    CV_Assert(mask.type() == CV_8UC1);

    BinaryClassificationMetrics metrics{};

    const int rows = predicted_mask.rows;
    const int cols = predicted_mask.cols;

    for (int y = 0; y < rows; ++y) {
        const uchar* predicted_ptr = predicted_mask.ptr<uchar>(y);
        const uchar* mask_ptr = mask.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x) {
            const bool predicted_pixel = (predicted_ptr[x] > 0);
            const bool mask_pixel = (mask_ptr[x] > 0);

            if (predicted_pixel && mask_pixel) {
                metrics.TP++;
            } else if (predicted_pixel && !mask_pixel) {
                metrics.FP++;
            } else if (!predicted_pixel && mask_pixel) {
                metrics.FN++;
            } else {
                metrics.TN++;
            }
        }
    }

    return metrics;
}