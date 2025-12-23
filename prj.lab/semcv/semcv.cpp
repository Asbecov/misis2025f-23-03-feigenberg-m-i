#include <fstream>
#include <utility>
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
    int sum = 0, sum2 = 0;
    int min = std::numeric_limits<int>::infinity();
    int max = -std::numeric_limits<int>::infinity();
    int count = 0;

    const int rows = img.rows, cols = img.cols;
    for (int y = 0; y < rows; ++y) {
        const uchar* img_column_ptr = img.ptr(y);
        const uchar* mask_column_ptr = mask.empty() ? nullptr : mask.ptr<uchar>(y);
        for (int x = 0; x < cols; ++x) {
            if (mask_column_ptr && mask_column_ptr[x] == 0) continue;
            const uchar value = img_column_ptr[x];
            sum += value;
            sum2 += value * value;
            if (value < min) min = value;
            if (value > max) max = value;
            ++count;
        }
    }

    if (count == 0) {
        s.count = s.minimum = s.maximum = 0;
        s.mean = s.variance = s.stddev = std::numeric_limits<double>::quiet_NaN();
        return s;
    }

    const double mean = static_cast<double>(sum) / count;
    const double variance = std::max(0.0, static_cast<double>(sum2) / count - mean * mean);
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


cv::Mat create_segmentation_mask(const cv::Mat& img, const double threshold_fg) {
    const cv::Mat gray = to_grayscale(img);

    cv::Mat binary_image;
    cv::threshold(gray, binary_image, 0, 255, cv::THRESH_OTSU + cv::THRESH_BINARY_INV);

    const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
    cv::morphologyEx(binary_image, binary_image, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 3);

    cv::Mat sure_bg;
    cv::dilate(binary_image, sure_bg, kernel, cv::Point(-1, -1), 3);

    cv::Mat dist;
    cv::distanceTransform(binary_image, dist, cv::DIST_L2, 5);

    double max_val;
    cv::minMaxLoc(dist, nullptr, &max_val);

    cv::Mat sure_fg;
    cv::threshold(dist, sure_fg, threshold_fg * max_val, 255, cv::THRESH_BINARY);
    sure_fg.convertTo(sure_fg, CV_8U);

    cv::Mat unknown;
    cv::subtract(sure_bg, sure_fg, unknown);

    cv::Mat markers;
    cv::connectedComponents(sure_fg, markers);
    markers += 1;
    markers.setTo(0, unknown == 255);

    cv::watershed(img, markers);

    return markers;
}

cv::Mat overlay_segmentation(const cv::Mat& img, const cv::Mat& mask) {
    CV_Assert(mask.type() == CV_32S);  // маска после watershed
    CV_Assert(img.size() == mask.size());

    cv::Mat result;
    if (img.channels() == 1) {
        cv::cvtColor(img, result, cv::COLOR_GRAY2BGR);
    } else {
        result = img.clone();
    }

    std::vector<cv::Vec3b> colors;
    int n_labels = 0;
    mask.forEach<int>([&](const int val, const int* pos) { if (val > n_labels) n_labels = val; });
    colors.resize(n_labels + 1);
    colors[0] = cv::Vec3b(0, 0, 0);
    for (int i = 1; i <= n_labels; ++i) {
        colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
    }

    for (int y = 0; y < mask.rows; ++y) {
        const int* mask_ptr = mask.ptr<int>(y);
        cv::Vec3b* res_ptr = result.ptr<cv::Vec3b>(y);
        for (int x = 0; x < mask.cols; ++x) {
            int idx = mask_ptr[x];
            if (idx > 0) {
                res_ptr[x] = colors[idx];
            } else if (idx == -1) {
                res_ptr[x] = cv::Vec3b(0, 0, 255);
            }
        }
    }

    cv::addWeighted(result, 0.5, img, 0.5, 0, result);

    return result;
}

SegmentationMetrics calc_segmentation_metrics(const cv::Mat& predicted_markers, const std::vector<cv::Mat>& gt_masks, const double iou_threshold) {
    CV_Assert(predicted_markers.type() == CV_32S);

    SegmentationMetrics result;

    std::map<int, cv::Mat> pred_instances_map;
    for (int y = 0; y < predicted_markers.rows; ++y) {
        const int* row_ptr = predicted_markers.ptr<int>(y);

        for (int x = 0; x < predicted_markers.cols; ++x) {
            int label = row_ptr[x];
            if (label <= 0) continue;

            if (!pred_instances_map.contains(label)) pred_instances_map[label] = cv::Mat::zeros(predicted_markers.size(), CV_8UC1);
            pred_instances_map[label].at<uchar>(y, x) = 255;
        }
    }

    std::vector<cv::Mat> pred_instances;
    for (auto it = pred_instances_map.begin(); it != pred_instances_map.end(); ++it) {
        if (cv::countNonZero(it->second) > 20) pred_instances.push_back(it->second);
    }

    const size_t pred_size = pred_instances.size();
    const size_t gt_size   = gt_masks.size();

    struct IoUPair {
        size_t pred_idx;
        size_t gt_idx;
        double iou;
        BinaryClassificationMetrics metrics;
    };

    std::vector<IoUPair> pairs;

    for (size_t i = 0; i < pred_size; ++i) {
        for (size_t j = 0; j < gt_size; ++j) {
            BinaryClassificationMetrics metrics = calc_binary_metrics(pred_instances[i], gt_masks[j]);
            if (const double iou = metrics.IoU(); iou >= iou_threshold) pairs.push_back(IoUPair{i, j, metrics.IoU(), metrics});
        }
    }

    std::ranges::sort(pairs, [](const IoUPair& a, const IoUPair& b) { return a.iou > b.iou; });

    std::vector<bool> pred_used(pred_size, false);
    std::vector<bool> gt_used(gt_size, false);

    double sum_iou = 0.0;
    double sum_precision = 0.0;
    double sum_recall = 0.0;
    double sum_accuracy = 0.0;

    for (const auto& [pred_idx, gt_idx, iou, metrics] : pairs) {
        if (pred_used[pred_idx] || gt_used[gt_idx]) continue;
        if (iou < iou_threshold) continue;

        pred_used[pred_idx] = true;
        gt_used[gt_idx] = true;

        result.TP_instances++;

        sum_iou += iou;
        sum_precision += metrics.Precision();
        sum_recall += metrics.TPR();
        sum_accuracy += metrics.Accuracy();
    }

    result.FP_instances = pred_size - result.TP_instances;
    result.FN_instances = gt_size - result.TP_instances;

    if (gt_size > 0) {
        result.mean_iou      = sum_iou / gt_size;
        result.mean_recall   = sum_recall / gt_size;
        result.mean_accuracy = sum_accuracy / gt_size;
    } else {
        result.mean_iou = result.mean_recall = result.mean_accuracy = 0.0;
    }

    if (pred_size > 0) {
        result.mean_precision = sum_precision / pred_size;
    } else {
        result.mean_precision = 0.0;
    }

    return result;
}

static cv::Rect mask_to_bbox(const cv::Mat& m) {
    CV_Assert(m.type() == CV_8UC1);

    std::vector<cv::Point> pts;
    cv::findNonZero(m, pts);

    if (pts.empty()) return {};
    return cv::boundingRect(pts);
}

std::vector<Detection> detect(const cv::Mat &img, const std::vector<double>& scales) {
    CV_Assert(!img.empty());

    cv::Mat gray = to_grayscale(img);

    cv::Mat gx, gy, grad;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, grad);


    double mx;
    cv::minMaxLoc(grad, nullptr, &mx);
    double gradMax = std::max(1e-6, mx);

    struct ScaleResult {
        double quality = 0.0;
        std::vector<Detection> detections;
    };

    std::vector<ScaleResult> scale_results;

    for (const double scale : scales) {
        cv::Mat scaled = img.clone();
        if (scale != 1.0) cv::resize(img, scaled, cv::Size(), scale, scale, cv::INTER_AREA);

        cv::Mat markers = create_segmentation_mask(scaled);

        std::map<int, cv::Mat> instances_map;
        for (int y = 0; y < markers.rows; ++y) {
            const int* row = markers.ptr<int>(y);
            for (int x = 0; x < markers.cols; ++x) {
                const int label = row[x];
                if (label <= 0) continue;

                if (!instances_map.contains(label)) instances_map[label] = cv::Mat::zeros(markers.size(), CV_8UC1);
                instances_map[label].at<uchar>(y, x) = 255;
            }
        }

        ScaleResult sr;

        for (auto& it_instances : instances_map) {
            const cv::Mat img_scaled = it_instances.second;
            if (cv::countNonZero(img_scaled) < 30) continue;

            cv::Mat new_img;
            if (scale != 1.0) cv::resize(img_scaled, new_img, img.size(), 0, 0, cv::INTER_NEAREST);
            else new_img = std::move(img_scaled);
            const int area = cv::countNonZero(new_img);
            if (area < 30) continue;

            const cv::Rect bbox = mask_to_bbox(new_img);
            if (bbox.area() <= 0) continue;

            cv::Mat er, boundary;
            const cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
            cv::erode(new_img, er, kernel);
            cv::subtract(new_img, er, boundary);

            double sumG = 0.0;
            int cntG = 0;
            for (int y = 0; y < boundary.rows; ++y) {
                const uchar* bp = boundary.ptr<uchar>(y);
                const float* gp = grad.ptr<float>(y);
                for (int x = 0; x < boundary.cols; ++x) {
                    if (bp[x]) {
                        sumG += gp[x];
                        ++cntG;
                    }
                }
            }
            const double boundary_contrast = (cntG > 0) ? (sumG / cntG) / gradMax : 0.0;

            const double compactness = std::min(1.0, area / static_cast<double>(bbox.area()));

            const double score = std::clamp(0.6 * boundary_contrast + 0.4 * compactness, 0.0, 1.0);
            if (score < 0.2) continue;

            Detection detection;
            detection.mask = std::move(new_img);
            detection.bbox = bbox;
            detection.score = score;

            sr.detections.emplace_back(detection);
        }

        if (!sr.detections.empty()) {
            double sum = 0.0;
            for (const auto& d : sr.detections) sum += d.score;
            sr.quality = sum / sr.detections.size();
        }

        scale_results.push_back(std::move(sr));
    }

    auto best_it = std::ranges::max_element(scale_results, [](const ScaleResult& a, const ScaleResult& b) { return a.quality < b.quality; });

    if (best_it == scale_results.end()) return {};

    std::vector<Detection>& result = best_it->detections;
    std::ranges::sort(result, [](const Detection& a, const Detection& b) {return a.score > b.score;});

    return result;
}

cv::Mat visualize_detection(const cv::Mat &img, const std::vector<Detection> &detections, const double alpha) {
    CV_Assert(!img.empty());

    for (size_t i = 0; i < detections.size(); ++i) {
        const Detection detection = detections[i];

        cv::Scalar color(
           50 + (i * 97)  % 205,
           50 + (i * 151) % 205,
           50 + (i * 211) % 205
       );

        if (!detection.mask.empty()) {
            CV_Assert(detection.mask.size() == img.size());

            cv::Mat colored(img.size(), img.type(), color);
            cv::Mat mask_bin;
            cv::threshold(detection.mask, mask_bin, 0, 255, cv::THRESH_BINARY);

            cv::Mat blended;
            cv::addWeighted(img, 1.0 - alpha, colored, alpha, 0.0, blended);
            blended.copyTo(img, mask_bin);
        }

        cv::rectangle(img, detection.bbox, color, 2);
    }

    return img;
}


