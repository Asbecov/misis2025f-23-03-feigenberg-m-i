#include <opencv2/opencv.hpp>
#include <semcv/semcv.hpp>
#include <iostream>
#include <fstream>
#include <array>
#include <iomanip>
#include <ranges>

struct RegionMasks {
    cv::Mat background;
    cv::Mat square;
    cv::Mat circle;
};

static RegionMasks generate_masks() {
    RegionMasks masks;
    masks.background = cv::Mat::zeros(IMG_SIZE, IMG_SIZE, CV_8UC1);
    masks.square = cv::Mat::zeros(IMG_SIZE, IMG_SIZE, CV_8UC1);
    masks.circle = cv::Mat::zeros(IMG_SIZE, IMG_SIZE, CV_8UC1);

    constexpr int square_x = (IMG_SIZE - SQ_SIZE) / 2;
    constexpr int square_y = (IMG_SIZE - SQ_SIZE) / 2;
    const cv::Rect sq_rect(square_x, square_y, SQ_SIZE, SQ_SIZE);
    const cv::Point center(IMG_SIZE / 2, IMG_SIZE / 2);

    cv::circle(masks.circle, center, RADIUS, cv::Scalar(255), cv::FILLED);

    masks.square(sq_rect).setTo(cv::Scalar(255));
    cv::circle(masks.square, center, RADIUS, cv::Scalar(0), cv::FILLED);

    masks.background.setTo(cv::Scalar(255));
    masks.background(sq_rect).setTo(cv::Scalar(0));

    return masks;
}

void gen_save_stats(const cv::Mat& img, const std::array<int, 3>& brightness, const int sigma, const std::string& path) {
    std::ofstream ofs(path, std::ios::app);
    if (!ofs) {
        std::cerr << "Could not open file " << path << std::endl;
        return;
    }

    const double theory_variance = std::pow(sigma, 2);
    const RegionMasks masks = generate_masks();
    ofs << "Stats for the following parameters: brightness {"
        << brightness[0] << ", " << brightness[1]
        << ", " << brightness[2]
        << "}, sigma " << sigma << std::endl;

    ofs << "name, theory_level, mean_exp, diff_mean, var_theory, var_exp, diff_var, count, stddev, min, max" << std::endl;

    const std::array<std::pair<std::string, cv::Mat>, 3> masks_names = {
        {
            {"Background", masks.background},
            {"Square", masks.square},
            {"Circle", masks.circle}
        }
    };

    int iteration_count = 0;
    for (const auto [name, mask] : masks_names) {
        const auto [count, mean, variance, stddev, minimum, maximum] = calc_distribution_stats(img, mask);
        const double theory_level = brightness[iteration_count];
        double mean_diff = mean - theory_level;
        double var_diff = variance - theory_level;

        ofs << name << " "
            << std::fixed << std::setprecision(3)
            << theory_level << ", "
            << mean << ", "
            << mean_diff << ", "
            << theory_variance << ", "
            << variance << ", "
            << var_diff << ", "
            << count << ", "
            << stddev << ", "
            << minimum << ", "
            << maximum <<  std::endl;
        iteration_count++;
    }
    ofs << std::endl;
    ofs.close();
    std::cout << "Saved stats" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: task02 <output_image_path> [--hist]\n";
        return 1;
    }
    const std::string path = argv[1];
    const bool write_hist = (argc >= 3 && std::string(argv[2]) == "--hist");

    constexpr std::array<std::array<int, 3>,4> levels = {
        {{{0, 127, 255}},
        {{20, 127, 235}},
        {{55, 127, 200}},
        {{90, 127, 165}}
        }
    };
    constexpr std::array<int,3> sigmas = {3,7,15};

    std::string save_path = path.substr(0,  path.find_last_of('.')) + "_stats.txt";
    std::vector<cv::Mat> samples;
    std::vector<std::vector<cv::Mat>> noise_samples;
    for (auto lv : levels) {
        const cv::Mat image = gen_tgtimg00(lv[0], lv[1], lv[2]);
        gen_save_stats(image, lv, 0, save_path);
        std::vector<cv::Mat> noise;
        for (const int sigma : sigmas) {
            const cv::Mat noisy_image = add_noise_gau(image, sigma);
            gen_save_stats(noisy_image, lv, sigma, save_path);
            noise.emplace_back(noisy_image);
        }
        samples.emplace_back(image);
        noise_samples.emplace_back(noise);
    }

    cv::Mat sample_base;
    cv::hconcat(samples, sample_base);

    std::vector<cv::Mat> noise_bases;
    for (auto noise : noise_samples) {
        cv::Mat noise_base;
        cv::vconcat(noise, noise_base);
        noise_bases.emplace_back(noise_base);
    }

    cv::Mat noise_base;
    cv::hconcat(noise_bases, noise_base);

    cv::Mat final_image;
    cv::vconcat(sample_base, noise_base, final_image);

    if (!cv::imwrite(path, final_image)) {
        std::cerr << "failed to save: " << path << "\n";
        return 1;
    }

    if (write_hist) {
        std::vector<cv::Mat> hist_rows;

        auto draw_row = [](const std::vector<cv::Mat>& imgs_row) {
            std::vector<cv::Mat> hists;
            bool toggle_bg = false;
            for (const auto& img : imgs_row) {
                cv::Scalar bg = toggle_bg ? cv::Scalar(230,230,230) : cv::Scalar(210,210,210);
                toggle_bg = !toggle_bg;
                hists.push_back(draw_histogram_8u(img, bg, cv::Scalar(32,32,32)));
            }
            cv::Mat row;
            cv::hconcat(hists, row);
            return row;
        };

        hist_rows.push_back(draw_row(samples));

        for (size_t sigma_idx = 0; sigma_idx < sigmas.size(); ++sigma_idx) {
            std::vector<cv::Mat> row_imgs;
            for (size_t tile_idx = 0; tile_idx < noise_samples.size(); ++tile_idx) {
                row_imgs.push_back(noise_samples[tile_idx][sigma_idx]);
            }
            hist_rows.push_back(draw_row(row_imgs));
        }

        cv::Mat hist_final;
        cv::vconcat(hist_rows, hist_final);

        std::string hist_path = path.substr(0, path.find_last_of('.')) + "_hist.png";
        if (!cv::imwrite(hist_path, hist_final)) {
            std::cerr << "Failed to save histogram image: " << hist_path << std::endl;
        } else {
            std::cout << "Saved histogram image to " << hist_path << std::endl;
        }
    }

    std::cout << "Saved image to " << path << std::endl;
    return 0;
}