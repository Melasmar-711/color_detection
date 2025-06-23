#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/PolygonStamped.h>
#include <std_msgs/String.h>
#include <opencv2/opencv.hpp>
#include <json/json.h>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <array>
#include <limits>
#include <algorithm>

using namespace message_filters;

class ColorDetector {
public:
    // Constructor expects exactly three JSON file paths
    ColorDetector(const std::vector<std::string>& hist_paths) : it_(nh_) {
        // 1) Read histogramâ€voting parameters from ROS param server (private namespace)
        ros::NodeHandle pnh("~");
        pnh.param("hist_threshold",     hist_threshold_,     0.9);
        pnh.param("min_votes_required", min_votes_required_, 2);

        // 2) Load all three JSON histogram files
        loadMultipleHistograms(hist_paths);

        // 3) Now that we know which colors exist, assign perâ€color channel weights.
        //    The weight order is: { w_HS2D, w_H1D, w_S1D, w_V1D }.
        //    You can adjust these numbers however you like.
        for (const auto& kv : hist_data_.color_hists) {
            const std::string& color_name = kv.first;
            // Example defaults; edit as needed:
            if (color_name == "red") {
                channel_weights_[color_name] = {1,  1,  1,  0.4};  // 0.4 perfect on real
            }
            else if (color_name == "white") {
                channel_weights_[color_name] = {1,  1,  1,  2};    // 2 perfect on real
            }
            else if (color_name == "yellow") {
                channel_weights_[color_name] = {1,  1,  1,  0.4};   //0.7 perfect on real 
            }
            else {
                // Any additional colors (if youâ€™ve appended suffixes), give them a default:
                channel_weights_[color_name] = {1.0, 1.0, 1.0, 1.0};
            }
        }

        // 4) Initialize subscribers / synchronizer / publisher
        initializeROSComponents();

        // 5) Create display windows
        cv::namedWindow("Detection Result", cv::WINDOW_NORMAL);
        cv::namedWindow("Circle Mask",      cv::WINDOW_NORMAL);
    }

private:
    struct ColorHistograms {
        cv::Mat hs2d;  // 2D HÃ—S reference histogram
        cv::Mat h1d;   // 1D H reference histogram
        cv::Mat s1d;   // 1D S reference histogram
        cv::Mat v1d;   // 1D V reference histogram
    };

    struct HistogramData {
        std::array<int, 2> bins_2d;
        std::array<std::array<float, 2>, 2> ranges_2d;
        std::vector<int> bins_1d;
        std::vector<std::array<float, 2>> ranges_1d;
        std::map<std::string, ColorHistograms> color_hists;
        bool initialized = false;
    } hist_data_;

    double hist_threshold_;
    int    min_votes_required_;
    const float ROI_SCALE = 1.0f;

    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    message_filters::Subscriber<sensor_msgs::Image>       image_sub_;
    message_filters::Subscriber<geometry_msgs::PolygonStamped> boxes_sub_;
    typedef sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PolygonStamped> SyncPolicy;
    typedef Synchronizer<SyncPolicy> Sync;
    boost::shared_ptr<Sync> sync_;
    ros::Publisher colors_pub_;

    // Perâ€color channel weights { w_HS2D, w_H1D, w_S1D, w_V1D }
    std::map<std::string, std::array<double, 4>> channel_weights_;

    // ------------------------------------------------------------
    // 1) Initialize ROS subscribers, synchronizer, and publisher
    void initializeROSComponents() {
        image_sub_.subscribe(nh_, "/camera/image_raw", 1);
        boxes_sub_.subscribe(nh_, "/detected_boxes",   1);
        sync_.reset(new Sync(SyncPolicy(10), image_sub_, boxes_sub_));
        sync_->registerCallback(boost::bind(&ColorDetector::detectionCallback, this, _1, _2));
        colors_pub_ = nh_.advertise<std_msgs::String>("/object_colors", 10);
    }

    // ------------------------------------------------------------
    // 2) Load three JSON files into hist_data_.color_hists
    void loadMultipleHistograms(const std::vector<std::string>& paths) {
        if (paths.size() != 3) {
            throw std::runtime_error("Exactly three JSON paths must be provided");
        }
        for (const auto& path : paths) {
            loadHistogramsFromFile(path);
        }
        if (hist_data_.color_hists.empty()) {
            throw std::runtime_error("No histograms were loaded from any JSON.");
        }
        ROS_INFO("Total colors loaded: %zu", hist_data_.color_hists.size());
    }

    // Extract color name from filename â€œ<something>/<color>_histograms.jsonâ€
    std::string extractColorName(const std::string& filepath) {
        size_t slash_pos = filepath.find_last_of("/\\");
        std::string filename = (slash_pos == std::string::npos) ? filepath
                                                                 : filepath.substr(slash_pos + 1);
        const std::string suffix = "_histograms.json";
        if (filename.size() > suffix.size()
            && filename.substr(filename.size() - suffix.size()) == suffix)
        {
            return filename.substr(0, filename.size() - suffix.size());
        }
        size_t dotpos = filename.rfind(".json");
        if (dotpos != std::string::npos) {
            return filename.substr(0, dotpos);
        }
        return filename;
    }

    // Load a single JSON (â€œ<color>_histograms.jsonâ€) into hist_data_
    void loadHistogramsFromFile(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Histogram file not found: " + path);
        }
        Json::Value root;
        file >> root;

        // If first JSON: read bins/ranges; otherwise verify consistency
        if (!hist_data_.initialized) {
            hist_data_.bins_2d[0] = root["metadata"]["hist_2d_bins"][0].asInt();
            hist_data_.bins_2d[1] = root["metadata"]["hist_2d_bins"][1].asInt();
            hist_data_.ranges_2d[0][0] = root["metadata"]["ranges_2d"][0].asFloat();
            hist_data_.ranges_2d[0][1] = root["metadata"]["ranges_2d"][1].asFloat();
            hist_data_.ranges_2d[1][0] = root["metadata"]["ranges_2d"][2].asFloat();
            hist_data_.ranges_2d[1][1] = root["metadata"]["ranges_2d"][3].asFloat();

            hist_data_.bins_1d.push_back(root["metadata"]["hist_1d_bins"][0].asInt());
            hist_data_.bins_1d.push_back(root["metadata"]["hist_1d_bins"][1].asInt());
            hist_data_.bins_1d.push_back(root["metadata"]["hist_1d_bins"][2].asInt());

            for (int i = 0; i < 3; ++i) {
                std::array<float, 2> r = {
                    root["metadata"]["ranges_1d"][2 * i].asFloat(),
                    root["metadata"]["ranges_1d"][2 * i + 1].asFloat()
                };
                hist_data_.ranges_1d.push_back(r);
            }
            hist_data_.initialized = true;
        }
        else {
            // Verify 2D bins match
            int b2_0 = root["metadata"]["hist_2d_bins"][0].asInt();
            int b2_1 = root["metadata"]["hist_2d_bins"][1].asInt();
            if (b2_0 != hist_data_.bins_2d[0] || b2_1 != hist_data_.bins_2d[1]) {
                throw std::runtime_error("Mismatched 2D bins in " + path);
            }
            // Verify 2D ranges match
            float r2d0_min = root["metadata"]["ranges_2d"][0].asFloat();
            float r2d0_max = root["metadata"]["ranges_2d"][1].asFloat();
            float r2d1_min = root["metadata"]["ranges_2d"][2].asFloat();
            float r2d1_max = root["metadata"]["ranges_2d"][3].asFloat();
            if (r2d0_min != hist_data_.ranges_2d[0][0] ||
                r2d0_max != hist_data_.ranges_2d[0][1] ||
                r2d1_min != hist_data_.ranges_2d[1][0] ||
                r2d1_max != hist_data_.ranges_2d[1][1])
            {
                throw std::runtime_error("Mismatched 2D ranges in " + path);
            }

            // Verify 1D bins and ranges match
            for (int i = 0; i < 3; ++i) {
                int b1 = root["metadata"]["hist_1d_bins"][i].asInt();
                if (b1 != hist_data_.bins_1d[i]) {
                    throw std::runtime_error("Mismatched 1D bins in " + path);
                }
                float r1_min = root["metadata"]["ranges_1d"][2*i].asFloat();
                float r1_max = root["metadata"]["ranges_1d"][2*i + 1].asFloat();
                if (r1_min != hist_data_.ranges_1d[i][0] ||
                    r1_max != hist_data_.ranges_1d[i][1])
                {
                    throw std::runtime_error("Mismatched 1D ranges in " + path);
                }
            }
        }

        // Load this JSONâ€™s color histograms
        std::string color_name = extractColorName(path);
        const Json::Value& H = root["histograms"]["hs2d"];
        const Json::Value& h = root["histograms"]["h1d"];
        const Json::Value& s = root["histograms"]["s1d"];
        const Json::Value& v = root["histograms"]["v1d"];

        ColorHistograms ch;
        // Allocate & fill hs2d (size: bins_2d[0] Ã— bins_2d[1])
        ch.hs2d = cv::Mat(hist_data_.bins_2d[0], hist_data_.bins_2d[1], CV_32F);
        for (int i = 0; i < hist_data_.bins_2d[0]; ++i) {
            for (int j = 0; j < hist_data_.bins_2d[1]; ++j) {
                ch.hs2d.at<float>(i, j) = H[i][j].asFloat();
            }
        }
        cv::normalize(ch.hs2d, ch.hs2d, 1.0, 0.0, cv::NORM_L1);

        // Allocate & fill h1d (size: bins_1d[0])
        ch.h1d = cv::Mat(hist_data_.bins_1d[0], 1, CV_32F);
        for (int i = 0; i < hist_data_.bins_1d[0]; ++i) {
            ch.h1d.at<float>(i) = h[i].asFloat();
        }
        cv::normalize(ch.h1d, ch.h1d, 1.0, 0.0, cv::NORM_L1);

        // Allocate & fill s1d (size: bins_1d[1])
        ch.s1d = cv::Mat(hist_data_.bins_1d[1], 1, CV_32F);
        for (int i = 0; i < hist_data_.bins_1d[1]; ++i) {
            ch.s1d.at<float>(i) = s[i].asFloat();
        }
        cv::normalize(ch.s1d, ch.s1d, 1.0, 0.0, cv::NORM_L1);

        // Allocate & fill v1d (size: bins_1d[2])
        ch.v1d = cv::Mat(hist_data_.bins_1d[2], 1, CV_32F);
        for (int i = 0; i < hist_data_.bins_1d[2]; ++i) {
            ch.v1d.at<float>(i) = v[i].asFloat();
        }
        cv::normalize(ch.v1d, ch.v1d, 1.0, 0.0, cv::NORM_L1);

        // Store under a unique key if needed
        std::string uniqueName = color_name;
        int suffix = 1;
        while (hist_data_.color_hists.count(uniqueName)) {
            uniqueName = color_name + "_" + std::to_string(suffix++);
        }
        hist_data_.color_hists[uniqueName] = ch;

        ROS_INFO("Loaded histograms for color \"%s\" (file %s)",
                  uniqueName.c_str(), path.c_str());
    }

    // Build a 2D HÃ—S histogram from a query ROI (HSV) with a circular mask
    cv::Mat computeHS2DHist(const cv::Mat& hsv, const cv::Mat& mask) {
        std::vector<cv::Mat> hsv_ch;
        cv::split(hsv, hsv_ch);  // hsv_ch[0]=H, hsv_ch[1]=S, hsv_ch[2]=V

        cv::Mat hs_mat;
        const cv::Mat arrHS[2] = { hsv_ch[0], hsv_ch[1] };
        cv::merge(arrHS, 2, hs_mat);

        const int    channels[2] = {0, 1};
        const int    histSize[2] = { hist_data_.bins_2d[0], hist_data_.bins_2d[1] };
        const float  h_ranges[2] = { hist_data_.ranges_2d[0][0], hist_data_.ranges_2d[0][1] };
        const float  s_ranges[2] = { hist_data_.ranges_2d[1][0], hist_data_.ranges_2d[1][1] };
        const float* ranges[2]   = { h_ranges, s_ranges };

        cv::Mat hist2d;
        const cv::Mat images2D[] = { hs_mat };
        cv::calcHist(
            images2D, 1, channels,
            mask,
            hist2d,
            2, histSize, ranges,
            true, false
        );
        cv::normalize(hist2d, hist2d, 1.0, 0.0, cv::NORM_L1);
        return hist2d;
    }

    // Build a 1D histogram for one channel (0=H, 1=S, 2=V) with circular mask
    cv::Mat compute1DHist(const cv::Mat& hsv, int channel, int bins, const std::array<float,2>& range, const cv::Mat& mask) {
        std::vector<cv::Mat> hsv_ch;
        cv::split(hsv, hsv_ch);

        const cv::Mat single = hsv_ch[channel];
        const int    ch[1]    = {0};
        const int    histSize[1] = { bins };
        const float  r[2]     = { range[0], range[1] };
        const float* ranges[1] = { r };

        cv::Mat hist1d;
        const cv::Mat images1D[] = { single };
        cv::calcHist(
            images1D, 1, ch,
            mask,
            hist1d,
            1, histSize, ranges,
            true, false
        );
        cv::normalize(hist1d, hist1d, 1.0, 0.0, cv::NORM_L1);
        return hist1d;
    }

    // Compare two histograms by Bhattacharyya distance
    double compareHistogram(const cv::Mat& a, const cv::Mat& b) {
        return cv::compareHist(a, b, cv::HISTCMP_BHATTACHARYYA);
    }

    // Voting logic with tieâ€breaking by **perâ€color** weighted distance,
    // where Vâ€channel has the **lowest** weight.
std::string classifyROI(const cv::Mat& roi_bgr) {
    // Convert ROI to HSV
    cv::Mat hsv;
    cv::cvtColor(roi_bgr, hsv, cv::COLOR_BGR2HSV);

    // Build circular mask inside ROI
    int roi_h = hsv.rows;
    int roi_w = hsv.cols;
    cv::Mat mask = cv::Mat::zeros(roi_h, roi_w, CV_8UC1);
    cv::Point center(roi_w / 2, roi_h / 2);
    int radius = std::min(center.x, center.y);
    cv::circle(mask, center, radius, cv::Scalar(255), -1);

    // Show the circular mask
    cv::imshow("Circle Mask", mask);
    cv::waitKey(1);

    // Compute 4 query histograms with the mask
    cv::Mat q_hs2d = computeHS2DHist(hsv, mask);
    cv::Mat q_h1d  = compute1DHist(hsv, 0, hist_data_.bins_1d[0], hist_data_.ranges_1d[0], mask);
    cv::Mat q_s1d  = compute1DHist(hsv, 1, hist_data_.bins_1d[1], hist_data_.ranges_1d[1], mask);
    cv::Mat q_v1d  = compute1DHist(hsv, 2, hist_data_.bins_1d[2], hist_data_.ranges_1d[2], mask);

    // Variables for each color's channel distances
    double red_d_hs2d = 0, red_d_h1d = 0, red_d_s1d = 0, red_d_v1d = 0;
    double white_d_hs2d = 0, white_d_h1d = 0, white_d_s1d = 0, white_d_v1d = 0;
    double yellow_d_hs2d = 0, yellow_d_h1d = 0, yellow_d_s1d = 0, yellow_d_v1d = 0;

    std::string best_color = "unknown";
    int best_votes = 0;
    double best_metric = std::numeric_limits<double>::max();
    double min_total_dist = std::numeric_limits<double>::max();  // Track minimum total distance
    std::string min_dist_color;                                  // Color with min total distance

    ROS_INFO("---- Voting debug ----");

    // First pass: Compute distances and track minimum total distance
    for (const auto& kv : hist_data_.color_hists) {
        const std::string& color_name = kv.first;
        const ColorHistograms& ref = kv.second;
        const auto& w = channel_weights_[color_name];

        double d_hs2d = compareHistogram(q_hs2d, ref.hs2d);
        double d_h1d  = compareHistogram(q_h1d,  ref.h1d);
        double d_s1d  = compareHistogram(q_s1d,  ref.s1d);
        double d_v1d  = compareHistogram(q_v1d,  ref.v1d);
	double total_dist =0;

        // Store distances in variables for each color
        if (color_name == "red") {
            red_d_hs2d = d_hs2d;
            red_d_h1d  = d_h1d;
            red_d_s1d  = d_s1d;
            red_d_v1d  = d_v1d;
            total_dist = d_hs2d * w[0] +d_h1d  * w[1] +d_s1d  * w[2] +d_v1d  * w[3];
        } else if (color_name == "white") {
            white_d_hs2d = d_hs2d;
            white_d_h1d  = d_h1d;
            white_d_s1d  = d_s1d;
            white_d_v1d  = d_v1d;
            total_dist = d_hs2d * w[0] +d_h1d  * w[1] +d_s1d  * w[2] +d_v1d  * w[3];
        } else if (color_name == "yellow") {
            yellow_d_hs2d = d_hs2d;
            yellow_d_h1d  = d_h1d;
            yellow_d_s1d  = d_s1d;
            yellow_d_v1d  = d_v1d;
            total_dist = d_hs2d * w[0] +d_h1d  * w[1] +d_s1d  * w[2] +d_v1d  * w[3];
        }

        // Track minimum total distance
        if (total_dist < min_total_dist) {
            min_total_dist = total_dist;
            min_dist_color = color_name;
        }
    }

    // Second pass: Apply voting with extra vote condition
    for (const auto& kv : hist_data_.color_hists) {
        const std::string& color_name = kv.first;
        const ColorHistograms& ref = kv.second;
        const auto& w = channel_weights_[color_name];

        // Retrieve precomputed distances
        double d_hs2d, d_h1d, d_s1d, d_v1d;
        if (color_name == "red") {
            d_hs2d = red_d_hs2d;
            d_h1d  = red_d_h1d;
            d_s1d  = red_d_s1d;
            d_v1d  = red_d_v1d;
        } else if (color_name == "white") {
            d_hs2d = white_d_hs2d;
            d_h1d  = white_d_h1d;
            d_s1d  = white_d_s1d;
            d_v1d  = white_d_v1d;
        } else if (color_name == "yellow") {
            d_hs2d = yellow_d_hs2d;
            d_h1d  = yellow_d_h1d;
            d_s1d  = yellow_d_s1d;
            d_v1d  = yellow_d_v1d;
        }

        double total_dist = d_hs2d + d_h1d + d_s1d + d_v1d;
        int votes = 0;
        if (d_hs2d < hist_threshold_) ++votes;
        if (d_h1d  < hist_threshold_) ++votes;
        if (d_s1d  < hist_threshold_) ++votes;
        if (d_v1d  < hist_threshold_) votes+=2;

        // EXTRA VOTE: Award to color with smallest total distance
        if (color_name == min_dist_color) {
            votes += 1;
            ROS_INFO_STREAM("Extra vote given to " << color_name 
                            << " for smallest total distance: " << min_total_dist);
        }

        ROS_INFO_STREAM("Ref[" << color_name << "]: "
                          << "d_hs2d=" << d_hs2d << "  "
                          << "d_h1d="  << d_h1d  << "  "
                          << "d_s1d="  << d_s1d  << "  "
                          << "d_v1d="  << d_v1d  << "  "
                          << "votes="  << votes  << "  "
                          << "total_dist=" << total_dist);

        // [Rest of voting logic remains unchanged]
        if (votes >= min_votes_required_) {
            if (votes > best_votes) {
                best_votes  = votes;
                best_metric = total_dist;
                best_color  = color_name;
            } else if (votes == best_votes) {
                double weighted_dist =
                    d_hs2d * w[0] +
                    d_h1d  * w[1] +
                    d_s1d  * w[2] +
                    d_v1d  * w[3];

                ROS_INFO_STREAM("Ref[" << color_name << "] weighted_dist=" << weighted_dist
                                      << " (using weights " << w[0] << "," << w[1] << ","
                                      << w[2] << "," << w[3] << ")");

                if (weighted_dist < best_metric) {
                    best_metric = weighted_dist;
                    best_color  = color_name;
                }
            }
        }
    }
    ROS_INFO_STREAM("Final chosen: " << best_color
                    << " (votes=" << best_votes
                    << ", metric=" << best_metric << ")");

    // Now you can use red_d_hs2d, white_d_h1d, yellow_d_v1d, etc., for custom logic below
    // Example:
    // if (red_d_hs2d < 0.5 && red_d_v1d < 0.3) { ... }

    return best_color;
}


    // Use the full bounding box (no scaling)
    cv::Rect getScaledROI(const cv::Rect& orig) {
        return orig;
    }

    // For each detection: classify, draw, and publish
    void processDetections(cv::Mat& frame_bgr, const geometry_msgs::PolygonStampedConstPtr& boxes_msg) {
        std::string color_results;
        const auto& pts = boxes_msg->polygon.points;

        // Each box is 4 points in rowâ€major order
        for (size_t i = 0; i + 3 < pts.size(); i += 4) {
            int x0 = static_cast<int>(pts[i].x);
            int y0 = static_cast<int>(pts[i].y);
            int x2 = static_cast<int>(pts[i+2].x);
            int y2 = static_cast<int>(pts[i+2].y);
            int w  = x2 - x0;
            int h  = y2 - y0;
            cv::Rect box(x0, y0, w, h);

            // Clip to image bounds
            box &= cv::Rect(0, 0, frame_bgr.cols, frame_bgr.rows);
            if (box.area() < 25) continue;

            cv::Rect roi_box = getScaledROI(box);
            if (roi_box.area() < 1) continue;

            cv::Mat roi_bgr = frame_bgr(roi_box);
            std::string color = classifyROI(roi_bgr);

            // Draw bounding box and label
            cv::rectangle(frame_bgr, box, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame_bgr, color,
                        box.tl() + cv::Point(0, -5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7, cv::Scalar(0, 255, 0), 2);

            color_results += color + ",";
        }

        if (!color_results.empty()) {
            std_msgs::String msg;
            msg.data = color_results;
            colors_pub_.publish(msg);
        }
        cv::imshow("Detection Result", frame_bgr);
        cv::waitKey(1);
    }

    // Synchronized callback
    void detectionCallback(const sensor_msgs::ImageConstPtr& img_msg,
                           const geometry_msgs::PolygonStampedConstPtr& boxes_msg) {
        try {
            cv::Mat frame_bgr = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
            processDetections(frame_bgr, boxes_msg);
        } catch (const std::exception& e) {
            ROS_ERROR("Processing error: %s", e.what());
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "multi_histogram_voting");
    if (argc != 4) {
        ROS_ERROR("Usage: rosrun <your_package> multi_histogram_voting <hist1.json> <hist2.json> <hist3.json>");
        return 1;
    }
    std::vector<std::string> json_paths = { argv[1], argv[2], argv[3] };
    try {
        ColorDetector detector(json_paths);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL("Exception: %s", e.what());
        return 1;
    }
    return 0;
}
