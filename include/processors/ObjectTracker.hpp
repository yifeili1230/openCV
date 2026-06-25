#pragma once

#include <limits>
#include <map>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "core/IFrameProcessor.hpp"

namespace video_engine {

class ObjectTracker : public IFrameProcessor {
public:
    explicit ObjectTracker(int max_distance = 80, int max_age = 8, int trail_length = 25)
        : max_distance_(max_distance), max_age_(max_age), trail_length_(trail_length) {}

    void process(FrameContext& ctx) override {
        std::vector<cv::Point> centroids;
        for (const auto& detection : ctx.detections) {
            cv::Point center(detection.bbox.x + detection.bbox.width / 2,
                             detection.bbox.y + detection.bbox.height / 2);
            centroids.push_back(center);
        }

        std::vector<Detection> matched_detections;
        std::vector<std::vector<cv::Point>> updated_trails;
        std::map<int, cv::Point> active_tracks;
        for (const auto& [id, point] : tracks_) {
            if (age_[id] <= max_age_) {
                active_tracks[id] = point;
            }
        }

        for (size_t i = 0; i < centroids.size(); ++i) {
            int best_id = -1;
            double best_distance = std::numeric_limits<double>::max();
            for (const auto& [id, point] : active_tracks) {
                double dist = cv::norm(centroids[i] - point);
                if (dist < best_distance && dist < max_distance_) {
                    best_distance = dist;
                    best_id = id;
                }
            }

            if (best_id >= 0) {
                matched_detections.push_back(Detection{ctx.detections[i].bbox, best_id});
                tracks_[best_id] = centroids[i];
                age_[best_id] = 0;
                if (trails_.count(best_id) == 0) {
                    trails_[best_id] = {};
                }
                trails_[best_id].push_back(centroids[i]);
                if (static_cast<int>(trails_[best_id].size()) > trail_length_) {
                    trails_[best_id].erase(trails_[best_id].begin());
                }
            } else {
                int new_id = next_id_++;
                matched_detections.push_back(Detection{ctx.detections[i].bbox, new_id});
                tracks_[new_id] = centroids[i];
                age_[new_id] = 0;
                trails_[new_id] = {centroids[i]};
            }
        }

        for (auto& [id, point] : tracks_) {
            ++age_[id];
            if (age_[id] > max_age_) {
                tracks_.erase(id);
                age_.erase(id);
                trails_.erase(id);
            }
        }

        ctx.detections = matched_detections;
        ctx.trails.clear();
        for (const auto& [id, trail] : trails_) {
            ctx.trails.push_back(trail);
        }
    }

    std::string name() const override {
        return "tracker";
    }

private:
    int max_distance_;
    int max_age_;
    int trail_length_;
    int next_id_ = 1;
    std::map<int, cv::Point> tracks_;
    std::map<int, int> age_;
    std::map<int, std::vector<cv::Point>> trails_;
};

}  // namespace video_engine
