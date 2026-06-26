#pragma once

#include <algorithm>  // Sorts candidate track matches by distance.
#include <map>        // Stores active tracks, ages, and trails by track ID.
#include <set>        // Tracks which detections and IDs are already assigned.
#include <vector>     // Stores centroids, candidates, and output detections.

#include <opencv2/imgproc.hpp>  // Provides cv::Point and cv::norm helpers.

#include "core/IFrameProcessor.hpp"  // Defines the frame processor interface.

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

        std::map<int, cv::Point> active_tracks;
        for (const auto& [id, point] : tracks_) {
            if (age_[id] <= max_age_) {
                active_tracks[id] = point;
            }
        }

        struct Candidate {
            size_t detection_index;
            int track_id;
            double distance;
        };

        std::vector<Candidate> candidates;
        for (size_t i = 0; i < centroids.size(); ++i) {
            for (const auto& [id, point] : active_tracks) {
                double dist = cv::norm(centroids[i] - point);
                if (dist < max_distance_) {
                    candidates.push_back(Candidate{i, id, dist});
                }
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
            return lhs.distance < rhs.distance;
        });

        std::vector<int> assigned_ids(centroids.size(), -1);
        std::set<size_t> used_detections;
        std::set<int> used_tracks;
        for (const auto& candidate : candidates) {
            if (used_detections.count(candidate.detection_index) > 0 ||
                used_tracks.count(candidate.track_id) > 0) {
                continue;
            }
            assigned_ids[candidate.detection_index] = candidate.track_id;
            used_detections.insert(candidate.detection_index);
            used_tracks.insert(candidate.track_id);
        }

        for (size_t i = 0; i < centroids.size(); ++i) {
            if (assigned_ids[i] < 0) {
                assigned_ids[i] = next_id_++;
            }
        }

        std::vector<Detection> matched_detections;
        std::set<int> updated_tracks;
        for (size_t i = 0; i < centroids.size(); ++i) {
            const int id = assigned_ids[i];
            matched_detections.push_back(Detection{ctx.detections[i].bbox, id});
            tracks_[id] = centroids[i];
            age_[id] = 0;
            trails_[id].push_back(centroids[i]);
            if (static_cast<int>(trails_[id].size()) > trail_length_) {
                trails_[id].erase(trails_[id].begin());
            }
            updated_tracks.insert(id);
        }

        std::vector<int> expired_ids;
        for (auto& [id, point] : tracks_) {
            if (updated_tracks.count(id) == 0) {
                ++age_[id];
            }
            if (age_[id] > max_age_) {
                expired_ids.push_back(id);
            }
        }
        for (int id : expired_ids) {
            tracks_.erase(id);
            age_.erase(id);
            trails_.erase(id);
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
