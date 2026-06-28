#pragma once

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include "analytics/AnalyticsTypes.hpp"

namespace video_engine {

class SquatSummaryWriter {
public:
    static std::filesystem::path outputPathForSource(
        const std::string& source, const std::filesystem::path& output_directory) {
        const std::string stem =
            source == "webcam" ? "webcam" : std::filesystem::path(source).stem().string();
        return output_directory / (stem.empty() ? "squat.json" : stem + ".json");
    }

    static std::string toJson(const SquatSessionSummary& summary) {
        std::ostringstream output;
        output << std::fixed << std::setprecision(3);
        output << "{\n";
        output << "  \"source\": \"" << escape(summary.source) << "\",\n";
        output << "  \"exercise\": \"squat\",\n";
        output << "  \"processed_frames\": " << summary.processed_frames << ",\n";
        output << "  \"valid_analysis_frames\": " << summary.valid_analysis_frames << ",\n";
        output << "  \"invalid_analysis_frames\": " << summary.invalid_analysis_frames << ",\n";
        output << "  \"total_reps\": " << summary.reps.size() << ",\n";
        if (summary.reps.empty()) {
            output << "  \"reps\": []\n";
            output << "}\n";
            return output.str();
        }
        output << "  \"reps\": [\n";
        for (std::size_t index = 0; index < summary.reps.size(); ++index) {
            const auto& rep = summary.reps[index];
            output << "    {\n";
            output << "      \"rep\": " << rep.rep_index << ",\n";
            output << "      \"start_time_seconds\": " << rep.start_time_seconds << ",\n";
            output << "      \"end_time_seconds\": " << rep.end_time_seconds << ",\n";
            output << "      \"descent_time_seconds\": " << rep.descent_time_seconds << ",\n";
            output << "      \"ascent_time_seconds\": " << rep.ascent_time_seconds << ",\n";
            output << "      \"minimum_knee_angle_degrees\": "
                   << rep.minimum_knee_angle_degrees << ",\n";
            output << "      \"average_normalized_speed_per_second\": "
                   << rep.average_normalized_speed_per_second << ",\n";
            output << "      \"peak_normalized_speed_per_second\": "
                   << rep.peak_normalized_speed_per_second << "\n";
            output << "    }";
            if (index + 1 < summary.reps.size()) {
                output << ",";
            }
            output << "\n";
        }
        output << "  ]\n";
        output << "}\n";
        return output.str();
    }

    static void write(const std::filesystem::path& path,
                      const SquatSessionSummary& summary) {
        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(path.parent_path());
        }
        std::ofstream output(path);
        if (!output.is_open()) {
            throw std::runtime_error("Failed to open squat summary: " + path.string());
        }
        output << toJson(summary);
        if (!output.good()) {
            throw std::runtime_error("Failed to write squat summary: " + path.string());
        }
    }

private:
    static std::string escape(const std::string& value) {
        std::string escaped;
        escaped.reserve(value.size());
        for (const char character : value) {
            if (character == '\\' || character == '"') {
                escaped.push_back('\\');
            }
            escaped.push_back(character);
        }
        return escaped;
    }
};

}  // namespace video_engine
