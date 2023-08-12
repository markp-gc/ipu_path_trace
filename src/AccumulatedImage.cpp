// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "AccumulatedImage.hpp"

#include "codelets/TraceRecord.hpp"

#include <cmath>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

void saveHdrImage(cv::Mat& hdrImage, const std::string& fileName) {
  auto baseName = fileName.substr(0, fileName.find_last_of('.'));
  cv::imwrite(baseName + ".exr", hdrImage);
}

AccumulatedImage::AccumulatedImage(std::size_t w, std::size_t h)
    : hdrImage(h, w, CV_8UC3) {
  reset();
}

AccumulatedImage::~AccumulatedImage() {}

const cv::Mat& AccumulatedImage::updateLdrImage() {
  // IPU does tone mapping and format conversion now so just return the image:
  return hdrImage;
}

void AccumulatedImage::saveImages(const std::string& fileName) {
  cv::imwrite(fileName, updateLdrImage());
}

/// Accumulate the trace results:
void AccumulatedImage::accumulate(const std::vector<TraceRecord>& traces) {
  #pragma omp parallel for schedule(static, 256) num_threads(16)
  for (std::size_t i = 0; i < traces.size(); ++i) {
    auto& t = traces[i];
    auto c = t.u;
    auto r = t.v;
    if (c >= hdrImage.cols || r >= hdrImage.rows) {
      // Skip as this entry is just worklist padding
    } else {
      hdrImage.at<cv::Vec3b>(r, c) = cv::Vec3b(t.r, t.g, t.b);
    }
  }
}

void AccumulatedImage::reset() {
  hdrImage = cv::Vec3b(0, 0, 0);
}
