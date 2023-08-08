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
    : hdrImage(h, w, CV_32FC3) {
  reset();
}

AccumulatedImage::~AccumulatedImage() {}

const cv::Mat& AccumulatedImage::updateLdrImage(std::size_t step, float exposure, float gamma) {
  // IPU does tone mapping now so this is just a format conversion:
  hdrImage.convertTo(image, CV_8UC3, 255.0);
  return image;
}

void AccumulatedImage::saveImages(const std::string& fileName, std::size_t step, float exposure, float gamma) {
  cv::imwrite(fileName, updateLdrImage(step, exposure, gamma));

  // The image accumulated so far must be divided by the number
  // of iterations in order that the final integrand is divided
  // by the total sample count:
  cv::Mat scaledImage = hdrImage * 1.f / step;
  saveHdrImage(scaledImage, fileName);
}

/// Accumulate the trace results converting from RGB to BGR in the process:
void AccumulatedImage::accumulate(const std::vector<TraceRecord>& traces) {
  #pragma omp parallel for schedule(static, 256) num_threads(16)
  for (std::size_t i = 0; i < traces.size(); ++i) {
    auto& t = traces[i];
    auto c = t.u;
    auto r = t.v;
    if (c >= hdrImage.cols || r >= hdrImage.rows) {
      // Skip as this entry is just worklist padding
    } else {
      hdrImage.at<cv::Vec3f>(r, c) = cv::Vec3f((float)t.b, (float)t.g, (float)t.r);
    }
  }
}

void AccumulatedImage::reset() {
  hdrImage = cv::Vec3f(0.f, 0.f, 0.f);
}
