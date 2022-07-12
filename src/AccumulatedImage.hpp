// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <opencv2/imgproc.hpp>

struct TraceRecord;

void saveLdrImage(cv::Mat& hdrImage, const std::string& fileName);

void saveHdrImage(cv::Mat& hdrImage, const std::string& fileName);

struct AccumulatedImage {
  AccumulatedImage(std::size_t w, std::size_t h);
  virtual ~AccumulatedImage();

  /// Tone map the HDR image and return a reference to the result.
  const cv::Mat& updateLdrImage(std::size_t step, float exposure, float gamma);

  void saveImages(const std::string& fileName, std::size_t step, float exposure, float gamma);

  /// Accumulate the trace results converting from RGB to BGR in the process:
  void accumulate(const std::vector<TraceRecord>& traces);

  void reset();

private:
  cv::Mat hdrImage;
  cv::Mat image;
};
