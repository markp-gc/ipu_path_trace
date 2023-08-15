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
  const cv::Mat& updateLdrImage();

  void saveImages(const std::string& fileName);

  /// Accumulate the trace results converting from RGB to BGR in the process:
  void accumulate(const std::vector<TraceRecord>& traces);

  void reset();

  /// Return a copy of the raw HDR image:
  cv::Mat getHdrImage() const { return image; }

private:
  cv::Mat image;
};
