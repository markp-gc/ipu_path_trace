// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#pragma once

#include <thread>
#include "ipu_utils.hpp"

#include <pvti/pvti.hpp>

/// Class for launching asynchronous processing tasks
/// in a separate thread. Tasks that are launched must
/// be stand alone and eventually terminate of their
/// own accord.
class AsyncTask {
public:
  AsyncTask() : running(false) {}
  virtual ~AsyncTask() {}

  /// Run a thread forwarding all arguments to thread constructor.
  /// The thread must terminate of its own accord.
  /// Throws std::logic_error if a task was already in progress.
  template <typename... Args>
  void run(std::function<void()>&& f) {
    if (job != nullptr) {
      auto error = "Attempted to run AsyncTask while a job was in progress.";
      ipu_utils::logger()->error(error);
      throw std::logic_error(error);
    }

    // Wrap the function object that was passed in in another
    // function that ensures running flag is set and unset by the
    // thread we launch:
    asyncFunc = f;
    auto wrapperFunc = [&]() {
      running = true;
      asyncFunc();
      running = false;
    };

    pvti::Tracepoint scoped(&asyncTraceChannel, "thread_launch");
    job.reset(new std::thread(wrapperFunc));
  }

  /// Wait for the job to complete. Throws std::system_error if
  /// the thread could not be joined.
  void waitForCompletion() {
    if (job != nullptr) {
      try {
        pvti::Tracepoint scoped(&asyncTraceChannel, "thread_join");
        job->join();
        job.reset();
      } catch (std::system_error& e) {
        ipu_utils::logger()->error("Thread could not be joined.");
      }
    }
  }

  bool isRunning() const {
    return running;
  }

private:
  std::function<void()> asyncFunc;
  std::unique_ptr<std::thread> job;
  std::atomic<bool> running;
  pvti::TraceChannel asyncTraceChannel = {"async_task"};
};
