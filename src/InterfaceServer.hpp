// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include "ipu_utils.hpp"
#include "AsyncTask.hpp"

#include <PacketComms.h>
#include <PacketSerialisation.h>
#include <VideoLib.h>
#include <network/TcpSocket.h>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <chrono>
#include <iostream>

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

namespace {

const std::vector<std::string> packetTypes {
    "stop",                // Tell server to stop rendering and exit (client -> server)
    "detach",              // Detach the remote-ui but continue: server can destroy the
                           // communication interface and continue (client -> server)
    "progress",            // Send render progress (server -> client)
    "sample_rate",         // Send throughput measurement (server -> client)
    "env_rotation",        // Update environment light rotation (client -> server)
    "exposure",            // Update tone-map exposure (client -> server)
    "gamma",               // Update tone-map gamma (client -> server)
    "fov",                 // Update field-of-view (client -> server)
    "load_nif",            // Insruct server to load a new
                           // NIF environemnt light (client -> server)
    "render_preview",      // used to send compressed video packets
                           // for render preview (server -> client)
    "hdr_header",          // Header for sending full uncompressed HDR
                           // image data (server -> client).
    "hdr_packet",          // Packet containing a portion of the full uncompressed
                           // HDR image (server -> client).
    "interactive_samples", // New value for interactive samples per step
    "ready",               // Used to sync with the other side once all other subscribers are ready (bi-directional)
};

// Struct and serialize function for HDR
// image data header packet.
struct HdrHeader {
  std::int32_t width;  // image width
  std::int32_t height; // image height
  // Data will be broken into this many packets
  // for transmission so that the comms-link is not
  // blocking on a single giant image packet:
  std::uint32_t packets;
};

template <typename T>
void serialize(T& ar, HdrHeader& s) {
  ar(s.width, s.height, s.packets);
}


struct HdrPacket {
  std::uint32_t id;
  std::vector<float> data;
};

template <typename T>
void serialize(T& ar, HdrPacket& p) {
  ar(p.id, p.data);
}

// Struct and serialize function to send
// telemetry in a single packet:
struct SampleRates {
  float pathRate;
  float rayRate;
};

template <typename T>
void serialize(T& ar, SampleRates& s) {
  ar(s.pathRate, s.rayRate);
}

}  // end anonymous namespace

using namespace std::chrono_literals;

class InterfaceServer {
  void communicate() {
    ipu_utils::logger()->info("User interface server listening on port {}", port);
    serverSocket.Bind(port);
    serverSocket.Listen(0);
    connection = serverSocket.Accept();
    if (connection) {
      ipu_utils::logger()->info("User interface client connected.");
      connection->setBlocking(false);
      PacketDemuxer receiver(*connection, packetTypes);
      sender.reset(new PacketMuxer(*connection, packetTypes));

      // Lambda that enqueues video packets via the Muxing system:
      FFMpegStdFunctionIO videoIO(FFMpegCustomIO::WriteBuffer, [&](uint8_t* buffer, int size) {
        if (sender) {
          ipu_utils::logger()->debug("Sending compressed video packet of size: {}", size);
          sender->emplacePacket("render_preview", reinterpret_cast<VectorStream::CharType*>(buffer), size);
          return sender->ok() ? size : -1;
        }
        return -1;
      });
      videoStream.reset(new LibAvWriter(videoIO));

      auto subs1 = receiver.subscribe("env_rotation",
                                      [&](const ComPacket::ConstSharedPacket& packet) {
                                        deserialise(packet, state.envRotationDegrees);
                                        ipu_utils::logger()->trace("Env rotation new value: {}", state.envRotationDegrees);
                                        stateUpdated = true;
                                      });

      auto subs2 = receiver.subscribe("detach",
                                      [&](const ComPacket::ConstSharedPacket& packet) {
                                        deserialise(packet, state.detach);
                                        ipu_utils::logger()->trace("Remote UI detached.");
                                        stateUpdated = true;
                                      });

      auto subs3 = receiver.subscribe("stop",
                                      [&](const ComPacket::ConstSharedPacket& packet) {
                                        deserialise(packet, state.stop);
                                        ipu_utils::logger()->trace("Render stopped by remote UI.");
                                        stateUpdated = true;
                                      });

      // NOTE: Tone mapping is not done on IPU so for exposure and gamma changes we
      // don't mark state as updated to avoid causing an unecessary render re-start.
      auto subs4 = receiver.subscribe("exposure",
                                      [&](const ComPacket::ConstSharedPacket& packet) {
                                        deserialise(packet, state.exposure);
                                        ipu_utils::logger()->trace("Exposure new value: {}", state.exposure);
                                        stateUpdated = true;
                                      });

      auto subs5 = receiver.subscribe("gamma",
                                      [&](const ComPacket::ConstSharedPacket& packet) {
                                        deserialise(packet, state.gamma);
                                        ipu_utils::logger()->trace("Gamma new value: {}", state.gamma);
                                        stateUpdated = true;
                                      });

      auto subs6 = receiver.subscribe("fov",
                                      [&](const ComPacket::ConstSharedPacket& packet) {
                                        deserialise(packet, state.fov);
                                        // To radians:
                                        state.fov = state.fov * (M_PI / 180.f);
                                        ipu_utils::logger()->trace("FOV new value: {}", state.fov);
                                        stateUpdated = true;
                                      });

      auto subs7 = receiver.subscribe("load_nif",
                                      [&](const ComPacket::ConstSharedPacket& packet) {
                                        deserialise(packet, state.newNif);
                                        ipu_utils::logger()->trace("Received new NIF path: {}", state.newNif);
                                        stateUpdated = true;
                                      });

      auto subs8 = receiver.subscribe("interactive_samples",
                                      [&](const ComPacket::ConstSharedPacket& packet) {
                                        deserialise(packet, state.interactiveSamples);
                                        ipu_utils::logger()->trace("Interactive samples new value: {}", state.interactiveSamples);
                                        stateUpdated = true;
                                      });

      ipu_utils::logger()->info("User interface server entering Tx/Rx loop.");
      syncWithClient(*sender, receiver, "ready");
      serverReady = true;
      while (!stopServer && receiver.ok()) {
        std::this_thread::sleep_for(5ms);
      }
    }
    ipu_utils::logger()->info("User interface server Tx/Rx loop exited.");
  }

  /// Wait until server has initialised everything and enters its main loop:
  void waitForServerReady() {
    while (!serverReady) {
      std::this_thread::sleep_for(5ms);
    }
  }

public:
  enum class Status {
    Stop,
    Restart,
    Continue,
    Disconnected
  };

  struct State {
    float envRotationDegrees = 0.f;
    float exposure = 0.f;
    float gamma = 2.2f;
    float fov = 90.f;
    std::uint32_t interactiveSamples = 1;
    std::string newNif;
    bool stop = false;
    bool detach = false;
  };

  /// Return a copy of the state and mark it as consumed:
  State consumeState() {
    State tmp = state;
    stateUpdated = false;  // Clear the update flag.
    state.newNif.clear();  // Clear model load request.
    return tmp;
  }

  const State& getState() const {
    return state;
  }

  /// Has the state changed since it was last consumed?:
  bool stateChanged() const {
    return stateUpdated;
  }

  InterfaceServer(int portNumber)
      : port(portNumber),
        stopServer(false),
        serverReady(false),
        stateUpdated(false) {}

  /// Launches the UI thread and blocks until a connection is
  /// made and all server state is initialised. Note that some
  /// server state can not be initialised until after the client
  /// has connected.
  void start() {
    stopServer = false;
    serverReady = false;
    stateUpdated = false;
    thread.reset(new std::thread(&InterfaceServer::communicate, this));
    waitForServerReady();
  }

  void initialiseVideoStream(std::size_t width, std::size_t height) {
    if (videoStream) {
      videoStream->AddVideoStream(width, height, 30, video::FourCc('F', 'M', 'P', '4'));
    } else {
      ipu_utils::logger()->warn("No object to add video stream to.");
    }
  }

  void stop() {
    stopServer = true;
    if (thread != nullptr) {
      try {
        thread->join();
        thread.reset();
        ipu_utils::logger()->trace("Server thread joined successfuly");
        sender.reset();
      } catch (std::system_error& e) {
        ipu_utils::logger()->error("User interface server thread could not be joined.");
      }
    }
  }

  void updateProgress(int step, int totalSteps) {
    if (sender) {
      serialise(*sender, "progress", step / (float)totalSteps);
    }
  }

  void updateSampleRate(float pathRate, float rayRate) {
    if (sender) {
      serialise(*sender, "sample_rate", SampleRates{pathRate, rayRate});
    }
  }

  void sendPreviewImage(const cv::Mat& ldrImage) {
    VideoFrame frame(ldrImage.data, AV_PIX_FMT_BGR24, ldrImage.cols, ldrImage.rows, ldrImage.step);
    bool ok = videoStream->PutVideoFrame(frame);
    if (!ok) {
      ipu_utils::logger()->warn("Could not send video frame.");
    }
  }

  bool startSendingRawImage(cv::Mat&& rawImage, std::size_t step) {
    // Wait for any previous tasks to complete:
    if (sendHdrTask.isRunning()) {
      ipu_utils::logger()->debug("Large data transfer still in progress, dropping request");
      return false;
    }

    // Even if the thread has finished we must still join it:
    sendHdrTask.waitForCompletion();

    // We send one whole row of image data in each packet:
    hdrImage = rawImage * (1.f / step);
    if (hdrImage.channels() != 3) {
      throw std::logic_error("Only transmission of 3 channel raw data is supported.");
    }
    auto chunkSize = hdrImage.cols * hdrImage.channels();
    const std::uint32_t floats = hdrImage.cols * hdrImage.rows * hdrImage.channels();
    const std::uint32_t chunks = floats / chunkSize;

    // Send the header packet:
    if (sender) {
      ipu_utils::logger()->debug("Initiating large data transfer: {} chunks", chunks);
      serialise(*sender, "hdr_header", HdrHeader{hdrImage.cols, hdrImage.rows, chunks});
    } else {
      ipu_utils::logger()->debug("No muxer available: large data transfer aborted.");
      return false;
    }

    // Launch async task to send chunks in the background:
    sendHdrTask.run([&, floats, chunks, chunkSize] () {
      // Copy data into a vector for now, can optimise later:
      std::vector<float> data(chunkSize);
      ipu_utils::logger()->debug("Data size: {} chunksize: {}", data.size(), chunkSize);

      cv::cvtColor(hdrImage, hdrImage, cv::COLOR_BGR2RGB);

      auto startTime = std::chrono::steady_clock::now();
      for (std::uint32_t c = 0; c < chunks; ++c) {
        float* sendPtr = hdrImage.ptr<float>(c);
        std::copy(sendPtr, sendPtr + data.size(), data.begin());
        serialise(*sender, "hdr_packet", HdrPacket{c, data});
        ipu_utils::logger()->debug("large transfer: sent chunk {} / {}", c + 1, chunks);
        // Throttle the send rate to maintain interactivity:
        std::this_thread::sleep_for(2ms);
      }
      auto endTime = std::chrono::steady_clock::now();
      auto secs = std::chrono::duration<double>(endTime - startTime).count();
      auto mib = floats * sizeof(float) / (1024.f * 1024.f);
      ipu_utils::logger()->info("{} MiB raw image transmitted in {} seconds", mib, secs);
    });
    return true;
  }

  virtual ~InterfaceServer() {
    sendHdrTask.waitForCompletion();
    stop();
  }

private:
  int port;
  TcpSocket serverSocket;
  std::unique_ptr<std::thread> thread;
  std::atomic<bool> stopServer;
  std::atomic<bool> serverReady;
  std::atomic<bool> stateUpdated;
  std::unique_ptr<TcpSocket> connection;
  std::unique_ptr<PacketMuxer> sender;
  std::unique_ptr<LibAvWriter> videoStream;
  State state;
  cv::Mat hdrImage;
  AsyncTask sendHdrTask;
};
