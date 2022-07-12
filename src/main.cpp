// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <cstdlib>

#include "PathTracerApp.hpp"

/// Process the command line options for the path tracing application.
boost::program_options::options_description getStandardOptions() {
  namespace po = boost::program_options;
  po::options_description desc("Options");
  desc.add_options()
  ("help", "Show command help.")
  ("model",
   po::bool_switch()->default_value(false),
   "If set then use IPU model instead of hardware."
  )
  ("ipus",
   po::value<std::size_t>()->default_value(1),
   "Number of IPUs to use."
  )
  ("save-exe",
   po::value<std::string>()->default_value(""),
   "Save the Poplar graph executable after compilation using this name (prefix)."
  )
  ("load-exe",
   po::value<std::string>()->default_value(""),
   "Load a previously saved executable with this name (prefix) and skip graph and program construction. "
  )
  ("compile-only", po::bool_switch()->default_value(false),
   "If set and save-exe is also set then exit after compiling and saving the graph.")
  ("defer-attach", po::bool_switch()->default_value(false),
   "If true hardware devices will not attach until execution is ready to begin. If false they will be attached (reserved) before compilation starts. ")
  ("log-level", po::value<std::string>()->default_value("info"),
  "Set the log level to one of the following: 'trace', 'debug', 'info', 'warn', 'err', 'critical', 'off'.")
  ;
  return desc;
}

boost::program_options::variables_map parseOptions(int argc, char** argv, boost::program_options::options_description& desc) {
  namespace po = boost::program_options;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    throw std::runtime_error("Show help");
  }

  po::notify(vm);

#ifdef NO_VIRTUAL_GRAPHS
  // Defining NO_VIRTUAL_GRAPHS is a work around for a bug
  // in Poplar SDK 2.5 but it limits us to using 1 IPU:
  if (vm.at("ipus").as<std::size_t>() > 1) {
    throw std::logic_error(
        "You have compiled the application with virtual "
        "graphs disabled but selected more than 1 IPU.");
  }
#endif

  // Check options are set correctly:
  auto saveExe = !vm.at("save-exe").as<std::string>().empty();
  auto loadExe = !vm.at("load-exe").as<std::string>().empty();
  if (saveExe && loadExe) {
    throw std::logic_error("You can not set both save-exe and load-exe.");
  }

  // Check tile widths and heights are compatible:
  auto width = vm.at("width").as<std::uint32_t>();
  auto tileWidth = vm.at("tile-width").as<std::uint32_t>();
  auto height = vm.at("height").as<std::uint32_t>();
  auto tileHeight = vm.at("tile-height").as<std::uint32_t>();

  if (width % tileWidth || height % tileHeight) {
    throw std::logic_error("Tile size does not divide evenly into image size.");
  }

  return vm;
}

void setupLogging(const boost::program_options::variables_map& args) {
  std::map<std::string, spdlog::level::level_enum> levelFromStr = {
      {"trace", spdlog::level::trace},
      {"debug", spdlog::level::debug},
      {"info", spdlog::level::info},
      {"warn", spdlog::level::warn},
      {"err", spdlog::level::err},
      {"critical", spdlog::level::critical},
      {"off", spdlog::level::off}};

  const auto levelStr = args["log-level"].as<std::string>();
  try {
    spdlog::set_level(levelFromStr.at(levelStr));
  } catch (const std::exception& e) {
    std::stringstream ss;
    ss << "Invalid log-level: '" << levelStr << "'";
    throw std::runtime_error(ss.str());
  }
  spdlog::set_pattern("[%H:%M:%S.%f] [%L] [%t] %v");
}

/// Boiler plate code to set-up logging and formatting then
/// run the application via a GraphManager:
int main(int argc, char** argv) {
  PathTracerApp app;
  auto desc = getStandardOptions();
  app.addToolOptions(desc);
  auto opts = parseOptions(argc, argv, desc);
  setupLogging(opts);
  app.init(opts);
  return ipu_utils::GraphManager().run(app);
}
