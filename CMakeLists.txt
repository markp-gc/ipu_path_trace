cmake_minimum_required(VERSION 3.10)

project(ipu_trace)

# Build the CPU MC ray tracer
# sub-project for reference:
add_subdirectory(${PROJECT_SOURCE_DIR}/external/light)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(Eigen3 REQUIRED NO_MODULE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -Werror -Wno-unused-parameter -Wno-ignored-qualifiers)
set(CMAKE_CXX_FLAGS_DEBUG "-g -Og")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

execute_process(COMMAND bash "-c" "popc --version | cut -d ' ' -f3 | head -1" OUTPUT_VARIABLE POPLAR_VERSION)
string(REPLACE "." ";" VERSION_LIST ${POPLAR_VERSION})
list(GET VERSION_LIST 0 POPLAR_VERSION_MAJOR)
list(GET VERSION_LIST 1 POPLAR_VERSION_MINOR)
list(GET VERSION_LIST 2 POPLAR_VERSION_PATCH)
message(STATUS "Detected Poplar version ${POPLAR_VERSION_MAJOR}.${POPLAR_VERSION_MINOR}.${POPLAR_VERSION_PATCH}")

# With Poplar SDK 2.5.0 we need to disable virtual graphs due to an SDK bug:
if(${POPLAR_VERSION} MATCHES "2.5.?")
  message(WARNING "Poplar version 2.5.x detected. Enabling workaround for virtual graph bug.")
  add_compile_definitions(NO_VIRTUAL_GRAPHS)
endif(${POPLAR_VERSION} MATCHES "2.5.?")

file(GLOB IPU_TRACE_SRC ${CMAKE_SOURCE_DIR}/src/*.hpp ${CMAKE_SOURCE_DIR}/src/*.cpp)

message(STATUS "HDF5 LIBRARIES: ${HDF5_LIBRARIES}")

# Following projects are required for the experimental user-interface:

# Configure packetcomms sub-project:
add_subdirectory(external/packetcomms)
add_subdirectory(external/videolib)

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/external
  ${CMAKE_SOURCE_DIR}/src
  ${HDF5_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${NANOGUI_EXTRA_INCS}
  ${PACKETCOMMS_INCLUDES}
  ${VIDEOLIB_INCLUDES}
)
add_definitions(${HDF5_DEFINITIONS})

message(STATUS "VIDEO: ${VIDEOLIB_INCLUDES}")

# Build the neural network library:
add_subdirectory(${CMAKE_SOURCE_DIR}/src/keras)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/neural_networks)

add_executable(ipu_trace ${IPU_TRACE_SRC} codelets.gp)
target_link_libraries(ipu_trace
  ${PACKETCOMMS_LIBRARIES}
  ${VIDEOLIB_LIBRARIES}
  Boost::program_options
  ${OpenCV_LIBS}
  neural_networks
  keras_utils
  ${HDF5_LIBRARIES}
  -lpoplin -lpopnn -lpopops -lpoputil -lpoprand -lpoplar
  OpenMP::OpenMP_CXX -lpthread
  -lpvti)

file(GLOB LIGHT_SRC ${PROJECT_SOURCE_DIR}/light/src/*.hpp ${PROJECT_SOURCE_DIR}light/src/*.cpp)
add_custom_command(
  PRE_BUILD
  MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/src/codelets/codelets.cpp
  DEPENDS ${LIGHT_SRC}
  COMMAND popc -O3 -Werror -Wdouble-promotion -I${CMAKE_SOURCE_DIR}/src/codelets -I${CMAKE_SOURCE_DIR}/external ${CMAKE_SOURCE_DIR}/src/codelets/codelets.cpp -target ipu2,ipu21 -o codelets.gp
  OUTPUT codelets.gp
  WORKING_DIRECTORY ${CMAKE_BUILD_DIR}
)
