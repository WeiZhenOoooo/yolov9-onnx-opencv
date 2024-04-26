cmake_minimum_required(VERSION 3.25)
project(yolo-onnx-opencv)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "D:/CredLib_DIR/opencv47/build")
set(Yaml_Cpp_DIR "D:/CredLib_DIR/yaml-cpp")


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${Yaml_Cpp_DIR}/include spdlog)

add_executable(yolo-onnx-opencv main.cpp
        YOLODetector.cpp
        YOLODetector.h
        CLI11/CLI11.hpp
        Utils/Utils.cpp
        Utils/Utils.h
)
target_link_libraries(yolo-onnx-opencv ${OpenCV_LIBS} ${Yaml_Cpp_DIR}/lib/yaml-cpp.lib)