#include <iostream>
#include <opencv2/opencv.hpp>
#include "YOLODetector.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include "CLI11/CLI11.hpp"
#include <spdlog/spdlog.h>

int main(int argc, char** argv) {
    CLI::App app{"yolo onnx opencv dnn pred description"};
    argv = app.ensure_utf8(argv);
    std::string onnxPath, yamlPath, imgPath, device = "cpu";
    app.add_option("-w,--weights", onnxPath, "onnx model path");
    app.add_option("-c,--config", yamlPath, "train yolo yaml path, to find classes definition");
    app.add_option("-i,--image", imgPath, "pred img path");
    app.add_option("-d,--device", device, "cuda device, i.e. 0 or 0,1,2,3 or cpu, default: cpu");
    CLI11_PARSE(app, argc, argv);
    if(onnxPath.empty()){
        spdlog::error("onnx model path is empty !!!, onnx model path: {}", onnxPath);
        return 0;
    } else {
        std::ifstream f(onnxPath.c_str());
        if(!f.good()) {
            spdlog::error("onnx model path is not exist !!!, onnx model path: {}, Please check it", onnxPath);
            return 0;
        }
    }
    if(yamlPath.empty()){
        spdlog::error("yolo yaml path is empty !!! , yolo yaml path: {}", yamlPath);
        return 0;
    } else {
        std::ifstream f(yamlPath.c_str());
        if(!f.good()) {
            spdlog::error("yolo yaml path is not exist !!!, yolo yaml path: {}, Please check it", yamlPath);
            return 0;
        }
    }
    if(imgPath.empty()){
        spdlog::error("pred img path is empty !!! , pred img path: {}", imgPath);
        return 0;
    } else {
        std::ifstream f(imgPath.c_str());
        if(!f.good()) {
            spdlog::error("pred img path is not exist !!!, pred img path: {}, Please check it", imgPath);
            return 0;
        }
    }
    spdlog::info("onnx model path: {}", onnxPath);
    spdlog::info("yolo yaml path: {}", yamlPath);
    spdlog::info("pred img path: {}", imgPath);
    YAML::Node config = YAML::LoadFile(yamlPath);
    std::map<int, std::string> classNames;
    if(!config["names"].IsNull() && config["names"].IsMap()){
        for(size_t i = 0; i < config["names"].size(); ++i){
            classNames[i] = config["names"][i].as<std::string>();
        }
    }
    if(!classNames.empty()){
        cv::Mat img = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
        std::shared_ptr<YOLODetector> detector = std::make_shared<YOLODetector>();
        int width = 256, height = 256;
        float threshold = 0.7;
        detector->initConfig(onnxPath, width, height, threshold);
        std::vector<DetectResult> results;
        detector->detect(img, results);
        for (DetectResult& dr : results)
        {
            cv::Rect box = dr.box;
            cv::putText(img, classNames[dr.classId], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX,
                        .5, cv::Scalar(0, 0, 0));
        }
        cv::imshow("OpenCV DNN", img);
        cv::waitKey(0);
        results.clear();
    } else {
        std::cout << "yaml parse error! yamlPath: " << yamlPath << std::endl;
    }
    return 0;
}