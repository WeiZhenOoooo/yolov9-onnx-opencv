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
    bool isImage = false, isVideo = false;
    int imgSize = 640;
    float threshold = 0.5;
    app.add_option("-w,--weights", onnxPath, "onnx model path");
    app.add_option("-c,--config", yamlPath, "train yolo yaml path, to find classes definition");
    app.add_option("-i,--input", imgPath, "pred img path");
    app.add_option("-z,--imgsz", imgSize, "img size, default:640 * 640");
    app.add_option("-t,--threshold", threshold, "threshold score, default: 0.5");
    app.add_flag("--image, !--no-image", isImage, "Image inference mode");
    app.add_flag("--video, !--no-video", isVideo, "Video inference mode");
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
    if(!isImage && !isVideo){
        spdlog::error("Please select inference mode");
        return 0;
    }
    spdlog::info("onnx model path: {}", onnxPath);
    spdlog::info("yolo yaml path: {}", yamlPath);
    spdlog::info("pred img path: {}", imgPath);
    spdlog::info("pred img size: {}", imgSize);
    spdlog::info("confidence threshold: {}", threshold);
    spdlog::info("Image inference mode: {}", isImage);
    spdlog::info("Video inference mode: {}", isVideo);
    YAML::Node config = YAML::LoadFile(yamlPath);
    std::map<int, std::string> classNames;
    if(!config["names"].IsNull() && config["names"].IsMap()){
        for(size_t i = 0; i < config["names"].size(); ++i){
            classNames[i] = config["names"][i].as<std::string>();
        }
    }
    spdlog::info("classNames size: {}", classNames.size());
    if(!classNames.empty()){
        cv::Mat input = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
        cv::Mat img;
        input.copyTo(img);
        std::shared_ptr<YOLODetector> detector = std::make_shared<YOLODetector>();
        int width = imgSize, height = imgSize;
        detector->initConfig(onnxPath, width, height, threshold);
        std::vector<DetectResult> results;
        if(isImage){
            detector->detect(img, results);
            for (DetectResult& dr : results)
            {
                cv::Rect box = dr.box;
                box.x = int(box.x);
                box.y = int(box.y);
                box.width = int(box.width);
                box.height = int(box.height);
                std::string tips = classNames[dr.classId];
                tips.append(": ");
                tips.append(std::to_string(dr.score));
                cv::putText(input, tips, cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX,
                            .5, cv::Scalar(255, 0, 0));
                cv::rectangle(input, box, cv::Scalar(0, 0, 255), 2, 8);
            }
            cv::imshow("OpenCV DNN", input);
            cv::waitKey(0);
        } else if(isVideo){
            cv::VideoCapture cap(imgPath);
            if(!cap.isOpened()){
                spdlog::error("Error opening video stream or file!! path: {}", imgPath);
                return 0;
            }
            cv::Mat mat;
            while (true){
                cv::Mat frame;
                cap.read(mat); // 读取新的帧
                frame = mat.clone();
                if(frame.empty()){
                    break;
                }
                detector->detect(frame, results);
                for (DetectResult& dr : results)
                {
                    cv::Rect box = dr.box;
                    std::string tips = classNames[dr.classId];
                    tips.append(": ");
                    tips.append(std::to_string(dr.score));
                    cv::putText(frame, tips, cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX,
                                .5, cv::Scalar(255, 0, 0));
                    cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2, 8);
                }
                cv::imshow("OpenCV DNN", frame);
                cv::waitKey(1);
            }
            //todo:
            spdlog::info("Video inference mode todo");
        }
        results.clear();
    } else {
        spdlog::error("yaml parse error! yamlPath: {}", yamlPath);
    }
    return 0;
}
