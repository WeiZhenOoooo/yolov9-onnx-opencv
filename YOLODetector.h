//
// Created by WeiZhen on 2024/4/7.
//

#ifndef ONNX_SCRIPT_YOLODETECTOR_H
#define ONNX_SCRIPT_YOLODETECTOR_H

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

struct DetectResult
{
    int classId;
    float score;
    cv::Rect box;
};

class YOLODetector {
public:
    YOLODetector();
    void initConfig(const std::string& onnxpath, int iw, int ih, float threshold);
    void detect(cv::Mat& frame, std::vector<DetectResult>& results);

private:
    int input_w = 640;
    int input_h = 640;
    cv::dnn::Net net;
    float threshold_score = 0.25;

    /**
     * 等比resize图像, resize图像最大边到指定尺寸
     * @param mat
     * @param max_edge
     * @return
     */
    cv::Mat resize_max_edge(cv::Mat mat, int max_edge);
};


#endif //ONNX_SCRIPT_YOLODETECTOR_H
