//
// Created by WeiZhen on 2024/4/7.
//

#ifndef ONNX_SCRIPT_YOLODETECTOR_H
#define ONNX_SCRIPT_YOLODETECTOR_H

#include <opencv2/opencv.hpp>

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
    int input_w = 256;
    int input_h = 256;
    cv::dnn::Net net;
    float threshold_score = 0.25;
};


#endif //ONNX_SCRIPT_YOLODETECTOR_H
