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

    /**
     * 加载模型配置
     * @param onnxpath  onnx 模型路径
     * @param iw    模型输入图像 width
     * @param ih    模型输入图像 height
     * @param threshold 输出预测结果 score threshold
     * @param device cuda device
     */
    void initConfig(const std::string& onnxpath, int iw, int ih, float threshold, bool isCuda);

    /**
     * 预测
     * @param frame 预测图像
     * @param results 预测结果
     */
    void detect(cv::Mat& frame, std::vector<DetectResult>& results);
private:
    int input_w = 640;
    int input_h = 640;
    cv::dnn::Net net;
    float threshold_score = 0.25;
};


#endif //ONNX_SCRIPT_YOLODETECTOR_H
