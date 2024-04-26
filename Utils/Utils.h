//
// Created by WeiZhen on 2024/4/26.
//

#ifndef YOLO_ONNX_OPENCV_UTILS_H
#define YOLO_ONNX_OPENCV_UTILS_H
#include <opencv2/opencv.hpp>

/**
 * 图像工具类
 */
class Utils {
public:
    /**
     * 等比resize图像, resize图像最大边到指定尺寸
     * @param mat
     * @param max_edge
     * @return
     */
    static cv::Mat resize_max_edge(const cv::Mat& mat, int max_edge);
};

/**
 * CUDA工具类
 */
class CUDAUtils{
public:

    /**
     * 获取当前设备的CUDA数量
     * @return
     */
    static int getCUDACount();
};

#endif //YOLO_ONNX_OPENCV_UTILS_H
