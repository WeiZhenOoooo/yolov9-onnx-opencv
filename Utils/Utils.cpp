//
// Created by WeiZhen on 2024/4/26.
//

#include "Utils.h"
#include <fstream>


cv::Mat Utils::resize_max_edge(const cv::Mat& mat, int max_edge) {
    cv::Mat res;
    if(!mat.empty()){
        int width = mat.cols;
        int height = mat.rows;
        int new_width, new_height;
        if(width > height){
            new_width = max_edge;
            new_height = max_edge * (float(height) / width);
        } else if(width < height){
            new_height = max_edge;
            new_width = max_edge * (float(width) / height);
        } else {
            new_width = max_edge;
            new_height = max_edge;
        }
        cv::resize(mat, res, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    }
    return res;
}

int CUDAUtils::getCUDACount() {
    return cv::cuda::getCudaEnabledDeviceCount();
}

bool FileUtils::fileIsExist(const std::string path) {
    bool res = false;
    if(!path.empty()){
        std::ifstream f(path.c_str());
        if(f.good()){
            res = true;
        }
    }
    return res;
}
