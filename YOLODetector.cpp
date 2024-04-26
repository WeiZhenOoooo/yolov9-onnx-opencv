//
// Created by WeiZhen on 2024/4/7.
//

#include "YOLODetector.h"
#include "Utils/Utils.h"

YOLODetector::YOLODetector() = default;

void YOLODetector::initConfig(const std::string& onnxpath, int iw, int ih, float threshold, bool isCuda) {
    this->input_w = iw;
    this->input_h = ih;
    this->threshold_score = threshold;
    this->net = cv::dnn::readNetFromONNX(onnxpath);
    if(isCuda){
        spdlog::info("Intialize Model By GPU");
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        spdlog::info("Intialize Model By CPU");
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

void YOLODetector::detect(cv::Mat &img, std::vector<DetectResult> &results) {
    // 图象预处理 - 格式化操作
    cv::Mat frame;
    img.copyTo(frame);
    int imgMax = std::max(img.cols, img.rows);
    float ratio = float(imgMax) / float(this->input_h);
    frame = Utils::resize_max_edge(frame, this->input_h);
//    frame = resize_max_edge(frame, this->input_h);
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));

    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(this->input_w, this->input_h), cv::Scalar(0, 0, 0),
                                          true, false);

    this->net.setInput(blob);

    std::vector<cv::Mat> outputs;
    this->net.forward(outputs, this->net.getUnconnectedOutLayersNames());

    cv::Mat preds = outputs.at(0);
    cv::Mat det_output(preds.size[1], preds.size[2], CV_32F, preds.ptr<float>());
    det_output = det_output.t();
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    for (int i = 0; i < det_output.rows; i++)
    {
        cv::Mat classes_scores = det_output.row(i).colRange(4, det_output.cols);
        cv::Point classIdPoint;
        double score;
        minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

        // 置信度 0～1之间
        if (score > this->threshold_score)
        {
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);
            int x = static_cast<int>((cx - 0.5 * ow) * ratio);
            int y = static_cast<int>((cy - 0.5 * oh) * ratio);
            int width = static_cast<int>(ow * ratio);
            int height = static_cast<int>(oh * ratio);
            cv::Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;

            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(score);
        }
    }

    // NMS
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
    for (int index : indexes)
    {
        DetectResult dr;
        int idx = classIds[index];
        dr.box = boxes[index];
        dr.classId = idx;
        dr.score = confidences[index];
        results.push_back(dr);
    }
//
//    std::ostringstream ss;
//    std::vector<double> layersTimings;
//    double freq = cv::getTickFrequency() / 1000.0;
//    double time = this->net.getPerfProfile(layersTimings) / freq;
//    ss << "FPS: " << 1000 / time << " ; time : " << time << " ms";
//    putText(frame, ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}

