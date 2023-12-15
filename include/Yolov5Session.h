#pragma once
#include <string>
#include <openvino/openvino.hpp>

#include "ISession.h"


class Yolov5Session: public ISession
{
public:
    Yolov5Session();
    ~Yolov5Session();


    bool Initialize(const std::string& modelPath) override;

    std::vector<ResultNode> Detect(const cv::Mat& image) override;


private:
    ov::Tensor Preprocess(cv::Mat& image);

    /// @brief resize 图像的大小以符合模型输入的大小
    /// @param image 需要处理的图像
    /// @return 返回是否转换成功
    bool ConvertSize(cv::Mat& image);

    /// @brief 根据这个input_data构造这个ov::tensor
    /// @return 
    ov::Tensor BuildTensor(const cv::Mat& image);


private:
    // 阻塞推理
    ov::Tensor Infer(const ov::Tensor& tensor);


private:

    // 
    std::vector<ResultNode> Postprocess(const ov::Tensor& tenser, const cv::Size& oriImageShape);


    void ParseRawOutput(const ov::Tensor& tensor, float conf_threshold, std::vector<cv::Rect>& boxes, std::vector<float>& confs, std::vector<int>& classIds);

    void GetBestClassInfo(std::vector<float>::iterator it, const int& numClasses, float& bestConf, int& bestClassId);

    void GetOriCoords(const cv::Size& currentShape, const cv::Size& originalShape, cv::Rect& outCoords);
private:
    
    bool ParseModel(std::shared_ptr<ov::Model> model);

    bool IsGPUAvailable(const ov::Core& core);

    bool BuildProcessor(std::shared_ptr<ov::Model> model);

protected:
    bool WarmUpModel() override; // 暂时有问题

private:
    std::shared_ptr<ov::Model> m_model;
    ov::CompiledModel m_compiled_model;

    ov::Core m_core;
    ov::InferRequest m_request;
    //std::shared_ptr<ov::preprocess::PrePostProcessor> m_ppp;

    bool useGpu = true;

    float ratio = 1.0f;     // 缩放比例, 保留用于输出后还原输出的坐标
    float dw = 0.f;         //
    float dh = 0.f;

    // [1, 3, 640, 640]
    std::size_t input_batch     = 0;
    std::size_t input_channels  = 0;
    std::size_t input_width     = 0;
    std::size_t input_height    = 0;

    // [1, 25200, 10]
    std::size_t output_batch    = 0;
    std::size_t output_box_num  = 0;
    std::size_t output_box_size = 0;
};
