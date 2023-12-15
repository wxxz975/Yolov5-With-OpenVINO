#include "Yolov5Session.h"

#include "slog.h"

Yolov5Session::Yolov5Session()
{

}


Yolov5Session::~Yolov5Session()
{

}

bool Yolov5Session::Initialize(const std::string& modelPath)
{
    m_model = m_core.read_model(modelPath);

    if (!ParseModel(m_model) )
        return false;

    if (!BuildProcessor(m_model))
        return false;
    
    m_compiled_model = m_core.compile_model(m_model);
    m_request = m_compiled_model.create_infer_request();

    return true;
}

std::vector<ResultNode> Yolov5Session::Detect(const cv::Mat& oriImage)
{
    cv::Mat inferImage = oriImage.clone();
    
    ov::Tensor input_tensor = Preprocess(inferImage);

    ov::Tensor output_tensor = Infer(input_tensor);

    return Postprocess(output_tensor, oriImage.size());
}

ov::Tensor Yolov5Session::Preprocess(cv::Mat &image)
{
    if(!ConvertSize(image)) {  // 这里个不能放入ov中进行预处理，因为需要保留原始图像到输入图像转化的比例、位置信息
        slog::info << "failed to Convert Size!\n";
        return ov::Tensor();
    }

    return BuildTensor(image);
}

bool Yolov5Session::ConvertSize(cv::Mat &image)
{
     float height = static_cast<float>(image.rows);
    float width = static_cast<float>(image.cols);

    float r = std::min(input_height / height, input_width / width);
    int padw = static_cast<int>(std::round(width * r));  // 需要放缩成为的值
    int padh = static_cast<int>(std::round(height * r));

    // 输入图像的宽高不一致的情况 
    if((int)width != padw || (int)height != padh) 
        cv::resize(image, image, cv::Size(padw, padh));
    

    // 把等比缩放得到的图像 计算需要填充padding值
    float _dw = (input_width - padw) / 2.f; 
    float _dh = (input_height - padh) / 2.f;
    // 除2是为了把添加的padding 平摊到左右两边, 是为了保证放缩后的图像在整个图像的正中央
    
    int top =  static_cast<int>(std::round(_dh - 0.1f));
    int bottom = static_cast<int>(std::round(_dh + 0.1f));
    int left = static_cast<int>(std::round(_dw - 0.1f));
    int right = static_cast<int>(std::round(_dw + 0.1f));
    cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT,
                        cv::Scalar(114, 114, 114));

    // 还原坐标只需要乘这个ratio即可
    this->ratio = 1 / r;  
    this->dw = _dw;
    this->dh = _dh;

    return true;
}

ov::Tensor Yolov5Session::BuildTensor(const cv::Mat &image)
{
    return ov::Tensor(
            m_compiled_model.input().get_element_type(), 
            m_compiled_model.input().get_shape(), 
            image.data
        );
}

ov::Tensor Yolov5Session::Infer(const ov::Tensor &tensor)
{
    m_request.set_input_tensor(tensor);
    m_request.infer();

    return m_request.get_output_tensor();
}

std::vector<ResultNode> Yolov5Session::Postprocess(const ov::Tensor &tenser, const cv::Size& oriImageShape)
{
    std::vector<ResultNode> result;
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;
    
    cv::Size resizedImageShape = { static_cast<int>(input_width), static_cast<int>(input_height) };
    
    ParseRawOutput(tenser, confidenceThreshold_, boxes, confs, classIds);

    std::vector<int> indices; // store the nms result (index)
    // nms
    cv::dnn::NMSBoxes(boxes, confs, confidenceThreshold_,  iouThreshold_, indices);

    for (int idx : indices)
    {
        ResultNode det;
        
        GetOriCoords(resizedImageShape, oriImageShape, boxes[idx]);

        det.x = boxes[idx].x;
        det.y = boxes[idx].y;
        det.w = boxes[idx].width;
        det.h = boxes[idx].height;

        det.confidence = confs[idx];
        det.classIdx = classIds[idx];
        
        result.emplace_back(det);
    }

    return result;
}

void Yolov5Session::ParseRawOutput(const ov::Tensor& tensor, float conf_threshold, std::vector<cv::Rect>& boxes, std::vector<float>& confs, std::vector<int>& classIds)
{
    float* rawOutput = reinterpret_cast<float*>(tensor.data());
    std::size_t output_size = tensor.get_size();
    ov::Shape output_shape = tensor.get_shape();

    slog::info << "output size:" << output_size << " output shape:" << output_shape << slog::endl;

    std::vector<float> output(rawOutput, rawOutput + output_size);

    std::size_t numClasses = output_box_size - YOLOV5_OUTBOX_ELEMENT_COUNT; // 这个受模型影响
    
    // only for batch size = 1
    for (auto it = output.begin(); it != output.begin() + output_size; it += output_box_size)
    {
        const RawResult* box = reinterpret_cast<const RawResult*>(&(*it)); 
        float clsConf = box->cls_conf;

        if (clsConf > conf_threshold)
        {
            int centerX = box->cx;
            int centerY = box->cy;
            int width = box->w;
            int height = box->h;
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            float objConf;
            int classId;
            GetBestClassInfo(it, numClasses, objConf, classId);

            float confidence = clsConf * objConf;

            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }
}

void Yolov5Session::GetBestClassInfo(std::vector<float>::iterator it, const int& numClasses, float& bestConf, int& bestClassId)
{
  // first 5 element are box and obj confidence
  bestClassId = 5;
  bestConf = 0;
  const int otherCnt = 5; // skip x, y, w, h, box_conf

  for (int i = otherCnt; i < numClasses + otherCnt; i++)
  {
    if (it[i] > bestConf)
    {
      bestConf = it[i];
      bestClassId = i - otherCnt;
    }
  }
}


void Yolov5Session::GetOriCoords(const cv::Size& currentShape, const cv::Size& originalShape, cv::Rect& outCoords)
{
  float gain = std::min(static_cast<float>(currentShape.height) / static_cast<float>(originalShape.height),
                        static_cast<float>(currentShape.width) / static_cast<float>(originalShape.width));

  int pad[2] = {
    static_cast<int>((static_cast<float>(currentShape.width) - static_cast<float>(originalShape.width) * gain) / 2.0f),
    static_cast<int>((static_cast<float>(currentShape.height) - static_cast<float>(originalShape.height) * gain) / 2.0f)
  };

  outCoords.x = static_cast<int>(std::round((static_cast<float>(outCoords.x - pad[0]) / gain)));
  outCoords.y = static_cast<int>(std::round((static_cast<float>(outCoords.y - pad[1]) / gain)));

  outCoords.width = static_cast<int>(std::round(((float)outCoords.width / gain)));
  outCoords.height = static_cast<int>(std::round(((float)outCoords.height / gain)));
}


bool Yolov5Session::ParseModel(std::shared_ptr<ov::Model> model)
{
    try
    {
        const auto& input = model->input();
        const auto& inputShape = input.get_shape();

        // input shape -> [batch, channels, height, width]
        input_batch     = inputShape[0];
        input_channels  = inputShape[1];
        input_height    = inputShape[2];
        input_width     = inputShape[3];
        slog::info <<  inputShape << slog::endl;


        const auto& output = model->output();
        const auto& outputShape = output.get_shape();
        // [1, 25200, 10]   10 -> cx, cy, w, h, conf + cls_conf1 + cls_conf2 + cls_conf3 + ...
        output_batch    = outputShape[0];
        output_box_num  = outputShape[1];
        output_box_size = outputShape[2];
        slog::info <<  outputShape << slog::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Failed ParseModel:" <<e.what() << '\n';
        return false;
    }
    
    return true;
}



bool Yolov5Session::IsGPUAvailable(const ov::Core& core)
{
    std::vector<std::string> avaliableDevice = core.get_available_devices();
    
    auto iter = std::find(avaliableDevice.begin(), avaliableDevice.end(), "GPU");

    return iter != avaliableDevice.end();
}

bool Yolov5Session::BuildProcessor(std::shared_ptr<ov::Model> model)
{
    try
    {
        auto ppp = std::make_shared<ov::preprocess::PrePostProcessor>(m_model);


        ppp->input().tensor()
                    //.set_shape({input_batch, input_channels, input_height, input_width}) //不设置, 应该默认就是这个大小
                    .set_element_type(ov::element::u8)
                    .set_layout("NHWC")
                    .set_color_format(ov::preprocess::ColorFormat::RGB);

        ppp->input().preprocess()
                        .convert_element_type(ov::element::f32)
                        .convert_color(ov::preprocess::ColorFormat::RGB)
                        .convert_layout("NCHW")
                        .scale(255.f);

        ppp->output().postprocess()
                .custom(
                    [&](const ov::Output<ov::Node>& node) {
                        
                        return node;
                });

        model = ppp->build();
    }
    catch(const std::exception& e)
    {
        std::cerr << "Failed BuildProcessor:" << e.what() << '\n';
        return false;
    }

    // slog::info << "build successfully!\n";
    return true;

}

template<class T>
static size_t MultiVec(const std::vector<T>& data)
{
    size_t total = 1;
    for(const auto& it : data)
        total *= it;
    
    return total;
}

bool Yolov5Session::WarmUpModel()
{
    return true;
}