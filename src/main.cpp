
#include <iostream>
#include <vector>
#include <openvino/openvino.hpp>

#include "Yolov5Session.h"
#include "Mics.h"

int main(int argc, char**argv)
{
    if(argc != 3) {
        std::cout << "Usage:" << argv[0] << " <xml_path> <img_path>\n";

        return 0;
    }


    Yolov5Session session;

    std::string xml_path = argv[1];//"/home/wxxz/weights/x_ray.xml";
    std::string img_path = argv[2];//"/home/wxxz/datasets/TestXray/P00266.jpg";

    if( session.Initialize(xml_path)) {

        cv::Mat image = cv::imread(img_path);

        auto result = session.Detect(image);

        //{0: 'Gun', 1: 'Knife', 2: 'Pliers', 3: 'Scissors', 4: 'Wrench'}

        std::vector<std::string> labels = {"Gun", "Knife", "Plier", "Scissors", "Wrench"};

        for(const auto & box : result)
            std::cout << "id:" << box.classIdx << " conf:" << box.confidence << "\n";

        cv::Mat rendered = RenderBoundingBoxes(image, result, labels);

        cv::imwrite("rendered.jpg", rendered);
    }
    

    return 0;
}