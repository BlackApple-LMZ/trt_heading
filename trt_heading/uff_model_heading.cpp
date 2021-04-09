//
// Created by limz on 2021/4/01.
//

#include "uff_model_heading.h"
#include <opencv2/opencv.hpp>

void uffModel::constructNetwork(SampleUniquePtr<nvuffparser::IUffParser>& parser, SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    // two input(image and dropout param) and one output tensor
    assert(mParams_.inputTensorNames.size() == 2);
    assert(mParams_.outputTensorNames.size() == 1);

    // Register tensorflow input
    parser->registerInput(mParams_.inputTensorNames[0].c_str(), nvinfer1::Dims3(3, 66, 200), nvuffparser::UffInputOrder::kNCHW);
	
	//todo check this registerInput
	nvinfer1::Dims inputDims;
	inputDims.nbDims = 1;
	parser->registerInput(mParams_.inputTensorNames[1].c_str(), inputDims, nvuffparser::UffInputOrder::kNC);
    parser->registerOutput(mParams_.outputTensorNames[0].c_str());

    parser->parse(mParams_.uffFileName.c_str(), *network, nvinfer1::DataType::kFLOAT);

    if (mParams_.int8)
    {
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
	mInputDims_ = network->getInput(0)->getDimensions();
}

bool uffModel::processInput(const samplesCommon::BufferManager& buffers, const std::vector<std::string>& inputTensorName) const
{
	//fill image input
	const int inputC = mInputDims_.d[0];
    const int inputH = mInputDims_.d[1];
    const int inputW = mInputDims_.d[2];

	// Fill data buffer
	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(inputTensorName[0]));
	float pixelMean[3]{ 175.365f, 175.257f, 163.012f }; // In BGR order 

	cv::Mat image = cv::imread("E:\\project\\tensorRT_test\\tensorRT_test\\data\heading\\image.png");
	cv::imwrite("E:\\project\\tensorRT_test\\tensorRT_test\\data\heading\\image_temp.png", image(cv::Rect(0, 240, image.cols, image.rows - 240)));
	resize(image(cv::Rect(0, 240, image.cols, image.rows - 240)), image, cv::Size(inputW, inputH));

	int channels = image.channels();
	std::cout << "channels: " << channels << std::endl;
	int index = 1;
	for (int i = 0; i < image.rows; i++) {
		uchar *pSrcData = image.ptr<uchar>(i);
		for (int j = 0; j < image.cols; j++, index += channels) {
			hostDataBuffer[index - 1] = (float(pSrcData[j]) - pixelMean[0]) / 255.0;
			hostDataBuffer[index] = (float(pSrcData[j]) - pixelMean[1]) / 255.0;
			hostDataBuffer[index + 1] = (float(pSrcData[j]) - pixelMean[2]) / 255.0;
		}
	}

	//fill drop input
	float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(inputTensorName[1]));
	hostInputBuffer[0] = 1.0;

    return true;
}

bool uffModel::verifyOutput(
	const samplesCommon::BufferManager& buffers, const std::vector<std::string>& outputTensorName)
{
	const float* pheading = static_cast<const float*>(buffers.getHostBuffer(mParams_.outputTensorNames[0]));
	heading_ = pheading[0];

	gLogInfo << "Predict heading: " << heading_ * 180 / PI << std::endl;
	return true;
}
