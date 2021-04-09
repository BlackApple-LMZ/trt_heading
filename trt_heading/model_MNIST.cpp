/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleUffMNIST.cpp
//! This file contains the implementation of the Uff MNIST sample.
//! It creates the network using the MNIST model converted to uff.
//!
//! It can be run with the following command line:
//! Command: ./sample_uff_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
//#include "logger.h"

#include "NvInfer.h"
#include "NvUffParser.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

#include<opencv2/opencv.hpp>

#include"model_MNIST.h"


void SampleUffMNIST::constructNetwork(
    SampleUniquePtr<nvuffparser::IUffParser>& parser, SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    // There should only be one input and one output tensor
    assert(mParams_.inputTensorNames.size() == 1);
    assert(mParams_.outputTensorNames.size() == 1);

    // Register tensorflow input
	//(INPUT_C, INPUT_H, INPUT_W)
    parser->registerInput(
        mParams_.inputTensorNames[0].c_str(), nvinfer1::Dims3(1, 28, 28), nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput(mParams_.outputTensorNames[0].c_str());

    parser->parse(mParams_.uffFileName.c_str(), *network, nvinfer1::DataType::kFLOAT);

    if (mParams_.int8)
    {
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

	assert(network->getNbInputs() == 1);
	mInputDims_ = network->getInput(0)->getDimensions();
	assert(mInputDims_.nbDims == 3);
}

//!
//! \brief Reads the input data, preprocesses, and stores the result in a managed buffer
//!
bool SampleUffMNIST::processInput(
    const samplesCommon::BufferManager& buffers, const std::vector<std::string>& inputTensorName) const
{
    const int inputH = mInputDims_.d[1];
    const int inputW = mInputDims_.d[2];

    std::vector<uint8_t> fileData(inputH * inputW);
    readPGMFile(locateFile(std::to_string(5) + ".pgm", mParams_.dataDirs), fileData.data(), inputH, inputW);

    // Print ASCII representation of digit
    gLogInfo << "Input:\n";
    for (int i = 0; i < inputH * inputW; i++)
    {
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    gLogInfo << std::endl;

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(inputTensorName[0]));

    for (int i = 0; i < inputH * inputW; i++)
    {
        hostInputBuffer[i] = 1.0 - float(fileData[i]) / 255.0;
    }
    return true;
}

bool SampleUffMNIST::verifyOutput(
    const samplesCommon::BufferManager& buffers, const std::vector<std::string>& outputTensorName)
{
    const float* prob = static_cast<const float*>(buffers.getHostBuffer(outputTensorName[0]));

    gLogInfo << "Output:\n";

    float val{0.0f};
    int idx{0};

    // Determine index with highest output value
    for (int i = 0; i < kDIGITS; i++)
    {
        if (val < prob[i])
        {
            val = prob[i];
            idx = i;
        }
    }

    // Print output values for each index
    for (int j = 0; j < kDIGITS; j++)
    {
        gLogInfo << j << "=> " << setw(10) << prob[j] << "\t : ";

        // Emphasize index with highest output value
        if (j == idx)
        {
            gLogInfo << "***";
        }
        gLogInfo << "\n";
    }

    gLogInfo << std::endl;
    return (idx == 5);
}

