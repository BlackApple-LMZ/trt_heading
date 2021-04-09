//
// Created by limz on 2021/4/01.
//

#include "uff_model_heading.h"

#include <iostream>

int main(int argc, char** argv)
{
	string paramFile;
	if (argc < 2) {
		paramFile = "E:\\project\\trt_heading\\trt_heading\\config\\param.yaml";
	}
	else
		paramFile = argv[1];

    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        return EXIT_SUCCESS;
    }
    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);
    gLogger.reportTestStart(sampleTest);

	uffModelBase * pModel = new uffModel;
    gLogInfo << "Building and running a GPU inference engine for Uff MNIST" << std::endl;

	if (!pModel->initParams(paramFile))
	{
		return 0;
	}
    if (!pModel->build())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!pModel->infer())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!pModel->teardown())
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}
