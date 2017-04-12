#ifndef TEST_CLASSIFIER_H
#define TEST_CLASSIFIER_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <time.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>

#include "image_generator.hpp"
#include "sampler.hpp"
#include "c_utils.hpp"

#ifdef WITH_LR
	#include "logistic_regression.hpp"
#endif

#ifdef WITH_HMC
	#include "hmc.hpp"
#endif

using namespace cv;
using namespace std;
using namespace Eigen;

class TestClassifier{
public:
	TestClassifier(int n_train, int n_predict, string firstFrameFilename, string gtFilename);
	void run();
private:
	int n_train, n_predict, num_frames;
	imageGenerator generator;
	string firstFrameFilename;
	string gtFilename;
	vector<Mat> images;
	vector<string> gt_vec;
};

#endif