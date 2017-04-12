#ifndef SAMPLER_H
#define SAMPLER_H

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

#ifdef WITH_LBP
#include "local_binary_pattern.hpp"
#endif

using namespace cv;
using namespace std;
using namespace Eigen;


class Sampler
{
public:
	Sampler();
	void extract(Mat& frame, Rect ground_truth, MatrixXd& features, VectorXd& labels, int n_samples);
	bool is_initialized();
private:
	bool initialized;
	mt19937 generator;
	vector<Rect> samples;
};

#endif