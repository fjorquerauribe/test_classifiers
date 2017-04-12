#include "sampler.hpp"

#ifndef PARAMS
const float POS_STD = 3.0;
const float SCALE_STD = 3.0;
const float OVERLAP_RATIO = 0.7;
#endif

Sampler::Sampler(){
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    this->generator.seed(seed1);
    this->initialized = false;
}

bool Sampler::is_initialized(){
	return this->initialized;
}

void Sampler::extract(Mat& frame, Rect ground_truth, MatrixXd& features, VectorXd& labels, int n_samples){
	vector<Rect> samples;
	normal_distribution<double> position_random_x(0.0, POS_STD);
	normal_distribution<double> position_random_y(0.0, POS_STD);
	normal_distribution<double> negative_random_pos(0.0, 20.0);
	Size img_size = frame.size();
	for (int i = 0; i < n_samples; ++i)
	{
		Rect sample;
		float _x, _y, _width, _height;
		float _dx = position_random_x(this->generator);
		float _dy = position_random_y(this->generator);
		_x = MIN(MAX(cvRound(ground_truth.x + _dx), 0), img_size.width);
		_y = MIN(MAX(cvRound(ground_truth.y + _dy), 0), img_size.height);
		_width = MIN(MAX(cvRound(ground_truth.width), 10.0), img_size.width);
		_height = MIN(MAX(cvRound(ground_truth.height), 10.0), img_size.height);

		if ( ((_x + _width) < img_size.width)
			&& (_x > 0)
			&& ((_y + _height) < img_size.height)
			&& (_y > 0)
			&& (_width < img_size.width)
			&& (_height < img_size.height)
			&& (_width > 0)
			&& (_height > 0) )
		{
			sample.x = _x;
			sample.y = _y;
			sample.width = _width;
			sample.height = _height;
		}
		else
		{
			sample.x = ground_truth.x;
			sample.y = ground_truth.y;
			sample.width = cvRound(ground_truth.width);
			sample.height = cvRound(ground_truth.height);
		}
		samples.push_back(sample);
	}
	for (int i = 0; i < n_samples; i++){
        Rect box = ground_truth;
        Rect intersection = (box & ground_truth);
        while( double(intersection.area())/double(ground_truth.area()) > OVERLAP_RATIO ){
            float _dx = negative_random_pos(generator);
            float _dy = negative_random_pos(generator);
            box.x = MIN(MAX(cvRound(ground_truth.x + _dx), 0), img_size.width);
            box.y = MIN(MAX(cvRound(ground_truth.y + _dy), 0), img_size.height);
            box.width = MIN(MAX(cvRound(ground_truth.width), 0), img_size.width - box.x);
            box.height = MIN(MAX(cvRound(ground_truth.height), 0), img_size.height - box.y);
            intersection = (box & ground_truth);
        }
        samples.push_back(box); 
    }
    Mat grayImg;
    cvtColor(frame, grayImg, CV_RGB2GRAY);

#ifdef WITH_LBP
	LocalBinaryPattern local_binary_pattern;
	local_binary_pattern.init(grayImg, samples, true, false, true);
	features = local_binary_pattern.sampleFeatureValue;
#endif
	labels = VectorXd(2 * n_samples);
	labels << VectorXd::Ones(n_samples), VectorXd::Zero(n_samples);

	this->initialized = true;
}