#include "test_classifier.hpp"

TestClassifier::TestClassifier(int n_train, int n_predict, string firstFrameFilename, string gtFilename){
	this->n_train = n_train;
	this->n_predict = n_predict;
	this->firstFrameFilename = firstFrameFilename;
	this->gtFilename = gtFilename;
	this->generator = imageGenerator(firstFrameFilename, gtFilename);
	this->num_frames = generator.getDatasetSize();
	this->gt_vec = generator.ground_truth;
	this->images = generator.images;
}

void TestClassifier::run(){
	Sampler sampler;
	Mat current_frame;
	Rect ground_truth;
	string string_gt;

	MatrixXd features;
	VectorXd labels;
#ifdef WITH_LR
	LogisticRegression logistic_regression;
#endif

#ifdef WITH_HMC
	Hamiltonian_MC hmc;
#endif
	for (int i = 0; i < this->num_frames; ++i)
	{
		current_frame = this->images[i].clone();
		string_gt = this->gt_vec[i];
		ground_truth = generator.stringToRect(string_gt);
		
		if (!sampler.is_initialized())
		{
			sampler.extract(current_frame, ground_truth, features, labels, this->n_train);
#ifdef WITH_LR
			logistic_regression = LogisticRegression(features, labels);
			logistic_regression.train(1e4,1e-3,1e-2);
#endif

#ifdef WITH_HMC
			hmc = Hamiltonian_MC(features, labels, 0.1);
			hmc.run(1e3,1e-2,10);
#endif
		}
		else
		{
			sampler.extract(current_frame, ground_truth, features, labels, this->n_predict);

#ifdef WITH_LR
			VectorXd phi = logistic_regression.predict(features, false);
#endif

#ifdef WITH_HMC
			VectorXd phi = hmc.predict(features, true, 10);
#endif
			cout << "------------------------------------------------------------------------------------" << endl;
			cout << "Frame: " << i << endl;
			C_utils::report(labels, phi, true);
			//cout << "n_samples:" << this->n_predict << ", features size:(" << 
			//features.rows() << "," << features.rows() << "), labels size:" << labels.size() << endl;

		}
	}
}

int main(int argc, char const *argv[])
{
	if(argc != 9) {
		cerr <<"Incorrect input list" << endl;
		cerr <<"exiting..." << endl;
		return EXIT_FAILURE;
	}
	else{
		string firstFrameFilename, gtFilename;
		int n_train, n_predict;
		if(strcmp(argv[1], "-img") == 0) {
			firstFrameFilename = argv[2];
		}
		else{
			cerr <<"No images given" << endl;
			cerr <<"exiting..." << endl;
			return EXIT_FAILURE;
		}
		if(strcmp(argv[3], "-gt") == 0) {
			gtFilename = argv[4];
		}
		else{
			cerr <<"No ground truth given" << endl;
			cerr <<"exiting..." << endl;
			return EXIT_FAILURE;
		}
		if(strcmp(argv[5], "-ntrain") == 0) {
			n_train = atoi(argv[6]);
		}
		else{
			n_train = 300;
		}
		if(strcmp(argv[7], "-npredict") == 0) {
			n_predict = atoi(argv[8]);
		}
		else{
			n_predict = 300;
		}
		TestClassifier test_classifier(n_train, n_predict, firstFrameFilename, gtFilename);
		test_classifier.run();
	}
	
	return 0;
}