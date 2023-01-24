#define _CRT_SECURE_NO_WARNINGS
#include <string>
#include <opencv2\opencv.hpp>
#include<iostream>
#include <algorithm>
#include <time.h>
#include<vector>
#include<string>
#include<fstream>
using namespace std;
using namespace cv;
using namespace cv::ml;


//SVM AND OPENCV FOR MASK DETECT AND real-time JUDGE
//select imamges with mask and no mask

void Collect_img()
{
	Mat oriImg, cropImg;
	
	vector<Rect>face;
	CascadeClassifier faceCascade;
	faceCascade.load("E:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");
	VideoCapture capture(0);
	int imgCounter = 1;
	while (true)
	{
		
		capture.read(oriImg);
		if (oriImg.empty())
		{
			break;
		}
		faceCascade.detectMultiScale(oriImg,face,1.1,10);
		for (int i = 0; i < face.size(); i++)
		{
			rectangle(oriImg, face[i].tl(), face[i].br(), Scalar(255, 0, 255), 3);
			Mat cropImg= oriImg(Range(face[i].y, face[i].y + face[i].height), Range(face[i].x, face[i].x + face[i].width)   );
			string imgName = "E:\\opencv\\detect_mask\\no\\train" + to_string(imgCounter) + ".jpg";
			imwrite(imgName, cropImg);
			cout << "collect" << imgCounter << endl;
			imgCounter += 1;
		}
		
		imshow("show", oriImg);
		int key = waitKey(10);
		if (key == 27 || imgCounter > 45)
		{
			break;
		}
	}

}


//data preparation 
Mat trainImg, testImg;
vector<int>trainLabel, testLabel;


void get_mask(Mat & trainImg, vector<int>& trainLabel)
{
	for (int ii1 = 1;  ii1<=40; ii1++)
	{
		string path1 = "E:\\opencv\\detect_mask\\mask\\train" + to_string(ii1) + ".jpg";
		Mat srcImg1 = imread(path1 );
		//cvtColor(srcImg1, srcImg1, COLOR_BGR2GRAY);
		// resize the image in case exceed the memory
		resize(srcImg1, srcImg1, Size(50, 50));
		srcImg1 = srcImg1.reshape(1, 1);
		trainImg.push_back(srcImg1);
		trainLabel.push_back(1);
		cout << "get" << ii1 << endl;
	}
}

void get_no(Mat& trainImg, vector<int>& trainLabel)
{

	for (int ii2 = 1; ii2 <= 40; ii2++)
	{
		string path2="E:\\opencv\\detect_mask\\no\\train"+ to_string(ii2) + ".jpg";
		Mat srcImg2 = imread(path2);
		//cvtColor(srcImg2, srcImg2, COLOR_BGR2GRAY);
		resize(srcImg2, srcImg2, Size(50, 50));
		srcImg2 = srcImg2.reshape(1, 1);
		trainImg.push_back(srcImg2);
		trainLabel.push_back(-1);
		cout << "get" << ii2 << endl;
	}
}
//1: train with parameters choose by myself 2: auto train
void trainSVM(int choice)
{
	Ptr<SVM>svm = SVM::create();
	switch (choice)
	{
	case 1:
		
		svm->setType(SVM::C_SVC);
		svm->setKernel(SVM::LINEAR);
		svm->setDegree(0);
		svm->setGamma(1);
		svm->setCoef0(0);
		svm->setC(1);
		svm->setNu(0);
		svm->setP(0);
		svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 0.01));
		svm->train(trainImg, ROW_SAMPLE, trainLabel);
		svm->save("E:\\opencv\\detect_mask\\svm_mask.xml");
		break;
	case 2:

		svm->trainAuto(trainImg, ROW_SAMPLE, trainLabel);

	default:
		break;
	}
	

	cout << "finish training" << endl;
}

void real_time_predict()
{
	string modelpath = "E:\\opencv\\detect_mask\\svm_mask.xml";
	Ptr<SVM> svm;
	svm = Algorithm::load<SVM>(modelpath);
	Mat frame;
	vector<Rect>face;
	CascadeClassifier faceCascade;
	faceCascade.load("E:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");
	VideoCapture capture(0);
	while (true)
	{
		capture.read(frame);
		faceCascade.detectMultiScale(frame, face, 1.1, 10);
		for (int i = 0; i < face.size(); i++)
		{
			rectangle(frame, face[i].tl(), face[i].br(), Scalar(255, 0, 255), 3);
			Mat ROI = frame(Range(face[i].y, face[i].y + face[i].height), Range(face[i].x, face[i].x + face[i].width));
			//cvtColor(ROI,ROI, COLOR_BGR2GRAY);
			resize(ROI,ROI,Size(50,50) );
			ROI=ROI.reshape(1, 1);
			ROI.convertTo(ROI, CV_32FC1);
			//PCA(ROI, Mat(), 0, 20);
			float response = svm->predict(ROI);
			cout << "result:" << response << endl;
			if (response == 1)
			{
				putText(frame, "wearing mask", Point(face[i].x, face[i].y - 20), 2, 1.5, (255, 255,255));
				imshow("img", frame);
			}
			else
			{
				putText(frame, "no mask", Point(face[i].x, face[i].y - 20), 2, 1.5, (255, 255, 255));
				imshow("img", frame);
			}
		}
		//imshow("img", frame);
		int key = waitKey(10);
		if (key == 27)
		{
			break;
		}


	}
}



void main()
{
	//colect images
	//Collect_img();
	//laod image
	get_mask(trainImg, trainLabel);
	get_no(trainImg, trainLabel);
	// prepare and train model
	trainImg.convertTo(trainImg, CV_32FC1);

	trainSVM(2);
	//load trained models and predict*/
	real_time_predict();
	
	

	

}