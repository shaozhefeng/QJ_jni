
#ifndef _TIMESTAMP_H_
#define _TIMESTAMP_H_

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include "svm.h"
#include <iostream>
#include <string>

void mergeFeature(std::vector<float> &feature1, std::vector<float> &feature2);

bool getMat(cv::Mat &mat, int x, int y, int width, int height);

void getNode(std::vector<float> &descriptors, svm_node *x);

void loadModel(std::string modelPath);

std::vector<float> detectTimeStamp(std::string videoPath, int coorX, int coorY, int width, int height);

#endif
