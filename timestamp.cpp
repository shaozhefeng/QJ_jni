#include "timestamp.h"

using std::endl;
using std::cin;
using std::cout;
using std::string;


void mergeFeature(std::vector<float> &feature1, std::vector<float> &feature2) {
    for (std::vector<float>::iterator it = feature2.begin(); it != feature2.end(); it++) {
        feature1.push_back(*it);
    }
}

bool getMat(cv::Mat &mat, int x, int y, int width, int height) {
    int row = mat.rows;
    int col = mat.cols;
    if (y >= 0 && (y + height) < row && x >= 0 && x + width < col) {
        mat = mat.operator()(cv::Range(y, y + height), cv::Range(x, x + width));
        cv::Size size(40, 40);
        cv::resize(mat, mat, size);
        cv::cvtColor(mat, mat, cv::COLOR_RGB2GRAY);
        return true;
    }
    return false;
}

void getNode(std::vector<float> &descriptors, svm_node *x) {
    int i = 0;
    for (std::vector<float>::iterator it = descriptors.begin(); it != descriptors.end(); it++, i++) {
        x[i].index = i + 1;
        x[i].value = (double) *it;
    }
    x[i].index = -1;
}

static struct svm_model *model = NULL;

void loadModel(string modelPath) {
    model = svm_load_model(modelPath.c_str());

    cout << "load model" << endl;
}


std::vector<float> detectTimeStamp(string videoPath, int coorX, int coorY, int width, int height) {

    cv::VideoCapture capture(videoPath);

    // check if we succeeded
    if (!capture.isOpened()) {
        cout << "loading video failed" << endl;
        std::vector<float> tmp;
        tmp.push_back(-1);
        return tmp;
    }

    // init hog params
    cv::Size win(40, 40);
    cv::Size block(16, 16);
    cv::Size stride(8, 8);
    cv::Size cell(8, 8);
    int bin = 9;
    cout << "loading video success" << endl;

//    cout << capture.get(CV_CAP_PROP_FRAME_COUNT) << endl;
    long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
    cout << "The number of the frames is " << totalFrameNumber << endl;
    cv::HOGDescriptor hog(win, block, stride, cell, bin);

    cv::Mat tmpImage;

    std::vector<float> descriptors1;
    std::vector<float> descriptors2;

    svm_node *x_node;
    x_node = new svm_node[1153];

    // the first frame
    bool hasFirst = capture.read(tmpImage);
    // the first frame is invalid
    if (!hasFirst) {
        cout << "the video's frame is invalid" << endl;
        std::vector<float> tmp;
        tmp.push_back(-1);
        return tmp;
    }

    bool isFirstMatValid = getMat(tmpImage, coorX, coorY, width, height);
    if (!isFirstMatValid) {
        cout << "the video's frame can't crop" << endl;
        cout << "or the coordinate you send is out of range" << endl;
        std::vector<float> tmp;
        tmp.push_back(-1);
        return tmp;
    }
    hog.compute(tmpImage, descriptors1);

    int lastGoal = 0;
    std::vector<float> timeArray;


    for (int i = 1; i < totalFrameNumber; i++) {
        bool isRead = capture.read(tmpImage);
        if (!isRead) {
            cout << "the " << i << " frame is invalid,so we skipped it " << endl;
            continue;
        }
        float timeStamp = capture.get(CV_CAP_PROP_POS_MSEC) / 1000;

        bool isMatValid = getMat(tmpImage, coorX, coorY, width, height);
        if (!isMatValid) {
            cout << "the " << i << " frame can't crop,so we skipped it " << endl;
            continue;
        }
        hog.compute(tmpImage, descriptors2);

        mergeFeature(descriptors1, descriptors2);

        getNode(descriptors1, x_node);

//        for(int i=0;;i++){
//            if(x_node[i].index==-1){
//                break;
//            }
//            cout<<x_node[i].index<<"----"<<x_node[i].value<<endl;
//        }

        double result = svm_predict(model, x_node);

        cout << result << "----" << i << endl;

        descriptors1 = descriptors2;
        descriptors2.clear();
        if (result == 1) {
            if (i - lastGoal < 100) {
                lastGoal = i;
            } else {
                lastGoal = i;
                timeArray.push_back(timeStamp);
            }
        }

    }
    delete[] x_node;
    return timeArray;
}





