#include "jni.h"
#include "demo_Demo.h"
#include "timestamp.h"
#include <iostream>


JNIEXPORT jdoubleArray JNICALL Java_demo_Demo_predict
        (JNIEnv *env, jobject thisObj, jstring videoPath, jint coorX, jint coorY, jint width, jint height) {
    const char *CVideo = env->GetStringUTFChars(videoPath, NULL);

    std::vector<float> results = detectTimeStamp(CVideo, coorX, coorY, width, height);
    jint resultsSzie = results.size();
    jdouble outCArray[resultsSzie];
    jint i = 0;
    for (std::vector<float>::iterator it = results.begin(); it != results.end(); it++, i++) {
        outCArray[i] = *it;
        //std::cout<<outCArray[i]<<"===="<<*it<<std::endl;
    }
    jdoubleArray outJNIArray = env->NewDoubleArray(resultsSzie);  // allocate
    env->SetDoubleArrayRegion(outJNIArray, 0, resultsSzie, outCArray);  // copy
    return outJNIArray;
}

JNIEXPORT void JNICALL Java_demo_Demo_load(JNIEnv *env, jobject thisObj, jstring svmPath) {
    const char *CModel = env->GetStringUTFChars(svmPath, NULL);
    loadModel(CModel);
}
