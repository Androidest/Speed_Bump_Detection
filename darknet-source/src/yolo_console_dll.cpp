#include <iostream>
#include <iomanip> 
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <thread>
using namespace std;

#define OPENCV
#include "yolo_v2_class.hpp"	// imported functions from DLL

#include <opencv2/opencv.hpp>			// C++
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio.hpp"
#include <opencv2/highgui/highgui.hpp>
#pragma comment(lib, "opencv_world340.lib")  
#else
#pragma comment(lib, "opencv_core2413.lib")  
#pragma comment(lib, "opencv_imgproc2413.lib")  
#pragma comment(lib, "opencv_highgui2413.lib") 
#endif

//===========Globals=====================
cv::Point2f center;
cv::Mat persp;
cv::Mat map1;
cv::Mat map2;
cv::Size imgSize;
cv::Size midSize;

//===========functions===================
string calcDistance(bbox_t r);
void drawBounds(cv::Mat frame, vector<bbox_t> result);
void detect();
void fisheyeCalibrate(cv::Mat K, cv::Mat D);
void perspectiveCalibrate(cv::Mat image);
void initCalibration();

int main()
{
	initCalibration();
	detect();
	return 0;
}

string calcDistance(bbox_t r)
{
	stringstream ss;
	vector<cv::Point2f> points = { cv::Point2f(r.x + r.w * 0.5, r.y + r.h) };
	cv::perspectiveTransform(points, points, persp);
	cv::Point2f p = points[0] - center - cv::Point2f(0,409.23);
	ss << 0.325 * cv::sqrt(p.x*p.x + p.y*p.y) / 60;
	return ss.str();
}

void drawBounds(cv::Mat frame, vector<bbox_t> result)
{
	cv::Scalar color(255, 160, 260);
	cv::Scalar color1(255, 255, 255);
	cv::Scalar color2(0, 255, 255);
	for (auto &i : result)
	{
		cv::rectangle(frame, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
		putText(frame, "Bump Distance:", cv::Point2f(i.x, i.y - 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, color1);
		putText(frame, calcDistance(i) + "m", cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, color2);
	}
	cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
	cv::imshow("Áº¿¡»ª-1556123", frame);
	cv::waitKey(1);
}

void detect()
{
	Detector detector("yolo-obj.cfg", "backup/yolo-obj_final.weights");
	detector.nms = 0.01;

	try
	{
		cv::Mat frame;
		cv::VideoCapture cap("a.mp4");
		cv::VideoWriter out("result.avi", -1, cap.get(CV_CAP_PROP_FPS), imgSize);

		for (cap >> frame; cap.isOpened(); cap >> frame)
		{
			cv::remap(frame, frame, map1, map2, CV_INTER_LINEAR); //È¥»û±ä
			vector<bbox_t> result = detector.detect(frame, 0.2);
			drawBounds(frame, result);
			out.write(frame);
		}
		out.release();
	}
	catch (exception &e) { cerr << "exception: " << e.what() << "\n"; getchar(); }
	catch (...) { cerr << "unknown exception \n"; getchar(); }
}

void fisheyeCalibrate(cv::Mat K, cv::Mat D)
{
	int numCornersHor = 9;
	int numCornersVer = 6;
	int numSquares = numCornersHor * numCornersVer;
	cv::Size board_sz = cv::Size(numCornersHor, numCornersVer);
	int calibration_flags = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC + cv::fisheye::CALIB_CHECK_COND + cv::fisheye::CALIB_FIX_SKEW;
	cv::TermCriteria subpix_criteria = cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1);
	cv::TermCriteria calib_criteria = cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 1e-6);

	cv::Mat image;
	vector<cv::Point2f> corners;
	vector<vector<cv::Point3f>> object_points;
	vector<vector<cv::Point2f>> image_points;

	vector<cv::Point3f> obj;
	for (int j = 0; j<numSquares; j++)
		obj.push_back(cv::Point3f(j / numCornersHor, j%numCornersHor, 0.0f));

	stringstream ss;
	for (int i = 1; ; ++i)
	{
		ss.str("");
		ss << i;
		image = cv::imread("chessboard/" + ss.str() + ".jpg");
		if (image.empty()) break;

		cv::cvtColor(image, image, CV_BGR2GRAY);
		bool found = cv::findChessboardCorners(image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH + CV_CALIB_CB_FAST_CHECK + CV_CALIB_CB_NORMALIZE_IMAGE);

		if (found)
		{
			cv::cornerSubPix(image, corners, cv::Size(11, 11), cv::Size(-1, -1), subpix_criteria);
			cv::drawChessboardCorners(image, board_sz, corners, found);
			image_points.push_back(corners);
			object_points.push_back(obj);
		}
		cv::resize(image, image, cv::Size(), 0.5, 0.5);
		cv::imshow("chessboard", image);
		cv::waitKey(1);
	}

	K = cv::Mat(3, 3, CV_32FC1);
	K.ptr<float>(0)[0] = 1;
	K.ptr<float>(1)[1] = 1;
	vector<cv::Mat> rvecs;
	vector<cv::Mat> tvecs;
	cv::fisheye::calibrate(object_points, image_points, imgSize, K, D, rvecs, tvecs, calibration_flags, calib_criteria);

	cout << "K:" << endl << K << endl;
	cout << "D:" << endl << D << endl;
}

void perspectiveCalibrate(cv::Mat image)
{
	cv::Mat image2;
	cv::Size board_sz = cv::Size(9, 6);
	vector<cv::Point2f> corners;
	
	bool found = findChessboardCorners(image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH + CV_CALIB_CB_FAST_CHECK + CV_CALIB_CB_NORMALIZE_IMAGE);
	vector<cv::Point2f> srcPoints = { corners[1], corners[6], corners[46], corners[51] };
	cv::drawChessboardCorners(image, board_sz, srcPoints, found);
	cv::resize(image, image2, cv::Size(), 0.6, 0.6);
	cv::imshow("chessboard", image2);
	cv::imwrite("removefisheye.jpg", image2);
	cv::waitKey(3);

	float w = 30;
	float cx = imgSize.width * 0.5;
	float cy = imgSize.height * 0.5;
	vector<cv::Point2f> destPoints = { cv::Point2f(cx - w,cy - w), cv::Point2f(cx + w,cy - w), cv::Point2f(cx - w,cy + w), cv::Point2f(cx + w,cy + w) };
	persp = cv::getPerspectiveTransform(srcPoints, destPoints);
	cv::warpPerspective(image, image, persp, imgSize);
	cv::resize(image, image2, cv::Size(), 0.6, 0.6);
	cv::imshow("chessboard", image2);
	cv::imwrite("perspectiveTransform.jpg", image2);
	midSize = image2.size();
}

void initCalibration()
{
	cv::Mat image = cv::imread("chessboard/perspective.jpg");
	imgSize = image.size();
	center.x = imgSize.width * 0.5;
	center.y = imgSize.height * 0.5;;

	double m[3][3] = { { 836.23175, 0.0, 937.65674 },
						{ 0.0, 835.4881, 557.33398 },
						{ 0.0,  0.0,  1.0 } };
	cv::Mat K = cv::Mat(3, 3, CV_64F, m);

	double m1[4][1] = { { -0.03737096743132243 },
						{ 0.06279398209636423 },
						{ -0.1137153829921847 },
						{ 0.07180008838602008 } };
	cv::Mat D = cv::Mat(4, 1, CV_64F, m1);

	fisheyeCalibrate(K, D);
	cv::Mat new_K;
	cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, imgSize, cv::Matx33d::eye(), new_K, 1);
	cv::fisheye::initUndistortRectifyMap(K, D, cv::Matx33d::eye(), new_K, imgSize, CV_16SC2, map1, map2);
	cv::remap(image, image, map1, map2, CV_INTER_LINEAR);
	perspectiveCalibrate(image);
}