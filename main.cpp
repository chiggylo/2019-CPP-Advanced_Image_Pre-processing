#include <opencv2/core/utility.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/video.hpp>
#include <fstream>
#include <sstream>
#include <opencv2/dnn.hpp>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace saliency;

using namespace dnn;

std::vector<std::string> classes;


// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255));

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
}

Mat kmeans(Mat saliencyImage, int kmean, bool show) {
	// standard kmean
	const unsigned int singleLineSize = saliencyImage.rows * saliencyImage.cols;
	Mat data = saliencyImage.reshape(1, singleLineSize);
	cv::Mat1f colors;
	std::vector<int> labels;
	kmeans(data, 4, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.), 2, cv::KMEANS_PP_CENTERS, colors);
	for (unsigned int i = 0; i < singleLineSize; i++) {
		data.at<float>(i, 0) = colors(labels[i], 0);
		data.at<float>(i, 1) = colors(labels[i], 1);
		data.at<float>(i, 2) = colors(labels[i], 2);
	}

	// shows the kmeans-level-mapping
	Mat outputImage = data.reshape(1, saliencyImage.rows);
	if (show) {
		imshow("kmeans-levels", outputImage);
		waitKey(0);
	}

	// find the value of the two brightest kmeans and get kmeans values
	int firstBrightest = 0;
	int secondBrightest = 0;
	Mat colorsInt;
	vector<int> kmeanValues;
	colors.convertTo(colorsInt, CV_8UC1, 255, 0);
	for (int i = 0; i < colorsInt.rows; i++) {
		if ((int)colorsInt.at<uchar>(0, i) > firstBrightest) {
			secondBrightest = firstBrightest;
			firstBrightest = (int)colorsInt.at<uchar>(0, i);
		}
		else {
			if ((int)colorsInt.at<uchar>(0, i) > secondBrightest) {
				secondBrightest = (int)colorsInt.at<uchar>(0, i);
			}
		}
		kmeanValues.push_back((int)(colorsInt.at<uchar>(0, i)));
	}

	// find the sum of kmeans pixel
	Mat outputImageInt;
	outputImage.convertTo(outputImageInt, CV_8UC1, 255, 0);
	int sumOfKmeansPixel[4]; // 4 kmeans level
	for (int i = 0; i < outputImage.rows; i++) {
		for (int j = 0; j < outputImage.cols; j++) {
			if ((int)outputImageInt.at<uchar>(i, j) == kmeanValues.at(0)) {
				sumOfKmeansPixel[0]++;
			}
			else if ((int)outputImageInt.at<uchar>(i, j) == kmeanValues.at(1)) {
				sumOfKmeansPixel[1]++;
			}
			else if ((int)outputImageInt.at<uchar>(i, j) == kmeanValues.at(2)) {
				sumOfKmeansPixel[2]++;
			}
			else if ((int)outputImageInt.at<uchar>(i, j) == kmeanValues.at(3)) {
				sumOfKmeansPixel[3]++;
			}
		}
	}

	// filter the image based on the brightest kmeans - if not enough pixel, get 2nd brightest
	bool toomanyPixels = false;
	int dim = outputImage.rows * outputImage.cols;
	for (int i = 0; i < 4; i++) {
		if (firstBrightest == kmeanValues.at(i)) {

			//cout << (float)sumOfKmeansPixel[i] / dim << endl;
			if ((float)sumOfKmeansPixel[i] / dim > 0.1) { // give a threshold
				toomanyPixels = true;
			}
		}
	}

	// start filtering
	for (int i = 0; i < outputImage.rows; i++) {
		for (int j = 0; j < outputImage.cols; j++) {
			if (!toomanyPixels) {
				if ((int)outputImageInt.at<uchar>(i, j) == secondBrightest) {
					outputImage.at<float>(i, j) = 255;
					//cout << "adding second brightest" << endl;
				}
			}
			if ((int)outputImageInt.at<uchar>(i, j) == firstBrightest) {
				outputImage.at<float>(i, j) = 255;
				//cout << "adding first brightest" << endl;
			}
			else if ((int)outputImageInt.at<uchar>(i, j) == secondBrightest) {
				if (!toomanyPixels) {
					if ((int)outputImageInt.at<uchar>(i, j) == secondBrightest) {
						outputImage.at<float>(i, j) = 255;
						//cout << "adding second brightest" << endl;
					}
				}
			}
			else {
				outputImage.at<float>(i, j) = 0;
			}
		}
	}
	return outputImage;
}



int main()
{
	// Saving frames to be process ----------------------------
	//// grab video
	//VideoCapture cap;
	//cap.open("PATH"); // select path to set video


	//// initialise background substraction 
	//Ptr<BackgroundSubtractor> blurSub;
	//blurSub = createBackgroundSubtractorMOG2();

	//// initialise frames
	//Mat backgroundBlur, frame, rgbImage;
	//vector<Mat> imageToProcess, rgbEquivalent;

	////cap.set(CAP_PROP_POS_FRAMES, 1300); // set the video starting point (optional)

	//// processing frames
	//bool autoFrame = true;
	//int count = 0;
	//int videoNumber = 0;
	//while (true) {
	//	cap >> frame;

	//	// when frames finishes, exit
	//	if (frame.empty()) {
	//		break;
	//	}

	//	// show the rgb video
	//	//imshow("RGB video", frame); // Optional
	//	rgbImage = frame.clone();

	//	// pre-processing each frame
	//	cvtColor(frame, frame, COLOR_BGR2GRAY); // to greyscale
	//	GaussianBlur(frame, frame, Size(15, 15), 0, 0, BORDER_DEFAULT); // to blur
	//	//imshow("Blurred video", frame); // Optional

	//	//update the background model using blurred image
	//	blurSub->apply(frame, backgroundBlur);

	//	// allow the saving of the images needed to be process
	//	if (count % 15 == 0) { // the interval for saving frame, eg 15 = every 15th frame
	//		string rgbPath = "rgb";
	//		rgbPath.append(to_string(videoNumber));
	//		rgbPath.append(".jpeg");
	//		string bsPath = "bs";
	//		bsPath.append(to_string(videoNumber));
	//		bsPath.append(".jpeg");

	//		// save image 
	//		imwrite(rgbPath, rgbImage);
	//		imwrite(bsPath, backgroundBlur);

	//		cout << "Extracting Video" << videoNumber << endl; // output to screen to notify user
	//		videoNumber += 1;
	//	}
	//	count += 1;
	//}
	////destroyWindow("RGB video");

	//return 0;
	// ------------------------------


	// Grab the image from path
	Mat processingImage;
	Mat rgbEquivalentImage;

	// Get user input of the specific image to process
	int input;
	cout << "enter: ";
	cin >> input;
	string bsPath = "bs";
	bsPath.append(to_string(input));
	bsPath.append(".jpeg");
	string rgbPath = "rgb";
	rgbPath.append(to_string(input));
	rgbPath.append(".jpeg");


	Mat im_gray = imread(bsPath, IMREAD_GRAYSCALE);
	Mat im_rgb = imread(rgbPath);

	processingImage = im_gray.clone();
	rgbEquivalentImage = im_rgb.clone();

	//imshow("before", processingImage); // optional to view image
	GaussianBlur(processingImage, processingImage, Size(15, 15), 0, 0, BORDER_DEFAULT); // to blur
	//imshow("after", processingImage); // optional to view image
	// waitKey(0);

	// Saliency processing
	// Initialise Saliency
	Ptr<Saliency> saliencyAlgorithm = StaticSaliencyFineGrained::create(); //instantiates the specific Saliency
	// Compute Saliency
	Mat backgroundSaliency;
	saliencyAlgorithm->computeSaliency(processingImage, backgroundSaliency);
	// Saliency k-means
	backgroundSaliency = kmeans(backgroundSaliency, 1, false);


	//imshow("Filtering Kmeans", backgroundSaliency); // optional to view image
	//waitKey(0);
	//destroyWindow("Filtering Kmeans");


	cout << "Processing boundary boxes of blobs" << endl;
	// Blobbing 
	// Finding the edge of the saliency.
	backgroundSaliency.convertTo(backgroundSaliency, CV_8UC1, 255, 0); // convert to greyscale 1 channel for edge detection
	int cannyThreshold = 10; // unsure threshold value effect
	Canny(backgroundSaliency, backgroundSaliency, cannyThreshold, cannyThreshold * 3, 3);
	
	// Finding the blobs
	vector<vector<Point> > contours1;
	vector<Vec4i> hierachy1;
	findContours(backgroundSaliency, contours1, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
	
	// Finding bounding boxes
	vector<vector<Point> > contours_poly(contours1.size());
	vector<Rect> boundRect(contours1.size());
	for (size_t i = 0; i < contours1.size(); i++)
	{
		approxPolyDP(Mat(contours1[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(contours_poly[i]);
	}
	// Draw blobs or bounding boxes (method 1 vs method 2)
	Mat drawing = Mat::zeros(backgroundSaliency.size(), CV_8UC1);
	for (size_t i = 0; i < contours1.size(); i++)
	{
		Scalar color = CV_RGB(255, 255, 255);
		
		// filter small blobs, extend the blobs THEN find nearby blobs to merge - not recommended as the small blobs can contribute to a bigger picture
		int distanceMerge = 0;
		if (boundRect[i].width < 25 || boundRect[i].height < 25) {
			continue;
		}
		if (boundRect[i].y + boundRect[i].height < 400) {
			continue;
		}
		else {
			int distanceMerge = 1; // 'dilate' the boxes by increasing reach
			rectangle(drawing, Point(boundRect[i].x - distanceMerge, boundRect[i].y - distanceMerge), Point(boundRect[i].x + boundRect[i].width + distanceMerge, boundRect[i].y + boundRect[i].height + distanceMerge), color);
	
		}

	}

	//imshow("Remaining blobs", drawing); // optional to view image
	//waitKey(0);
	//destroyWindow("Remaining blobs");

	cout << "Linking nearby blobs" << endl;
	// Connect Nearby Blobs
	// Get all non black points ------ sourced Online at: https://stackoverflow.com/questions/34700070/group-closer-white-pixels-together-and-draw-a-rectangle-around-them-in-opencv
	vector<Point> pts;
	findNonZero(drawing, pts);
	// Define the radius tolerance
	int th_distance = 75; // radius tolerance
	// Apply partition 
	// All pixels within the radius tolerance distance will belong to the same class (same label)
	vector<int> labels;
	// With lambda function (require C++11)
	int th2 = th_distance * th_distance;
	int n_labels = partition(pts, labels, [th2](const Point& lhs, const Point& rhs) {
		return ((lhs.x - rhs.x) * (lhs.x - rhs.x) + (lhs.y - rhs.y) * (lhs.y - rhs.y)) < th2;
		});
	// You can save all points in the same class in a vector (one for each class), just like findContours
	vector<vector<Point>> contours(n_labels);
	for (int j = 0; j < pts.size(); ++j)
	{
		contours[labels[j]].push_back(pts[j]);
	}
	// Get bounding boxes
	vector<Rect> boxes;
	for (int j = 0; j < contours.size(); ++j)
	{
		Rect box = boundingRect(contours[j]);
		boxes.push_back(box);
	}
	Mat res = Mat::zeros(backgroundSaliency.size(), CV_8UC1);
	// Drawing each bounding boxes with increase size for object coverage
	for (int j = 0; j < boxes.size(); ++j)
	{
		// draw extended boxes
		Rect enlarged_box = boxes[j] + Size(40, 40);
		enlarged_box -= Point(20, 20);

		rectangle(res, enlarged_box, Scalar(255, 255, 255), FILLED);

	}
	
	//imshow("Linked boxes", res);  // optional to view image
	//waitKey();
	//destroyWindow("Linked boxes");


	cout << "Filtering boxes" << endl;
	// Re-Blob to crop and sort
	Canny(res, res, cannyThreshold, cannyThreshold * 3, 3);
	vector<vector<Point> > contours2;
	vector<Vec4i> hierachy2;
	findContours(res, contours2, hierachy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
	Mat dst = Mat::zeros(drawing.size(), CV_8UC3);/*
	if (!contours2.empty())
	{
		for (int j = 0; j < contours2.size(); j++)
		{
			Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
			drawContours(dst, contours2, j, colour, FILLED, 8, hierachy2);
		}
	}*/



	// sort and crop based on filter
	//Mat Blob;
	//Blob = dst.clone();
	Rect BlobRect;
	vector<Mat> cropped;
	vector<Rect> roi;
	for (int j = 0; j < contours2.size(); j++)
	{
		BlobRect = boundingRect(contours2[j]);
		if (BlobRect.width < 100 + 20 || BlobRect.height < 100 + 20) {
			continue;
		}
		else if (BlobRect.y + BlobRect.height < 400 - 20) {
			continue;
		}
		else {
			Rect myROI(BlobRect.x, BlobRect.y, BlobRect.width, BlobRect.height);
			roi.push_back(myROI);
			cropped.push_back(rgbEquivalentImage(myROI));
			Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
			drawContours(dst, contours2, j, colour, FILLED, 8, hierachy2);
		}
	}
	//imshow("Remaining boxes", dst); // optional to view image
	//waitKey(0);
	//destroyWindow("Remaining boxes");

	// Show the remaining blobs filtered to be classified // optional to view image
	/*for (int j = 0; j < cropped.size(); j++) {
		imshow("Cropped boxes", cropped[j]);
		waitKey(0);
	}
	destroyWindow("Cropped boxes");*/

	// Initialize the parameters
	float confThreshold = 0.5; // Confidence threshold
	float nmsThreshold = 0.4;  // Non-maximum suppression threshold
	int inpWidth = 416;        // Width of network's input image
	int inpHeight = 416;       // Height of network's input image

	//// Load names of classes
	string classesFile = "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);


	//cout << workingdir() << endl;
	//// Give the configuration and weight files for the model
	String modelConfiguration = "yolov3.cfg";
	String modelWeights = "yolov3.weights";

	//// Load the network

	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
	std::vector<String> outNames = net.getUnconnectedOutLayersNames();

	cout << "Number of object detected: " << cropped.size() << endl;

	for (int j = 0; j < cropped.size(); j++) {

		static Mat blob;
		Mat classifyImage;
		classifyImage = cropped[j].clone();
		// Create a 4D blob from a frame.
		blobFromImage(classifyImage, blob, 1.0 / 255.0, Size(inpWidth, inpHeight), Scalar(), false, false);
		//Size cropDim(cropped.at(0).cols, cropped.at(0).rows);

		// Run a model.
		net.setInput(blob);

		vector<Mat> outs;

		net.forward(outs, outNames);

		//post processing
		vector<int> classIds;
		vector<float> confidences;
		vector<Rect> boxes1;

		for (size_t k = 0; k < outs.size(); ++k)
		{
			// Scan through all the bounding boxes output from the network and keep only the
			// ones with high confidence scores. Assign the box's class label as the class
			// with the highest score for the box.
			float* data = (float*)outs[k].data;
			for (int j = 0; j < outs[k].rows; ++j, data += outs[k].cols)
			{
				Mat scores = outs[k].row(j).colRange(5, outs[k].cols);
				Point classIdPoint;
				double confidence;
				// Get the value and location of the maximum score
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > confThreshold)
				{
					int centerX = (int)(data[0] * classifyImage.cols);
					int centerY = (int)(data[1] * classifyImage.rows);
					int width = (int)(data[2] * classifyImage.cols);
					int height = (int)(data[3] * classifyImage.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes1.push_back(Rect(left, top, width, height));
				}
			}
		}

		// Perform non maximum suppression to eliminate redundant overlapping boxes with
		// lower confidences
		vector<int> indices;
		NMSBoxes(boxes1, confidences, confThreshold, nmsThreshold, indices);
		for (size_t j = 0; j < indices.size(); ++j)
		{
			int idx = indices[j];
			Rect box = boxes1[idx];
			drawPred(classIds[idx], confidences[idx], box.x, box.y,
				box.x + box.width, box.y + box.height, classifyImage);
		}

		// Write the frame with the detection boxes
		Mat detectedFrame;
		classifyImage.convertTo(detectedFrame, CV_8U);
		//cout << "Cropped image" << j << " result" << endl;
		//imshow("result", detectedFrame);
		//imwrite(resultPath, detectedFrame);
		//waitKey(0);
		detectedFrame.copyTo(rgbEquivalentImage(roi.at(j)));
	}
	imshow("yolo result", rgbEquivalentImage);
	string resultPath = "result";
	resultPath.append(to_string(input));
	resultPath.append(".jpeg");
	waitKey(0);
	imwrite(resultPath, rgbEquivalentImage);

	destroyWindow("result");
}
