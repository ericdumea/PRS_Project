
// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <fstream>
#include <random>

#define K_MAX 10

#define CLASS_NUMBER 5


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void listMeanSquares() {
	FILE* f = fopen("points0.txt", "r");
	float x, y, nr, width, height;
	fscanf(f, "%f", &nr);
	float xmin = FLT_MAX, xmax = 0, ymin = FLT_MAX, ymax = 0;
	Point2f points[1000];
	for (int i = 0; i < nr; i++) {
		fscanf(f, "%f%f", &x, &y);
		points[i] = Point2f(x, y);
		if (x < xmin)
			xmin = x;
		if (x > xmax)
			xmax = x;
		if (y < ymin)
			ymin = y;
		if (y > ymax)
			ymax = y;
	}
	fclose(f);
	width = xmax - xmin + 10;
	height = ymax - ymin + 10;
	Mat img(height, width, CV_8UC3, Scalar(0, 0, 0));
	for (int i = 0; i < nr; i++) {
		img.at<Vec3b>(points[i].y - ymin, points[i].x - xmin) = Vec3b(255, 255, 255);
		points[i] = Point2f(points[i].x - xmin, points[i].y - ymin);
	}
	float teta0, teta1, sumxy = 0, sumx = 0, sumy = 0, sum2x = 0, sum2x2y = 0;
	for (int i = 0; i < nr; i++) {
		sumxy += points[i].x*points[i].y;
		sumx += points[i].x;
		sumy += points[i].y;
		sum2x += points[i].x*points[i].x;
		sum2x2y += pow(points[i].y, 2) - pow(points[i].x, 2);
	}
	teta1 = (nr*sumxy - sumx * sumy) / (nr*sum2x - sumx * sumx);
	teta0 = (sumy - teta1 * sumx) / nr;
	if (atan(teta1) > -PI / 4 && atan(teta1) < PI / 4)
		line(img, Point(0, teta0), Point(width, teta0 + teta1 * width), Scalar(255, 0, 0), 3);
	else
		line(img, Point(-teta0 / teta1, 0), Point((height - teta0) / teta1, height), Scalar(255, 0, 0), 3);
	float beta, r;
	beta = (-atan2(2 * sumxy - (2 * sumx*sumy) / nr, sum2x2y + pow(sumx, 2) / nr - pow(sumy, 2) / nr)) / 2;
	r = (cos(beta)*sumx + sin(beta)*sumy) / nr;
	if (abs(beta) > PI / 4 && abs(beta) < 3 * PI / 4)
		line(img, Point(r / cos(beta), 0), Point((r - height * sin(beta)) / cos(beta), height), Scalar(0, 0, 255));
	else
		line(img, Point(0, r / sin(beta)), Point(width, (r - width * cos(beta)) / sin(beta)), Scalar(0, 0, 255));
	imshow("Line Mean Squares", img);
	waitKey();
}

void ransac() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Point points[1000]; int psize = 0;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) == 0) {
					Point p;
					p.x = j;
					p.y = i;
					points[psize++] = p;
				}
			}
		}
		int t = 10, s = 2;
		float p = 0.99f, q = 0.8f;
		float T = q * (psize + 1);
		float K = log(1 - p) / log(1 - pow(q, s));
		int d = 0; int count = 0; int maxcount = 0;
		int afin = 0; int  bfin = 0; int cfin = 0;
		for (int m = 0; m < ceil(K); m++) {
			count = 0;
			Point point1, point2;
			int k1 = rand() % psize;
			point1 = points[k1];
			int k2 = rand() % psize;
			point2 = points[k2];
			int a = point1.y - point2.y;
			int b = point2.x - point1.x;
			int c = (point1.x*point2.y) - (point2.x*point1.y);
			for (int i = 0; i < psize; i++) {
				d = abs(a*points[i].x + b * points[i].y + c) / sqrt(pow(a, 2) + pow(b, 2));
				if (d < t) {
					count++;
				}
			}
			if (count > maxcount) {
				maxcount = count;
				afin = a;
				bfin = b;
				cfin = c;
			}
		}
		line(src, Point(0, -(cfin / bfin)), Point(width, ((-cfin - afin * width) / bfin)), Scalar(0, 0, 0));
		imshow("RANSAC", src);
		waitKey();
	}
}


void hough_tf() {
	char fname[MAX_PATH];

	typedef struct peak {
		int theta, rho, hval;
		bool operator < (const peak& o) const {
			return hval > o.hval;
		}
	} peak;

	while (openFileDlg(fname)) {
		Mat src;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int diag = sqrt(pow(height, 2) + pow(width, 2));
		float rho = 0;

		Mat hough = Mat::zeros(360, diag + 1, CV_32SC1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) == 255) {
					for (int k = 0; k < 360; k++) {
						rho = j * sin(k*PI/180) + i * cos(k*PI/180);
						if (rho >= 0 && rho <= diag + 1) {
							hough.at<int>(k,ceil(rho))++;
						}
					}
				}
			}
		}
		int maxHough = 0;
		for (int i = 0; i < 360; i++) {
			for (int j = 0; j < diag+1; j++) {
				if (hough.at<int>(i, j) > maxHough) {
					maxHough = hough.at<int>(i, j);
				}
			}
		}
		Mat houghImg;
		hough.convertTo(houghImg, CV_8UC1, 255.f / maxHough);
		imshow("Hough",houghImg);

		peak peaks[20000];
		int n = 0;

		for (int i = 0; i < 360; i++) {
			for (int j = 0; j < diag + 1; j++) {
				bool isLocalMax = true;
				for (int k1 = -1; k1 <= 1; k1++) {
					for (int k2 = -1; k2 <= 1; k2++) {
						if (i + k1 > 0 && i + k1 < 360 && j + k2 > 0 && j + k2 < diag + 1) {
							if (hough.at<int>(i + k1, j + k2) >= hough.at<int>(i,j) && !(k1==0 && k2==0)) {
								isLocalMax = false;
							}
						}
					}
				}
				if (isLocalMax && hough.at<int>(i,j) > 4) {
					peaks[n++].hval = hough.at<int>(i, j);
					peaks[n].rho = j;
					peaks[n].theta = i;
				}
			}
		}
		std::cout << "\n" << n;
		
		std::sort(peaks, peaks + n);

		Mat img(src.size(), CV_8UC3);

		for (int i = 0; i < n; i++) {
			int r = peaks[i].rho;
			int beta = peaks[i].theta;

			Point pt1(0, peaks[i].rho / sin(peaks[i].theta*PI/180));
			Point pt2(img.cols, (peaks[i].rho - img.cols*cos(peaks[i].theta*PI / 180)) / sin(peaks[i].theta*PI / 180));
			line(img, pt1, pt2, Scalar(0, 255, 0));

		}

		imshow("original", src);
		imshow("Final image", img);

		waitKey(0);
	}
}




void distanceTransform() {

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat srcImg = imread(fname, IMREAD_GRAYSCALE);
		Mat dt = srcImg.clone();

		int height = srcImg.rows;
		int width = srcImg.cols;
		int di[8] = { -1,-1,-1,0,0,1,1,1 };
		int dj[8] = { -1,0,1,-1,1,-1,0,1 };
		int weights[8] = { 3,2,3,2,2,3,2,3 };
		int min = INT_MAX;
		int val = 0;

		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {
				min = INT_MAX;
				if (srcImg.at<uchar>(i, j) != 0) {
					for (int k = 0; k < 4; k++) {
						val = srcImg.at<uchar>(i + di[k], j + dj[k]) + weights[k];
						if (val < min) {
							min = val;
						}

						if (min >= 0 || min <= 255) {
							if (srcImg.at<uchar>(i, j) > min) {
								srcImg.at<uchar>(i, j) = min;
							}
						}
					}
				}
			}
		}

		for (int i = height - 2; i > 0; i--) {
			for (int j = width - 2; j > 0; j--) {
				min = INT_MAX;
				if (srcImg.at<uchar>(i, j) != 0) {
					for (int k = 4; k < 8; k++) {
						val = srcImg.at<uchar>(i + di[k], j + dj[k]) + weights[k];
						if (val < min) {
							min = val;
						}
						if (min >= 0 || min <= 255) {
							if (srcImg.at<uchar>(i, j) > min) {
								srcImg.at<uchar>(i, j) = min;
							}
						}
					}
				}
			
			}
		}

		imshow("dt", srcImg);
		waitKey(0);

		char fname1[MAX_PATH];

		while (openFileDlg(fname1)){

			Mat img2 = imread(fname1, IMREAD_GRAYSCALE);
			int sum = 0;
			int nr = 0;
			for (int i = 0; i < img2.rows-1; i++) {
				for (int j = 0; j < img2.cols-1; j++) {
					if (img2.at<uchar>(i, j) == 0) {
						sum += srcImg.at<uchar>(i, j);
						nr++;
					}
				}
			}

			std::cout << std::endl << (float) sum / nr << std::endl;
			getchar();

		}
		

	}

}

Point centerOfMass(Mat src) {
	Point com;
	float medi = 0.0f, medj = 0.0f;
	int nr = 0 ;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == 0) {
				medi += i;
				medj += j;
				nr++;
			}
		}
	}

	com.x = ceil(medj / nr);
	com.y = ceil(medi / nr);
	std::cout << com.x << " ; " << com.y << std::endl;

	return com;
}

Mat translateImg(Mat srcImg, Point displacement) {
	Mat translatedImg = Mat(srcImg.rows, srcImg.cols, CV_8UC1);
	for (int i = abs(displacement.y); i < srcImg.rows - abs(displacement.y); i++) {
		for (int j = abs(displacement.x); j < srcImg.cols - abs(displacement.x); j++) {
			if (i - displacement.y >= 0 && i - displacement.y <= srcImg.rows && j - displacement.x >= 0 && j - displacement.x <= srcImg.cols) {
				if (srcImg.at<uchar>(i, j) == 0) {
					translatedImg.at<uchar>(i - displacement.y, j - displacement.x) = 0;
				}
				else {
					translatedImg.at<uchar>(i - displacement.y, j - displacement.x) = 255;
				}
			}
		}
	}
	return translatedImg;
}


void distanceTransform_withTranslation() {

	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat srcImg = imread(fname, IMREAD_GRAYSCALE);
		Mat dt = srcImg.clone();

		int height = srcImg.rows;
		int width = srcImg.cols;
		int di[8] = { -1,-1,-1,0,0,1,1,1 };
		int dj[8] = { -1,0,1,-1,1,-1,0,1 };
		int weights[8] = { 3,2,3,2,2,3,2,3 };
		int min = INT_MAX;
		int val = 0;

		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {
				min = INT_MAX;
				if (srcImg.at<uchar>(i, j) != 0) {
					for (int k = 0; k < 4; k++) {
						val = srcImg.at<uchar>(i + di[k], j + dj[k]) + weights[k];
						if (val < min) {
							min = val;
						}

						if (min >= 0 || min <= 255) {
							if (srcImg.at<uchar>(i, j) > min) {
								srcImg.at<uchar>(i, j) = min;
							}
						}
					}
				}
			}
		}

		for (int i = height - 2; i > 0; i--) {
			for (int j = width - 2; j > 0; j--) {
				min = INT_MAX;
				if (srcImg.at<uchar>(i, j) != 0) {
					for (int k = 4; k < 8; k++) {
						val = srcImg.at<uchar>(i + di[k], j + dj[k]) + weights[k];
						if (val < min) {
							min = val;
						}
						if (min >= 0 || min <= 255) {
							if (srcImg.at<uchar>(i, j) > min) {
								srcImg.at<uchar>(i, j) = min;
							}
						}
					}
				}

			}
		}

		//imshow("dt", srcImg);
		

		//waitKey(0);

		char fname1[MAX_PATH];

		while (openFileDlg(fname1)) {

			Mat img2 = imread(fname1, IMREAD_GRAYSCALE);

			Point comSrc = centerOfMass(srcImg);
			Point comObj = centerOfMass(img2);

			Point displacement = Point(comObj.x - comSrc.x, comObj.y - comSrc.y);

			Mat translatedImg = translateImg(img2, displacement);

			int sum = 0;
			int nr = 0;
			for (int i = 0; i < translatedImg.rows - 1; i++) {
				for (int j = 0; j < translatedImg.cols - 1; j++) {
					if (translatedImg.at<uchar>(i, j) == 0) {
						sum += srcImg.at<uchar>(i, j);
						nr++;
					}
				}
			}

			std::cout << std::endl << (float)sum / nr << std::endl;
			waitKey(0);
		}
	}
}

void statistical_data_analysis_lab5() {
	
	const int p = 400;
	const int N = 19 * 19;
	
	Mat I = Mat(p, N, CV_8UC1);
	char folder[256] = "Images/images_faces";
	char fname[256];
	for (int i = 1; i <= 400; i++) {
		sprintf(fname, "%s/face%05d.bmp", folder, i);
		Mat img = imread(fname, 0);
		int k = 0;
		for (int r = 0; r < img.rows; r++) {
			for (int c = 0; c < img.cols; c++) {
				I.at<uchar>(i - 1, k) = img.at<uchar>(r, c);
				k++;
			}
		}
	}

	//mean value

	float meanValues[N];
	float miu = 0.0f;
	for (int i = 0; i < p; i++) {
		miu = 0;
		for (int j = 0; j < N; j++) {
			miu += I.at<uchar>(i, j);
		}
		miu = (float)miu / p;
		meanValues[i] = miu;
	}

	std::ofstream meanValuesFile;
	meanValuesFile.open("meanValues.csv");
	for (int count = 0; count < N; count++) {
		meanValuesFile << meanValues[count] << ", ";
	}
	meanValuesFile.close();

	


	//covariance
	std::ofstream covFile;
	covFile.open("covariance.csv");

	Mat C = Mat(N, N, CV_32FC1);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float Cij = 0.0f;
			for (int k = 0; k < p; k++) {
				Cij += ((float)I.at<uchar>(k, i) - meanValues[i])*((float)I.at<uchar>(k, j) - meanValues[j]);
			}
			Cij = Cij / (float)N;
			covFile << Cij << ", ";
			C.at<float>(i, j) = Cij;
		}
		covFile << "\n";
	}

	covFile.close();

	Mat cor = Mat(N, N, CV_32FC1);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float roij = 0.0f;
			if (i != j) {
				roij = C.at<float>(i, j) / (meanValues[i] * meanValues[j]);
				cor.at<float>(i, j) = roij;
			}
			else {
				cor.at<float>(i, j) = 1.0f;
			}

		}
	}

	Mat chart = Mat(256, 256, CV_8UC1);

	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			chart.at<uchar>(i, j) = 255;
		}
	}

	for (int k = 0; k < p; k++) {
		int r = I.at<uchar>(k, (5 * 19) + 14);
		int c = I.at<uchar>(k, (5 * 19) + 4);
		chart.at<uchar>(r, c) = 0;
	}

	imshow("correlation image", chart);
	imshow("image", I);
	waitKey(0);

}

int getMinIndex(float* vector, int n) {

	float minimum = vector[0];
	int index = 0;
	for (int i = 0; i < n; i++) {
		if (vector[i] < minimum) {
			index = i;
			minimum = vector[i];
		}
	}
	return index;
}

void kMeans_points(int nrIterations, int K) {

	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat srcImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat srcImgRgb(srcImg.size(), CV_8UC3);
		cvtColor(srcImg, srcImgRgb, CV_GRAY2RGB);
		std::default_random_engine gen;
		std::uniform_int_distribution<int> dist_img(0, 255);
		Vec3b colors[K_MAX];
		for (int i = 0; i < K; i++)
			colors[i] = { (uchar)dist_img(gen), (uchar)dist_img(gen), (uchar)dist_img(gen) };
		const int height = srcImgRgb.rows;
		const int width = srcImgRgb.cols;
		Point m[K_MAX];
		std::uniform_int_distribution<int> dist_imgHW(0, height);
		for (int i = 0; i < K; i++) {
			m[i].x = dist_imgHW(gen);
			m[i].y = dist_imgHW(gen);
		}
		float distances[K_MAX];
		Mat L = Mat(height, width, CV_8UC1);
		Mat newL = Mat(height, width, CV_8UC1);
		for (int count = 0; count < nrIterations; count++) {
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					for (int k = 0; k < K; k++) {
						distances[k] = sqrt(((m[k].x - i) * (m[k].x - i)) + ((m[k].y - j) * (m[k].y - j)));
					}
					L.at<uchar>(i, j) = getMinIndex(distances, K);
				}
			}
			if (count > 0) {
				cv::Mat diff = L != newL;
				bool eq = cv::countNonZero(diff) == 0;
				if (eq == true) {
					std::cout << "assignent function produced same results at iteration: " << count;
					break;
				}
			}
			int sumsX[K_MAX];
			int sumsY[K_MAX];
			int noOfPoints[K_MAX];
			for (int i = 0; i < K; i++) {
				sumsX[i] = 0;
				sumsY[i] = 0;
				noOfPoints[i] = 0;
			}
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					sumsX[L.at<uchar>(i, j)] += i;
					sumsY[L.at<uchar>(i, j)] += j;
					noOfPoints[L.at<uchar>(i, j)]++;
				}
			}
			for (int c = 0; c < K; c++) {
				m[c].x = sumsX[c] / noOfPoints[c];
				m[c].y = sumsY[c] / noOfPoints[c];
			}

			L.copyTo(newL);
		}
		Mat voroni = Mat::zeros(srcImg.size(), CV_8UC3);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (srcImg.at<uchar>(i, j) != 255) {
					srcImgRgb.at<Vec3b>(i, j) = colors[L.at<uchar>(i, j)];
				}
				voroni.at<Vec3b>(i, j) = colors[L.at<uchar>(i, j)];
			}
		}
		imshow("result", srcImgRgb);
		imshow("Voroni image", voroni);
		waitKey(0);
	}
}

void KMeansClustering_lab06_grayScale(int noOfIterations, int K) {
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat srcImg = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		const int height = srcImg.rows;
		const int width = srcImg.cols;
		Mat rez = Mat(height, width, CV_8UC1);
		uchar m[K_MAX];
		std::default_random_engine gen;
		std::uniform_int_distribution<int> dist_img(0, 255);
		for (int i = 0; i < K; i++) {
			m[i] = dist_img(gen);
		}
		Mat L = Mat(height, width, CV_8UC1);
		Mat newL = Mat(height, width, CV_8UC1);
		float distances[K_MAX];
		for (int count = 0; count < noOfIterations; count++) {
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					for (int k = 0; k < K; k++) {
						distances[k] = sqrt((m[k] - srcImg.at<uchar>(i, j)) *  (m[k] - srcImg.at<uchar>(i, j)));
					}
					L.at<uchar>(i, j) = getMinIndex(distances, K);
				}
			}
			if (count > 0) {
				cv::Mat diff = L != newL;
				bool eq = cv::countNonZero(diff) == 0;
				if (eq == true) {
					std::cout << "assignent function produced same results at iteration: " << count;
					break;
				}
			}
			int sumsGray[K_MAX];
			int noOfPoints[K_MAX];
			for (int i = 0; i < K; i++) {
				sumsGray[i] = 0;
				noOfPoints[i] = 0;
			}
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					sumsGray[L.at<uchar>(i, j)] += srcImg.at<uchar>(i, j);
					noOfPoints[L.at<uchar>(i, j)]++;
				}
			}
			for (int c = 0; c < K; c++) {
				m[c] = sumsGray[c] / noOfPoints[c];
			}
			L.copyTo(newL);
		}
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				rez.at<uchar>(i, j) = m[L.at<uchar>(i, j)];
			}
		}
		imshow("gray", rez);
		waitKey(0);
	}
}

void KMeansClustering_lab06_RGB(int noOfIterations, int K) {
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat srcImg = imread(fname, CV_LOAD_IMAGE_COLOR);
		const int height = srcImg.rows;
		const int width = srcImg.cols;
		Mat rez = Mat(height, width, CV_8UC3);
		//generate centers
		std::default_random_engine gen;
		std::uniform_int_distribution<int> dist_img(0, 255);
		Vec3b m[K_MAX];
		for (int i = 0; i < K; i++) {
			m[i][0] = dist_img(gen);
			m[i][1] = dist_img(gen);
			m[i][2] = dist_img(gen);
		}

		Mat L = Mat(height, width, CV_8UC1);
		Mat newL = Mat(height, width, CV_8UC1);
		float distances[K_MAX];
		for (int count = 0; count < noOfIterations; count++) {
			//Assignment function
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					for (int k = 0; k < K; k++) {
						distances[k] = sqrt(
							((m[k][0] - srcImg.at<Vec3b>(i, j)[0]) * (m[k][0] - srcImg.at<Vec3b>(i, j)[0])) +
							((m[k][1] - srcImg.at<Vec3b>(i, j)[1]) * (m[k][1] - srcImg.at<Vec3b>(i, j)[1])) +
							((m[k][2] - srcImg.at<Vec3b>(i, j)[2]) * (m[k][2] - srcImg.at<Vec3b>(i, j)[2]))
						);
					}
					L.at<uchar>(i, j) = getMinIndex(distances, K);
				}
			}
			if (count > 0) {
				//check if assignment function produced new results
				bool isEqual = (sum(L != newL) == Scalar(0, 0, 0));
				if (isEqual == true) {
					std::cout << "assignent function produced same results at iteration: " << count;
					break;
				}

			}
			//update centers
			int sumsR[K_MAX];
			int sumsG[K_MAX];
			int sumsB[K_MAX];
			int noOfPoints[K_MAX];
			for (int i = 0; i < K; i++) {
				sumsR[i] = 0;
				sumsG[i] = 0;
				sumsB[i] = 0;
				noOfPoints[i] = 0;
			}

			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					sumsB[L.at<uchar>(i, j)] += srcImg.at<Vec3b>(i, j)[0];
					sumsG[L.at<uchar>(i, j)] += srcImg.at<Vec3b>(i, j)[1];
					sumsR[L.at<uchar>(i, j)] += srcImg.at<Vec3b>(i, j)[2];
					noOfPoints[L.at<uchar>(i, j)]++;
				}
			}
			for (int c = 0; c < K; c++) {
				if (noOfPoints[c] > 0) {
					m[c][0] = sumsB[c] / noOfPoints[c];
					m[c][1] = sumsG[c] / noOfPoints[c];
					m[c][2] = sumsR[c] / noOfPoints[c];
				}
			}
			L.copyTo(newL);
		}
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				rez.at<Vec3b>(i, j) = m[L.at<uchar>(i, j)];
			}
		}
		imshow("color", rez);
		waitKey(0);
	}
}

void pca() {
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {

		std::ifstream file;
		file.open(fname);

		int n = 0, d = 0;

		file >> n;
		file >> d;

		std::cout << n << " " << d << "\n";
		Mat X(n, d, CV_64FC1);

		for (int i = 0; i < n; i++)
			for (int j = 0; j < d; j++) {
				file >> X.at<double>(i, j);
			}
		Mat meanValue(d, 1, CV_64FC1);
		for (int i = 0; i < d; i++) {
			double mv = 0;
			for (int j = 0; j < n; j++) {
				mv += X.at<double>(j, i);
			}
			meanValue.at<double>(i, 0) = (double)mv / n;
		}
		for (int i = 0; i < n; i++)
			for (int j = 0; j < d; j++)
				X.at<double>(i, j) -= meanValue.at<double>(j, 0);
		Mat C = Mat(n, d, CV_64FC1);
		C = X.t()*X / (n - 1);
		Mat Lambda, Q;
		eigen(C, Lambda, Q);
		Q = Q.t();
		for (int i = 0; i < Lambda.rows; i++) {
			for (int j = 0; j < Lambda.cols; j++) {
				std::cout << Lambda.at<double>(i, j) << "\t";
			}
			std::cout << std::endl;
		}


		Mat Xcoeff = X * Q; int mini = Xcoeff.at<double>(0, 0), maxi = Xcoeff.at<double>(0, 0);
		for (int i = 0; i < n; i++)
			for (int j = 0; j < 2; j++) {
				if (Xcoeff.at<double>(i, j) < mini)
					mini = Xcoeff.at<double>(i, j);
				if (Xcoeff.at<double>(i, j) > maxi)
					maxi = Xcoeff.at<double>(i, j);
			}
		Mat image = Mat::zeros(maxi - mini, maxi - mini, CV_8UC1);
		for (int j = 0; j < n; j++) {
			image.at<uchar>(Xcoeff.at<double>(j, 0) - mini, Xcoeff.at<double>(j, 1) - mini) = 255;
		}
		
		for (int i = 0; i < image.cols; i++) {
			for (int j = 0; j < image.rows; j++) {
				image.at<uchar>(i, j) = 255 - image.at<uchar>(i, j);
			}
		}

		imshow("", image);
		waitKey(0);


		getchar();
		waitKey(0);
	}
}

int* colorHistogram(Mat srcImg, int nrBins) {
	//convert to hsv before histogram computation
	Mat img_hsv;
	cvtColor(srcImg, img_hsv, CV_BGR2HSV);
	int* histogram = new int[nrBins * 3];
	const int binSize = 256 / nrBins;
	for (int c = 0; c < nrBins * 3; c++) {
		histogram[c] = 0;
	}
		int currentBin = 0;
		for (int i = 0; i < img_hsv.rows; i++) {
			for (int j = 0; j < img_hsv.cols; j++) {
				currentBin = 0;
				while (currentBin < nrBins) {
					uchar red = img_hsv.at<Vec3b>(i, j)[2];
					uchar green = img_hsv.at<Vec3b>(i, j)[1];
					uchar blue = img_hsv.at<Vec3b>(i, j)[0];
					if (blue >= currentBin * binSize && blue < (currentBin + 1)*binSize) {
						histogram[currentBin]++;
					}
					if (green >= currentBin * binSize && green < (currentBin + 1)*binSize) {
						histogram[nrBins + currentBin]++;
					}
					if (red >= currentBin * binSize && red < (currentBin + 1)*binSize) {
						histogram[(nrBins * 2) + currentBin]++;
					}
					currentBin++;
				}

			}
		}
	return histogram;
}

//C = Mat of Floats
double getAccuracyFromConfusionMatrix(Mat C)
{
	double accuracy = 0.0;
	double nominator = 0.0;
	double denominator = 0.0;
	for (int i = 0; i < C.rows; i++) {
		for (int j = 0; j < C.cols; j++) {
			if (i == j) {
				nominator += C.at<float>(i, j);
			}
			denominator += C.at<float>(i, j);
		}

	}
	accuracy = nominator / denominator;
	return accuracy;

}

//k =  number of neighbors
void KNN_classifier_lab08(int k) {

	const int noOfBeans = 11;
	const int histSize = noOfBeans * 3;
	const int noOfInstances = 672;
	const int noOFTestInstances = 85;

	Mat y(noOfInstances, 1, CV_8UC1);
	Mat yTest(noOFTestInstances, 1, CV_8UC1);

	Mat X(noOfInstances, histSize, CV_32FC1);
	Mat XTest(noOFTestInstances, histSize, CV_32FC1);

	const int noOfClasses = 6;
	int fileNr = 0, rowX = 0;

	char classes[noOfClasses][10] =
	{ "beach", "city", "desert", "forest", "landscape", "snow" };

	int countTrain = 0;
	int countTest = 0;

	//read images from training set and add histogram vector to feature matrix
	char fname[256];
	char fnameTest[256];

	//load train instances
	for (int i = 0; i < noOfClasses; i++) {
		fileNr = 0;
		while (1) {
			sprintf(fname, "images_KNN/train/%s/%06d.jpeg", classes[i], fileNr++);
			Mat img = imread(fname);
			if (img.cols == 0) break;
			int* hist = colorHistogram(img, noOfBeans);
			//calculate the histogram in hist
			for (int d = 0; d < histSize; d++)
				X.at<float>(rowX, d) = hist[d];
			y.at<uchar>(rowX) = i;
			rowX++;
			countTrain++;
		}
	}

	rowX = 0;
	//load test instances
	for (int k = 0; k < noOfClasses; k++) {
		fileNr = 0;
		while (1) {
			sprintf(fnameTest, "images_KNN/test/%s/%06d.jpeg", classes[k], fileNr++);
			Mat img = imread(fnameTest);
			if (img.cols == 0) break;
			int* hist = colorHistogram(img, noOfBeans);
			//calculate the histogram in hist
			for (int d = 0; d < histSize; d++)
				XTest.at<float>(rowX, d) = hist[d];
			yTest.at<uchar>(rowX) = k;
			rowX++;
			countTest++;
		}
	}

	//confusion Matrix
	Mat C = Mat::zeros(noOfClasses, noOfClasses, CV_32FC1);
	// X - training set, y -label vector
	int testHist[histSize];
	for (int count = 0; count < noOFTestInstances; count++) {
		//obtain histogram for current image
		for (int d = 0; d < histSize; d++)
			testHist[d] = XTest.at<float>(count, d);
		//compute distances
		double distances[noOfInstances];
		for (int i = 0; i < noOfInstances; i++) {
			distances[i] = 0.0;
			for (int j = 0; j < histSize; j++) {
				distances[i] += (testHist[j] - X.at<float>(i, j))* (testHist[j] - X.at<float>(i, j));
			}
			distances[i] = sqrt(distances[i]);
		}
		//sort the distances
		double sortedDistances[noOfInstances];
		std::copy(distances, distances + noOfInstances, sortedDistances);
		std::sort(sortedDistances, sortedDistances + noOfInstances);
		//find the K nearest distances
		int voteHist[noOfClasses];
		for (int i = 0; i < noOfClasses; i++) {
			voteHist[i] = 0;
		}
		for (int i = 0; i < k; i++) {
			double currentDistance = sortedDistances[i];
			for (int j = 0; j < noOfInstances; j++) {
				if (currentDistance == distances[j]) {
					voteHist[(int)y.at<uchar>(j)] += 1;
				}
			}
		}
		int foundClass = -1;
		int max = -1;
		for (int idx = 0; idx < noOfClasses; idx++) {
			if (max < voteHist[idx]) {
				foundClass = idx;
				max = voteHist[idx];
			}
		}

		//update confusion matrix
		C.at<float>(foundClass, (int)yTest.at<uchar>(count)) += 1.0;
		std::cout << classes[foundClass] << " " << classes[yTest.at<uchar>(count)] << std::endl;

	}
	double accuracy = getAccuracyFromConfusionMatrix(C);
	std::cout << "Accuracy: " << accuracy << std::endl;

	getchar();
	waitKey();
	getchar();
}

//binarization of a grayscale image
Mat grayToBinary(Mat srcImg, int threshold) {
	Mat result = Mat::zeros(srcImg.rows, srcImg.cols, CV_8UC1);
	for (int i = 0; i < srcImg.rows; i++) {
		for (int j = 0; j < srcImg.cols; j++) {
			if (srcImg.at<uchar>(i, j) < threshold) {
				result.at<uchar>(i, j) = 0;
			}
			else {
				result.at<uchar>(i, j) = 255;
			}
		}
	}
	return result;
}


void bayesian_lab09() {
	
	const int featureSize = 28 * 28;
	const int noOfInstances = 3585;
	const int noOfClasses = 5;
	const int noOfTestInstances = 895;
	Mat X = Mat(noOfInstances, featureSize, CV_8UC1);
	Mat XTest = Mat(noOfTestInstances, featureSize, CV_8UC1);
	int rowX = 0;
	int rowXTest = 0;
	Mat y(noOfInstances, 1, CV_8UC1);
	Mat yTest(noOfTestInstances, 1, CV_8UC1);

	char classes[noOfClasses][10] =
	{ "0", "1", "2", "3", "4" };
	//load the train instances
	double priors[5];
	int elementsOfClassTrain[5];
	int priorNr;

	char fname[256];
	for (int i = 0; i < noOfClasses; i++) {
		priorNr = 0;
		while (1) {
			sprintf(fname, "images_Bayes/train/%s/%06d.png", classes[i], priorNr++);
			Mat img = imread(fname, CV_8UC1);
			if (img.cols == 0) break;
			Mat binary = grayToBinary(img, 128);

			int d = 0;
			for (int r = 0; r < binary.rows; r++) {
				for (int c = 0; c < binary.cols; c++) {
					X.at<uchar>(rowX, d) = binary.at<uchar>(r, c);
					d++;
				}
			}
			y.at<uchar>(rowX) = i;
			rowX++;
		}
		priors[i] = priorNr / (double)noOfInstances;
		elementsOfClassTrain[i] = priorNr;
	}

	//loatd the test instances
	char fnameTest[256];
	for (int i = 0; i < noOfClasses; i++) {
		switch (i) {
		case 0: priorNr = 784; break;
		case 1: priorNr = 908; break;
		case 2: priorNr = 827; break;
		case 3: priorNr = 808; break;
		case 4: priorNr = 258; break;
		}
		while (1) {
			sprintf(fnameTest, "images_Bayes/test/%s/%06d.png", classes[i], priorNr++);
			Mat img = imread(fnameTest, CV_8UC1);
			if (img.cols == 0) break;
			Mat binary = grayToBinary(img, 150);

			int d = 0;
			for (int r = 0; r < binary.rows; r++) {
				for (int c = 0; c < binary.cols; c++) {
					XTest.at<uchar>(rowXTest, d) = binary.at<uchar>(r, c);
					d++;
				}
			}
			yTest.at<uchar>(rowXTest) = i;
			rowXTest++;
		}
	}

	//compute likelihood
	//w/ laplace smoothing
	Mat likelihood = Mat::zeros(noOfClasses, featureSize, CV_64FC1);
	for (int k = 0; k < noOfInstances; k++) {
		for (int d = 0; d < featureSize; d++) {
			if (X.at<uchar>(k, d) == 255) {
				likelihood.at<double>((int)y.at<uchar>(k), d) += 1.0;
			}
		}
	}
	//laplace smoothing
	for (int r = 0; r < likelihood.rows; r++) {
		for (int c = 0; c < likelihood.cols; c++) {
			double value = likelihood.at<double>(r, c) + 1.0;
			likelihood.at<double>(r, c) = (value / (double)(noOfClasses + elementsOfClassTrain[r]));
		}
	}

	//classify test images
	Mat C = Mat::zeros(noOfClasses, noOfClasses, CV_32F);
	for (int count = 0; count < noOfTestInstances; count++) {
		Mat randImg = XTest.row(count);
		double classProbs[5];
		for (int c = 0; c < 5; c++) {
			classProbs[c] = log(priors[c]);
			for (int j = 0; j < featureSize; j++) {
				if (randImg.at<uchar>(0, j) == 255) {
					classProbs[c] += log(likelihood.at<double>(c, j));
				}
				else {
					classProbs[c] += log(1.0f - likelihood.at<double>(c, j));
				}
			}
		}
		double max = *std::max_element(classProbs, classProbs + 5);
		int predictedClass = -1;
		for (int i = 0; i < 5; i++) {
			if (max == classProbs[i]) {
				predictedClass = i;
			}
		}
		C.at<float>(predictedClass, yTest.at<uchar>(count)) += 1.0;
	}

	double accuracy = getAccuracyFromConfusionMatrix(C);
	std::cout << "Accuracy: " << accuracy << std::endl;

	waitKey(0);
	getchar();
	waitKey(0);
	getchar();


}

double Slope(int x0, int y0, int x1, int y1) {
	return (double)(y1 - y0) / (x1 - x0);
}

void fullLine(cv::Mat img, cv::Point a, cv::Point b, cv::Scalar color) {
	double slope = Slope(a.x, a.y, b.x, b.y);

	Point p(0, 0), q(img.cols, img.rows);

	p.y = -(a.x - p.x) * slope + a.y;
	q.y = -(b.x - q.x) * slope + b.y;

	line(img, p, q, color, 1, 8, 0);
}

void linearClassifies_perceptron_lab10() {
	Mat srcImg = imread("images_Perceptron/test01.bmp");
	int classes[] = { -1,1 };
	int rowX = 0;
	std::vector<std::pair<float, float>> features;
	std::vector<int> labels;
	for (int i = 0; i < srcImg.rows; i++) {
		for (int j = 0; j < srcImg.cols; j++) {
			if (!(srcImg.at<Vec3b>(i, j)[0] == 255 && srcImg.at<Vec3b>(i, j)[1] == 255 && srcImg.at<Vec3b>(i, j)[2] == 255)) {
				rowX++;
				features.push_back(std::make_pair((float)i, (float)j));
				if (srcImg.at<Vec3b>(i, j)[0] == 255) {
					labels.push_back(1);
				}
				if (srcImg.at<Vec3b>(i, j)[2] == 255) {
					labels.push_back(-1);
				}

			}

		}
	}
	Mat X = Mat(rowX, 3, CV_64FC1);
	Mat y = Mat(rowX, 1, CV_64FC1);
	for (auto i = 0; i != features.size(); i++) {
		X.at<double>(i, 0) = 1.0;
		X.at<double>(i, 1) = (double)features[i].first;
		X.at<double>(i, 2) = (double)features[i].second;
		y.at<double>(i) = (double)labels[i];
	}
	double n = 0.0001;
	int maxIterations = 100000;
	double w[3] = { 0.1,0.1,0.1 };
	double ELimit = 0.00001;
	

	for (int i = 0; i < maxIterations; i++) {
		double E = 0.0;
		for (int j = 0; j < rowX; j++) {
			double z = 0.0;
			for (int d = 0; d < 3; d++) {
				z += w[d] * X.at<double>(j, d);
			}
			if (z * y.at<double>(j) <= 0) {
				for (int c = 0; c < 3; c++) {
					w[c] += n * X.at<double>(j, c) * y.at<double>(j);
				}
				E += 1.0f;
			}
		}
		E = E / (double)rowX;
		if (E < ELimit)
			break;
	}

	Point2d one(1.0, 0.0);
	one.x = -w[0] / w[2];
	Point2d two(0.0, 1.0);
	two.y = -w[0] / w[1];
	fullLine(srcImg, one, two, Scalar(0, 255, 0));
	imshow("title", srcImg);
	waitKey(0);
}



struct weaklearner {
	int feature_i;
	int threshold;
	int classLabel;
	float error;

	int classify(Mat X) {
		if (X.at<double>(feature_i) < threshold)
			return classLabel;
		else
			return -classLabel;
	}
};

struct classifier {
	int T;
	float alphas[1000];
	weaklearner hs[1000];

	int classify(Mat X) {
		int sum = 0;
		for (int i = 0; i < T; i++) {
			sum += hs[i].classify(X);
		}
		return (sum > 0) - (sum < 0);
	}
};

weaklearner findWeakLearner(Mat X, Mat y, Mat w, int imgSize) {
	weaklearner best_h;
	int classLabels[2] = { -1, 1 };
	double best_err = FLT_MAX;
	std::vector<double> z(X.rows);
	for (int j = 0; j < X.cols; j++) {
		for (int treshold = 0; treshold < imgSize; treshold++) {
			for (int c = 0; c < 2; c++) {
				double e = 0.0f;
				for (int i = 0; i < X.rows; i++) {

					if (X.at<double>(i, j) < treshold) {
						z.at(i) = (classLabels[c]);
					}
					else {
						z.at(i) = (-classLabels[c]);
					}


					if (z.at(i)*y.at<double>(i) < 0) {
						e += w.at<double>(i);
					}
				}
				if (e < best_err) {
					best_err = e;
					best_h.threshold = treshold;
					best_h.error = e;
					best_h.classLabel = classLabels[c];
					best_h.feature_i = j;
				}
			}
		}
	}
	return best_h;

}

void drawBoundary(Mat srcImg, classifier clf) {
	for (int i = 0; i < srcImg.rows; i++) {
		for (int j = 0; j < srcImg.cols; j++) {
			if ((srcImg.at<Vec3b>(i, j)[0] == 255 && srcImg.at<Vec3b>(i, j)[1] == 255 && srcImg.at<Vec3b>(i, j)[2] == 255)) {
				Mat X = Mat(1, 2, CV_64FC1);
				X.at<double>(0, 0) = (double) i;
				X.at<double>(0, 1) = (double) j;
				if (clf.classify(X) == 1) {
					srcImg.at<Vec3b>(i, j)[0] = 0;
					srcImg.at<Vec3b>(i, j)[1] = 255;
					srcImg.at<Vec3b>(i, j)[2] = 255;
				}
				else {
					srcImg.at<Vec3b>(i, j)[0] = 225;
					srcImg.at<Vec3b>(i, j)[1] = 225;
					srcImg.at<Vec3b>(i, j)[2] = 0;

				}
			}
		}
	}
}

void adaBoost_lab11(int T) {

	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat srcImg = imread(fname);
		int classes[] = { -1, 1 };
		int rowX = 0;
		std::vector<std::pair<float, float>> features;
		std::vector<int> labels;
		for (int i = 0; i < srcImg.rows; i++) {
			for (int j = 0; j < srcImg.cols; j++) {
				if (!(srcImg.at<Vec3b>(i, j)[0] == 255 && srcImg.at<Vec3b>(i, j)[1] == 255 && srcImg.at<Vec3b>(i, j)[2] == 255)) {
					rowX++;
					features.push_back(std::make_pair((float)i, (float)j));
					if (srcImg.at<Vec3b>(i, j)[0] == 255) {
						labels.push_back(-1);
					}
					if (srcImg.at<Vec3b>(i, j)[2] == 255) {
						labels.push_back(1);
					}

				}

			}
		}
		Mat X = Mat(rowX, 2, CV_64FC1);
		Mat y = Mat(rowX, 1, CV_64FC1);
		Mat w = Mat(rowX, 1, CV_64FC1);
		for (int i = 0; i != features.size(); i++) {
			X.at<double>(i, 0) = (double)features[i].first;
			X.at<double>(i, 1) = (double)features[i].second;
			y.at<double>(i) = (double)labels[i];
			w.at<double>(i) = 1.0 / rowX;
		}
		std::vector<double>alfa;
		classifier classifier;
		for (int t = 0; t < T; t++) {
			weaklearner h = findWeakLearner(X, y, w, srcImg.rows);
			alfa.push_back(0.5* (log((1 - h.error) / h.error)));
			double s = 0.0;

			for (int i = 0; i < rowX; i++) {
				double newW = w.at<double>(i) * exp(-alfa.at(t) * y.at<double>(i) *h.classify(X.row(i)));
				w.at<double>(i) = newW;
				s += w.at<double>(i);
			}

			for (int i = 0; i < rowX; i++) {
				double normalizedW = w.at<double>(i) / s;
				w.at<double>(i) = normalizedW;
			}
			classifier.alphas[t] = alfa.at(t);
			classifier.T = T;
			classifier.hs[t] = h;
		}

		drawBoundary(srcImg, classifier);
		imshow("result", srcImg);
		waitKey(0);
	}
}


//---------------PROJECT---------------------
// Manduchi curb localization

//utility struct for the hought transform local maxima storage and sorting
struct peak {
	int theta, ro, hval;
	bool operator < (const peak& o) const {
		return hval > o.hval;
	}
};

double degToRad(int deg) {

	return (deg*PI) / 180.0;
}

Mat houghTransform(Mat srcImg, Mat bigSource) {

	Mat srcImgRgb(srcImg.size(), CV_8UC3);
	cvtColor(srcImg, srcImgRgb, CV_GRAY2RGB);
	//compute maxP  using pythagorean theorem . maxP =sqrt( widht^2 + height^2)
	int maxP = sqrt(srcImg.rows*srcImg.rows + srcImg.cols*srcImg.cols);
	int maxTheta = 360;
	int deltaTheta = 1;
	int deltaP = 1;
	int p;
	Mat hough = Mat::zeros(360, maxP + 1, CV_32SC1);
	Mat houghImg;
	//detect edges with canny edge 
	Mat gauss;
	Mat imageWithEdges;
	double k = 0.4;
	int pH = 50;
	int pL = (int)k*pH;
	GaussianBlur(srcImg, gauss, Size(5, 5), 0.8, 0.8);
	Canny(gauss, imageWithEdges, pL, pH, 3);


	for (int i = 0; i < imageWithEdges.rows; i++) {
		for (int j = 0; j < imageWithEdges.cols; j++) {
			//if edge point
			if (imageWithEdges.at<uchar>(i, j) == 255) {
				for (int theta = 0; theta < maxTheta; theta++) {
					p = j * cos(degToRad(theta)) + i * sin(degToRad(theta));
					if (p > 0 && p < maxP) {
						hough.at<int>(theta, p)++;
					}
				}
			}
		}
	}
	double min, max;
	minMaxLoc(hough, &min, &max);
	hough.convertTo(houghImg, CV_8UC1, 255.f / (float)max);
	//peak struct array to store the found peaks
	peak peaks[20000];
	int numberOfPeaks = 0;
	bool isGreater;
	//use 9x9 mask
	for (int theta = 4; theta < maxTheta - 4; theta += deltaTheta) {
		for (int p = 4; p < maxP - 4; p += deltaP) {
			isGreater = true;
			for (int k = theta - 4; k <= theta + 4; k++) {
				//test current element against all elements in the k x k window
				for (int l = p - 4; l <= p + 4; l++) {
					if (hough.at<int>(theta, p) == 0 || hough.at<int>(theta, p) < hough.at<int>(k, l)) {
						isGreater = false;
					}
				}
			}
			if (isGreater == true) {
				peaks[numberOfPeaks].hval = hough.at<int>(theta, p);
				peaks[numberOfPeaks].ro = p;
				peaks[numberOfPeaks].theta = theta;
				numberOfPeaks++;
			}
		}
	}
	std::sort(peaks, peaks + numberOfPeaks);
	//convert polar coordinates in cartesian coordinates and draw lines
	for (int i = 0; i < 10; i++) {
		Point pt1, pt2;
		double a = cos(degToRad(peaks[i].theta)), b = sin(degToRad(peaks[i].theta));
		double x0 = a * peaks[i].ro, y0 = b * peaks[i].ro;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(bigSource, pt1, pt2, Scalar(0, 255, 0));
		line(srcImgRgb, pt1, pt2, Scalar(0, 255, 0));
		std::cout << peaks[i].hval << std::endl;
	}

	imshow("Image->hough", srcImgRgb);

	return bigSource;

}

#define STRONG_EDGE 255
#define WEAK_EDGE 122
int H[256];

Mat filterFunction(Mat src, Mat mask)
{
	Mat dst(src.size(), CV_8UC1);

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			dst.at<uchar>(i, j) = src.at<uchar>(i, j);

	float sum = 0;
	double t = (double)getTickCount();
	for (int i = 0 + (mask.rows / 2); i < src.rows - (mask.rows / 2); i++)
		for (int j = 0 + (mask.cols / 2); j < src.cols - (mask.cols / 2); j++)
		{
			sum = 0;
			for (int u = 0; u < mask.rows; u++)
				for (int v = 0; v < mask.cols; v++)
				{
					sum = sum + (float)mask.at<float>(u, v)*src.at<uchar>(i + u - (mask.rows / 2), j + v - (mask.cols / 2));
				}
			if (sum < 0)
				sum = 0;
			if (sum > 255)
				sum = 255;
			dst.at<uchar>(i, j) = sum;
		}

	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("Time = %.3f [ms]\n", t * 1000);
	return dst;
}

Mat gaussFilter(Mat image) {

	Mat output(image.size(), CV_8UC1);


	int dimension = 3;

	float sigma = (float)dimension / 6;

	Mat gaussianFilter(dimension, dimension, CV_32F);

	for (int i = 0; i < gaussianFilter.rows; i++)
		for (int j = 0; j < gaussianFilter.cols; j++)
		{
			gaussianFilter.at<float>(i, j) = (float)(1 / (2 * CV_PI*pow(sigma, 2))) * exp(-(pow((int)(i - (dimension / 2)), 2) + pow((int)(j - (dimension / 2)), 2)) / (2 * pow(sigma, 2)));
		}

	Mat dst(image.size(), CV_8UC1);
	dst = filterFunction(image, gaussianFilter);

	return dst;
}


Mat calcGradients(Mat src) {

	char fname[MAX_PATH];

		Mat img = Mat(src.rows, src.cols, CV_8UC1);
		Mat gradX = Mat::zeros(img.rows, img.cols, CV_32F);
		Mat gradY = Mat::zeros(img.rows, img.cols, CV_32F);
		Mat gradImg = Mat::zeros(img.rows, img.cols, CV_32F);
		Mat ucharGrad = Mat(img.size(), CV_8UC1);
		Mat magImg = Mat::zeros(img.rows, img.cols, CV_8UC1);

		img = gaussFilter(src);

		for (int i = 1; i < img.rows - 1; i++) {
			for (int j = 1; j < img.cols - 1; j++) {
				gradX.at<float>(i, j) = ((-2) * img.at<uchar>(i, j - 1)) - img.at<uchar>(i - 1, j - 1) - img.at<uchar>(i + 1, j - 1) + img.at<uchar>(i + 1, j + 1) + img.at<uchar>(i - 1, j + 1) + 2 * img.at<uchar>(i, j + 1);
				gradY.at<float>(i, j) = (2 * img.at<uchar>(i - 1, j)) + img.at<uchar>(i - 1, j - 1) + img.at<uchar>(i - 1, j + 1) - img.at<uchar>(i + 1, j - 1) - img.at<uchar>(i + 1, j + 1) - (2 * img.at<uchar>(i + 1, j));
			}
		}

		float arctan = 0;

		for (int i = 1; i < img.rows - 1; i++) {
			for (int j = 1; j < img.cols - 1; j++) {
				float x = std::sqrt(std::pow(gradX.at<float>(i, j), 2.0f) + std::pow(gradY.at<float>(i, j), 2.0f));
				x = x / (4.0f * std::sqrt(2.0f));
				gradImg.at<float>(i, j) = x;
				//float grade;
				arctan = std::atan2f(gradY.at<float>(i, j), gradX.at<float>(i, j));

				if (arctan < 0) {
					arctan += 2 * CV_PI;
				}

				float grade = arctan * (180 / CV_PI);

				if ((grade < (0 + 45 / 2.0) || grade >(315 + 360) / 2.0) || grade > (135 + 180) / 2.0 && grade < (180 + 225) / 2.0) {
					magImg.at<uchar>(i, j) = 2; //2
				}
				else if (grade > (0 + 45 / 2.0) && grade < (45 + 90) / 2.0 || grade >(180 + 225) / 2.0 && grade < (225 + 270) / 2.0) {
					magImg.at<uchar>(i, j) = 1; //1
				}
				else if (grade > (45 + 90) / 2.0 && grade < (90 + 135) / 2.0 || grade >(225 + 270) / 2.0 && grade < (270 + 315) / 2.0) {
					magImg.at<uchar>(i, j) = 0; //0
				}
				else
					magImg.at<uchar>(i, j) = 3; //3


			}
		}

		Mat direction = Mat(img.size(), CV_8UC1);

		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				ucharGrad.at<uchar>(i, j) = (uchar)gradImg.at<float>(i, j);
			}
		}

		for (int i = 1; i < img.rows - 1; i++) {
			for (int j = 1; j < img.cols - 1; j++) {
				if (magImg.at<uchar>(i, j) == 0) {
					if (ucharGrad.at<uchar>(i, j) > ucharGrad.at<uchar>(i - 1, j) && ucharGrad.at<uchar>(i, j) > ucharGrad.at<uchar>(i + 1, j)) {
						direction.at<uchar>(i, j) = ucharGrad.at<uchar>(i, j);
					}
					else
						direction.at<uchar>(i, j) = 0;
				}
				else if (magImg.at<uchar>(i, j) == 1) {
					if (ucharGrad.at<uchar>(i, j) > ucharGrad.at<uchar>(i - 1, j + 1) && ucharGrad.at<uchar>(i, j) > ucharGrad.at<uchar>(i + 1, j - 1)) {
						direction.at<uchar>(i, j) = ucharGrad.at<uchar>(i, j);
					}
					else
						direction.at<uchar>(i, j) = 0;
				}
				else if (magImg.at<uchar>(i, j) == 2) {
					if (ucharGrad.at<uchar>(i, j) > ucharGrad.at<uchar>(i, j - 1) && ucharGrad.at<uchar>(i, j) > ucharGrad.at<uchar>(i, j + 1)) {
						direction.at<uchar>(i, j) = ucharGrad.at<uchar>(i, j);
					}
					else
						direction.at<uchar>(i, j) = 0;
				}
				else if (magImg.at<uchar>(i, j) == 3) {
					if (ucharGrad.at<uchar>(i, j) > ucharGrad.at<uchar>(i - 1, j - 1) && ucharGrad.at<uchar>(i, j) > ucharGrad.at<uchar>(i + 1, j + 1)) {
						direction.at<uchar>(i, j) = ucharGrad.at<uchar>(i, j);
					}
					else
						direction.at<uchar>(i, j) = 0;
				}
			}
		}

		for (int i = 0; i < 256; i++) {
			H[i] = 0;
		}

		for (int i = 1; i < img.rows - 1; i++) {
			for (int j = 1; j < img.cols - 1; j++) {
				H[direction.at<uchar>(i, j)] ++;

			}
		}

		for (int i = 0; i < 256; i++) {
			std::cout << H[i] << " ";
		}

		float p = 0.29f;
		int noNoEdge = (1 - p) * ((img.cols - 3) * (img.rows - 3) - H[0]);

		int adaptiveTh = 0, sum = 0;

		for (int i = 1; i < 256; i++) {
			if (sum >= noNoEdge) {
				adaptiveTh = i;
				break;
			}
			else {
				sum += H[i];
			}
		}
		std::cout << std::endl << img.cols*img.rows - H[0] << std::endl << adaptiveTh << std::endl;

		int th = adaptiveTh;
		int tl = 0.4 * th;

		Mat dst = Mat(direction.size(), CV_8UC1);


		for (int i = 1; i < direction.rows; i++) {
			for (int j = 1; j < direction.cols; j++) {
				if (direction.at<uchar>(i, j) >= th) {
					dst.at<uchar>(i, j) = STRONG_EDGE;
				}
				else if (direction.at<uchar>(i, j) >= tl) {
					dst.at<uchar>(i, j) = WEAK_EDGE;
				}
				else {
					dst.at<uchar>(i, j) = 0;
				}
			}
		}


		std::queue<Point2i> Q;

		int di[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
		int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };



		for (int i = 1; i < dst.rows - 1; i++) {
			for (int j = 1; j < dst.cols - 1; j++) {
				if (dst.at<uchar>(i, j) == STRONG_EDGE) {

					Q.push({ i, j });

					while (!Q.empty()) {
						Point2i q = Q.front();
						Q.pop();
						for (int k = 0; k < 8; k++) {
							if (dst.at<uchar>(q.x + di[k], q.y + dj[k]) == WEAK_EDGE) {
								dst.at<uchar>(q.x + di[k], q.y + dj[k]) = STRONG_EDGE;
								Q.push({ q.x + di[k], q.y + dj[k] });
							}
						}
					}
				}
			}
		}

		for (int i = 1; i < dst.rows - 1; i++) {
			for (int j = 1; j < dst.cols - 1; j++) {
				if (dst.at<uchar>(i, j) == WEAK_EDGE) {
					dst.at<uchar>(i, j) = 0;
				}
			}
		}


		//imshow("Initial Image ", img);
		//imshow("Directionimg", magImg * 63);
		//imshow("HOPEFULLY", direction);
		//imshow("Gradient", gradImg);
		//imshow("Hopefully", ucharGrad);
		//imshow("thresh", dst);

		return ucharGrad;
}

#define B_DISP 218.0
#define F_DISP 811.0

struct point3d {
	float x, y, z;
	int i, j;
};

void proj() {

	Mat srcImg = imread("images_manduchi/A000004-1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat d = imread("images_manduchi/D000004.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	Mat zs = Mat::zeros(d.size(), CV_32F);
	Mat ys = Mat::zeros(d.size(), CV_32F);

	float z = 0.0f, y = 0.0f;

	for (int i = d.rows/3; i < d.rows; i++) {
		for (int j = 0; j < d.cols; j++) {
			z = 0.0f;
			y = 0.0f;
			if (d.at<uchar>(i, j) != 1) {
				z = (float)B_DISP * F_DISP / (d.at<uchar>(i, j) / 4.0);
				y = (d.rows / 2.0 - i)*z / F_DISP;

				zs.at<float>(i, j) = z;

				if (y > -2000.0f && y < 2500.0f)
						ys.at<float>(i, j) = y;
				else
					ys.at<float>(i, j) = 0.0f;


				//std::cout << " " << y;
			}
		}
	}

	Mat zsNorm = Mat(zs.size(), CV_8UC1);
	Mat ysNorm = Mat(ys.size(), CV_8UC1);


	cv::normalize(zs, zsNorm, 0, 255, NORM_MINMAX, CV_8UC1);
	cv::normalize(ys, ysNorm, 0, 255, NORM_MINMAX, CV_8UC1);

	imshow("Original", srcImg);
	//imshow("Normalized disparity z", zsNorm);
	//imshow("Normalized disparity y", ysNorm);

	/*for (int i = 0; i < ys.rows; i++ ) {
		for (int j = 0; j < ys.cols; j++) {
			if(ys.at<float>(i, j)!=0.0f)
			std::printf("%x , %f\n", ysNorm.at<uchar>(i,j), ys.at<float>(i,j));
		}
	}*/

	Mat C = Mat(ys.size(), CV_32SC1);

	Mat yGrad = calcGradients(ysNorm);
	Mat iGrad = calcGradients(srcImg);

	imshow("Grad img brightness", iGrad);
	imshow("Grad disparity y", yGrad);

	for (int i = 0; i < yGrad.rows; i++) {
		for (int j = 0; j < yGrad.cols; j++) {
			int c = (int)yGrad.at<uchar>(i, j) * iGrad.at<uchar>(i, j);
			if (c < 8 && c > 3)
				C.at<int>(i, j) = c;
			else
				C.at<int>(i, j) = 0;
		}
	}

	Mat cNorm = Mat(C.size(), CV_8UC1);

	cv::normalize(C, cNorm, 0, 255, NORM_MINMAX, CV_8UC1);

	//std::cout << CNorm;

	imshow("C", cNorm);

	Mat srcImgRgb(srcImg.size(), CV_8UC3);
	cvtColor(srcImg, srcImgRgb, CV_GRAY2RGB);
	imshow("Hough", houghTransform(cNorm, srcImg));

	//std::cout << C;
	/*for (int i = yGrad.rows/2; i < yGrad.rows; i++) {
		for (int j = yGrad.cols/2; j < yGrad.cols; j++) {
			if (C.at<int>(i, j) != 0) {
				std::cout << C.at<int>(i, j) << " ";
			}
		}
		std::cout << '\n';
	}*/

	waitKey(0);

}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 12 - Hough\n");
		printf(" 13 - Distance Transform\n");
		printf(" 14 - Distance Transform With Translation\n");
		printf(" 15 - Statistical data analysis\n");
		printf(" 16 - K-means clustering points\n");
		printf(" 17 - K-means clustering grey\n");
		printf(" 18 - K-means clustering color\n");
		printf(" 19 - pca\n");
		printf(" 20 - KNN\n");
		printf(" 21 - Bayesian classifier\n");
		printf(" 22 - Perceptron\n");
		printf(" 23 - ADABOOST\n");

		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 12:
				hough_tf();
				break;
			case 13:
				distanceTransform();
				break;
			case 14:
				distanceTransform_withTranslation();
				break;
			case 15:
				statistical_data_analysis_lab5();
				break;
			case 16:
				kMeans_points(10, 10);
				break;
			case 17:
				KMeansClustering_lab06_grayScale(2,10);
				break;
			case 18:
				KMeansClustering_lab06_RGB(2,10);
				break;

			case 19:
				pca();
				break;
			case 20:
				KNN_classifier_lab08(7);
				break;
			case 21:
				bayesian_lab09();
				break;
			case 22:
				linearClassifies_perceptron_lab10();
				break;
			case 23:
				adaBoost_lab11(12);
				break;

			case 1001:
				proj();
				break;
		}
	}
	while (op!=0);
	return 0;
}