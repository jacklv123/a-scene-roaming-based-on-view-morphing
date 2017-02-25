#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv/cvaux.h>
#include<eigen/eigen>
#include<vector>
#include<opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/features2d.hpp>

#define pii pair<int,int>
#define A first;
#define B second
#define mp make_pair

using namespace std;
using namespace cv;
const int imgnum = 7;//用于图像变形的图片张数
const char pfix[30]="d";//图片名称前缀
const char img_type[30] = "jpg";//图片格式
char imgname[imgnum][20];//存储图片的名字

double fscale = 0.5;//图片缩放比
vector<pii> pos1,pos2;//相邻两张图片匹配特征点的位置

IplImage* im[imgnum] ,*imn,*res;
Mat m_Fundamental;//基础矩阵
CvMatrix3 *F;
CvSize cvsize;

int *scanlines[imgnum][2],*scanlinesVirtual;
int *length[imgnum][2], *lengthVirtual;
int line_count[imgnum];

uchar *dst[imgnum][2], *morphed;

int *run[imgnum][2];
int *num_run[imgnum][2];
int *corr[imgnum][2];

double alpha=1;
const double delta = 0.01;//改变delta可以改变图像变化的速度
int max_line_count=0;

void on_mouse1(int event, int x, int y, int flags, void* ustc);
void on_mouse2(int event, int x, int y, int flags, void* ustc);
void init();
void loadpic();
void Siftgetfeatures(int i);
void findfundmat(int i);
void MakeScanlines(int i);
void PreWarpImage(int i);
void FindRuns(int i);
void DynamicCorrespondMulti(int i);
void MakeAlphaScanlines(int i);
void MorphEpilinesMulti(int i);
void PostWarpImage(int i);

int main(){

	loadpic();
	init();
	cvNamedWindow("view1", 1);
	cvSetMouseCallback("view1", on_mouse1, 0);
	cvShowImage("view1", im[0]);

	cvNamedWindow("view2", 1);
	cvSetMouseCallback("view2", on_mouse2, 0);
	cvShowImage("view2", im[1]);

	for (int i = 0; i < imgnum-1; i++){
		Siftgetfeatures(i); 
		findfundmat(i); 
		MakeScanlines(i); 
		PreWarpImage(i);
		FindRuns(i);
		DynamicCorrespondMulti(i);
		max_line_count = max(max_line_count, line_count[i]);
	}

	scanlinesVirtual = new int[max_line_count * 10];
	lengthVirtual = new int[max_line_count * 10];
	morphed = new uchar[max(cvsize.height, cvsize.width)*max_line_count * 5 + 10];

	cvNamedWindow("view3", 1);
	cvShowImage("view3", res);
	cvWaitKey(0);

	int num = 0;//第i张图到第i+1张图变化
	alpha = 1;
	while (1){
		int key = cvWaitKey(10);
		if (key == 27){
			break;//esc
		}
		//通过点击m，n按键来变化图像
		//相邻两幅图像的渐变由alpha决定，alpha在[0,1]之间
		//当alpha大于1或者小于0时要考虑改变变化的图像
		switch (key){
			case 'n':
				if (alpha+delta<=1.0)
					alpha += delta;
				else if (num -1 >=0 ){
					num--;
					alpha = 0;
				}
				break;
			case 'm':
				if (alpha-delta>=0.0)
					alpha -= delta;
				else if (num +2  < imgnum){
					num++;
					alpha = 1;
				}
				break;
			default:
				break;
		}
		//cout << "alpha=" << alpha << endl;
		MakeAlphaScanlines(num);
		MorphEpilinesMulti(num);
		PostWarpImage(num);
		cvDeleteMoire(res);

		cvShowImage("view3", res);
	}
	
	cvDestroyAllWindows();
	for (int i = 0; i < imgnum; i++){
		cvReleaseImage(&im[i]);
	}
	return 0;

}
void loadpic(){//加载图片名字
	string st;
	char mid[20];
	for (int i = 0; i < imgnum; i++){
		sprintf(mid, "%d", i);
		st = (string)(pfix) + (string)(mid) + "."+(string)(img_type);
		strcpy(imgname[i], st.c_str());
		printf("%s\n", imgname[i]);
	}
}
void init(){//加载图片数据
	for (int i = 0; i < imgnum; i++){
		im[i] = cvLoadImage(imgname[i], 1);
	}
	
	cvsize.width = im[0]->width*fscale;
	cvsize.height = im[0]->height*fscale;

	for (int i = 0; i < imgnum; i++){//改变图像大小
		imn = cvCreateImage(cvsize, im[i]->depth, im[i]->nChannels);
		cvResize(im[i], imn, CV_INTER_AREA);
		*im[i] = *imn;
	}

	res = cvLoadImage(imgname[0], 1);
	imn = cvCreateImage(cvsize, res->depth, res->nChannels);
	cvResize(res, imn, CV_INTER_AREA);
	*res = *imn;

	F = new CvMatrix3;
}

void findfundmat(int i){//求基础矩阵
	int ptcount = pos1.size();
	Mat p1(ptcount, 2, CV_32F);
	Mat p2(ptcount, 2, CV_32F);
	pii pt;
	for (int i = 0; i < ptcount; i++){
		pt = pos1[i];
		p1.at<float>(i, 0) = pt.A;		
		p1.at<float>(i, 1) = pt.B;

		pt = pos2[i];
		p2.at<float>(i, 0) = pt.A;
		p2.at<float>(i, 1) = pt.B;
	}
	
	vector<uchar> m_RANSACStatus;
	m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);
	
	//for (int i = 0; i < 3;i++)for (int j = 0; j < 3; j++)
		//cout << m_Fundamental[i].at<double>(i, j) << endl;
}
/*Calculates scanlines coordinates for two cameras by fundamental matrix
void cvMakeScanlines( const CvMatrix3* matrix, CvSize img_size, int* scanlines1,
int* scanlines2, int* lengths1, int* lengths2, int* line_count );



matrix:: Fundamental matrix.imgSize:: Size of the image.scanlines1:: Pointer to the array of calculated scanlines of the first image.scanlines2:: Pointer to the array of calculated scanlines of the second image.lengths1:: Pointer to the array of calculated lengths (in pixels) of the first image scanlines.lengths2:: Pointer to the array of calculated lengths (in pixels) of the second image scanlines.line_count:: Pointer to the variable that stores the number of scanlines.
The function cvMakeScanlines
finds coordinates of scanlines for two images. This function returns the number of scanlines. The function does nothing except calculating the number of scanlines if the pointers scanlines1
or scanlines2
*/
void MakeScanlines(int i){
	//cout << i << endl;
	for (int i = 0; i < 3;i++)
	for (int j = 0; j < 3; j++)
		(*F).m[i][j] = m_Fundamental.at<double>(i, j);// cout << (*F).m[i][j] << endl;
	//cout << m_Fundamental.rows << ' ' << m_Fundamental.cols << endl;
	cvMakeScanlines(F, cvsize, 0, 0, 0, 0, &line_count[i]);
	cout << "line_count "<<i<<"=" << line_count[i] << endl;
	scanlines[i][0] = new int[line_count[i] * 4 + 10];
	scanlines[i][1] = new int[line_count[i] * 4 + 10];
	length[i][0] = new int[line_count[i] * 4 + 10];
	length[i][1] = new int[line_count[i] * 4 + 10];
	cvMakeScanlines(
		F, 
		cvsize,
		scanlines[i][0], scanlines[i][1], 
		length[i][0], length[i][1], 
		&line_count[i]);

	cout << "MakeScanlines "<<i<<" is done!" << endl;
}
/*Rectifies image
void cvPreWarpImage( int line_count, IplImage* img, uchar* dst,
int* dst_nums, int* scanlines );



line_count:: Number of scanlines for the image.img:: Image to prewarp.dst:: Data to store for the prewarp image.dst_nums:: Pointer to the array of lengths of scanlines.scanlines:: Pointer to the array of coordinates of scanlines.
The function cvPreWarpImage
rectifies the image so that the scanlines in the rectified image are horizontal. The output buffer of size max(width,height)*line_count*3
must be allocated before calling the function.
*/
void PreWarpImage(int i){
	int len = max(cvsize.height, cvsize.width)*line_count[i] * 5 + 10;
	dst[i][0] = new uchar[len];
	dst[i][1] = new uchar[len];
	
	cvPreWarpImage(line_count[i], 
		im[i], dst[i][0], length[i][0], scanlines[i][0]);
	cvPreWarpImage(line_count[i], 
		im[i+1], dst[i][1], length[i][1], scanlines[i][1]);
	cout << "PreWarpImage "<<i<<" is done!" << endl;
}
/*Retrieves scanlines from rectified image and breaks them down into runs
void cvFindRuns( int line_count, uchar* prewarp1, uchar* prewarp2,
int* line_lengths1, int* line_lengths2,
int* runs1, int* runs2,
int* num_runs1, int* num_runs2 );

line_count:: Number of the scanlines.prewarp1:: Prewarp data of the first image.prewarp2:: Prewarp data of the second image.line_lengths1:: Array of lengths of scanlines in the first image.line_lengths2:: Array of lengths of scanlines in the second image.runs1:: Array of runs in each scanline in the first image.runs2:: Array of runs in each scanline in the second image.num_runs1:: Array of numbers of runs in each scanline in the first image.num_runs2:: Array of numbers of runs in each scanline in the second image.
The function cvFindRuns
retrieves scanlines from the rectified image and breaks each scanline down into several runs, that is, series of pixels of almost the same brightness.*/

void FindRuns(int i){
	int len = cvsize.height*cvsize.width * 3 + 10;
	run[i][0] = new int[len];
	run[i][1] = new int[len];
	num_run[i][0] = new int[len];
	num_run[i][1] = new int[len];
	cvFindRuns(
		line_count[i],
		dst[i][0], dst[i][1],
		length[i][0], length[i][1],
		run[i][0], run[i][1],
		num_run[i][0], num_run[i][1]
		);
	cout << "FindRuns "<<i<<" is done!" << endl;
}
/*Finds correspondence between two sets of runs of two warped images
void cvDynamicCorrespondMulti( int line_count, int* first, int* first_runs,
int* second, int* second_runs,
int* first_corr, int* second_corr );

line_count:: Number of scanlines.first:: Array of runs of the first image.first_runs:: Array of numbers of runs in each scanline of the first image.second:: Array of runs of the second image.second_runs:: Array of numbers of runs in each scanline of the second image.first_corr:: Pointer to the array of correspondence information found for the first runs.second_corr:: Pointer to the array of correspondence information found for the second runs.
The function cvDynamicCorrespondMulti
finds correspondence between two sets of runs of two images. Memory must be allocated before calling this function. Memory size for one array of correspondence information is max( width,height )* numscanlines*3*sizeof ( int )
*/
void DynamicCorrespondMulti(int i){
	int len = max(cvsize.height,cvsize.width) * line_count[i]*4 + 10;
	corr[i][0] = new int[len];
	corr[i][1] = new int[len];
	cvDynamicCorrespondMulti(
		line_count[i],
		run[i][0], num_run[i][0],
		run[i][1], num_run[i][1],
		corr[i][0],
		corr[i][1]
		);
	cout << "DynamicCorrespondMulti "<<i<<" is done!" << endl;
}
/*Calculates coordinates of scanlines of image from virtual camera
void cvMakeAlphaScanlines( int* scanlines1, int* scanlines2,
int* scanlinesA, int* lengths,
int line_count, float alpha );

scanlines1:: Pointer to the array of the first scanlines.scanlines2:: Pointer to the array of the second scanlines.scanlinesA:: Pointer to the array of the scanlines found in the virtual image.lengths:: Pointer to the array of lengths of the scanlines found in the virtual image.line_count:: Number of scanlines.alpha:: Position of virtual camera (0.0 - 1.0)
. The function cvMakeAlphaScanlines
finds coordinates of scanlines for the virtual camera with the given camera position. Memory must be allocated before calling this function. Memory size for the array of correspondence runs is numscanlines*2*4*sizeof(int)
. Memory size for the array of the scanline lengths is numscanlines*2*4*sizeof(int)
*/
void MakeAlphaScanlines(int i){
	cvMakeAlphaScanlines(
		scanlines[i][0],
		scanlines[i][1],
		scanlinesVirtual,
		lengthVirtual,
		line_count[i],
		alpha
		);
	//cout << "MakeAlphaScanlines is done!" << endl;
}
/*Morphs two pre-warped images using information about stereo correspondence
void cvMorphEpilinesMulti( int line_count, uchar* first_pix, int* first_num,
uchar* second_pix, int* second_num,
uchar* dst_pix, int* dst_num,
float alpha, int* first, int* first_runs,
int* second, int* second_runs,
int* first_corr, int* second_corr );

line_count:: Number of scanlines in the prewarp image.first_pix:: Pointer to the first prewarp image.first_num:: Pointer to the array of numbers of points in each scanline in the first image.second_pix:: Pointer to the second prewarp image.second_num:: Pointer to the array of numbers of points in each scanline in the second image.dst_pix:: Pointer to the resulting morphed warped image.dst_num:: Pointer to the array of numbers of points in each line.alpha:: Virtual camera position (0.0 - 1.0)
.first:: First sequence of runs.first_runs:: Pointer to the number of runs in each scanline in the first image.second:: Second sequence of runs.second_runs:: Pointer to the number of runs in each scanline in the second image.first_corr:: Pointer to the array of correspondence information found for the first runs.second_corr:: Pointer to the array of correspondence information found for the second runs. The function cvMorphEpilinesMulti
morphs two pre-warped images using information about correspondence between the scanlines of two images.*/
void MorphEpilinesMulti(int i){
	
	cvMorphEpilinesMulti(
		line_count[i],
		dst[i][0], length[i][0],
		dst[i][1], length[i][1],
		morphed, lengthVirtual,
		alpha,//alpha
		run[i][0], num_run[i][0],
		run[i][1], num_run[i][1],
		corr[i][0],
		corr[i][1]
		);
	//cout << "MorphEpilinesMulti is done!" << endl;
}
/*Warps rectified morphed image back
void cvPostWarpImage( int line_count, uchar* src, int* src_nums,
IplImage* img, int* scanlines );

line_count:: Number of the scanlines.src:: Pointer to the prewarp image virtual image.src_nums:: Number of the scanlines in the image.img:: Resulting unwarp image.scanlines:: Pointer to the array of scanlines data.
The function cvPostWarpImage
warps the resultant image from the virtual camera by storing its rows across the scanlines whose coordinates are calculated by cvMakeAlphaScanlines
*/
void PostWarpImage(int i){
	
	cvPostWarpImage(
		line_count[i],
		morphed,
		lengthVirtual,
		res,
		scanlinesVirtual
		);
	
	//cout << "PostWarpImage is done!" << endl;
}
//手动提取匹配特征点
void on_mouse1(int event, int x, int y, int flags, void* ustc)
{
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 0.5, 0.5, 0, 1, CV_AA);

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		CvPoint pt = cvPoint(x, y);
		char temp[16];
		sprintf(temp, "(%d,%d)", pt.x, pt.y);
		pos1.push_back(mp(pt.x, pt.y));
		cvPutText(im[0], temp, pt, &font, cvScalar(255, 255, 255, 0));
		cvCircle(im[0], pt, 2, cvScalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
		cvShowImage("view1", im[0]);
	}
}
//手动提取匹配特征点
void on_mouse2(int event, int x, int y, int flags, void* ustc)
{
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 0.5, 0.5, 0, 1, CV_AA);

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		CvPoint pt = cvPoint(x, y);
		char temp[16];
		sprintf(temp, "(%d,%d)", pt.x, pt.y);
		pos2.push_back(mp(pt.x, pt.y));
		cvPutText(im[1], temp, pt, &font, cvScalar(255, 255, 255, 0));
		cvCircle(im[1], pt, 2, cvScalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
		cvShowImage("view2", im[1]);
	}
}
//用sift提取两幅相邻图片的特征点，并做特征点匹配。
void Siftgetfeatures(int i){
	Mat img1(im[i], 1);
	Mat img2(im[i+1], 1);
	SurfFeatureDetector detector;
	vector<KeyPoint> keypoints1, keypoints2;
	detector.detect(img1, keypoints1, Mat());
	detector.detect(img2, keypoints2, Mat());
	SurfDescriptorExtractor extractor;
	Mat descriptor1, descriptor2;
	extractor.compute(img1, keypoints1, descriptor1);
	extractor.compute(img2, keypoints2, descriptor2);

	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(descriptor1, descriptor2, matches, Mat());
	Mat imgmatches;
	drawMatches(img1, keypoints1, img2, keypoints2, matches, imgmatches, Scalar::all(-1), Scalar::all(-1));
	imshow("Matches", imgmatches);
	pos1.clear();
	pos2.clear();
	
	for (int i = 0; i < matches.size(); i++){
		
		pii p;
		p = mp(
			keypoints1[matches[i].queryIdx].pt.x,
			keypoints1[matches[i].queryIdx].pt.y);
		pos1.push_back(p);
		p = mp(
			keypoints2[matches[i].trainIdx].pt.x,
			keypoints2[matches[i].trainIdx].pt.y);
		pos2.push_back(p);
	}
}