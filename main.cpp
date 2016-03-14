#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <QDebug>

using namespace cv;
using namespace std;

struct variables {
    int thresh = 128;

    //sharpen
    int alpha=1500, beta=342, gamma=6952;
    int size = 1;
    int sigmaX=3000; //Gaussian kernel standard deviation in X direction. = sigmaY

    //contrast
    int scale = 3685;
    int betaC = 66400;

    // after histogram equilisation, scale = 2000 (2.0) and beta = 50000 (50) & thresh 80 works pretty well
    // 3000, 56888, thresh = ~90
    // 3000, 51216, thresh = 79 --- slightly better
    // 5000, 100000, thresh = 180 --also good
    //2734, 48485, thresh = 73, ele = 4;


    //morhpological operations
    int sSize = 6;

    int ssSizeX = 15;
    int ssSizeY = 2;

    // crop parameters
    int x = 96, y = 8;
    int width = 168, height = 270;
    int imRows = 0, imCols =0;
} var;

//1931 218

//finds the most likely pipe object from contours
void findPipe(Mat image)
{
    vector< vector< Point >> contours;
    vector< vector< Point >> pts; //likely pipe object(s)

    findContours(image.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    cvtColor(image, image, CV_GRAY2BGR);

    if(contours.size() > 0)
    {
        for(int i = 0; i < contours.size(); i++)
        {
            if(contourArea(contours[i]) > 600)
            {
                Rect boundRect = boundingRect(contours[i]);
                rectangle( image, boundRect.tl()
                           , boundRect.br(), Scalar(255, 255, 0), 2, 8, 0 );
                pts.push_back(contours[i]);
            }
        }

        for(int i = 0; i < pts.size(); i++)
        {
            //Construct a buffer used by the pca analysis
            Mat data_pts = Mat(pts[i].size(), 2, CV_64FC1);
            for (int j = 0; j < data_pts.rows; ++j)
            {
                data_pts.at<double>(j, 0) = pts[i][j].x;
                data_pts.at<double>(j, 1) = pts[i][j].y;
            }

            //Perform PCA analysis
            PCA pca_analysis(data_pts, Mat(), CV_PCA_DATA_AS_ROW);

            //Store the position of the object
            Point pos = Point(pca_analysis.mean.at<double>(0, 0),
                              pca_analysis.mean.at<double>(0, 1));

            //Store the eigenvalues and eigenvectors
            vector<Point2d> eigen_vecs(2);
            vector<double> eigen_val(2);
            for (int i = 0; i < 2; ++i)
            {
                eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                        pca_analysis.eigenvectors.at<double>(i, 1));

                eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
            }

            double angle = atan2(eigen_vecs[1].y, eigen_vecs[1].x); // orientation in radians
            angle = abs((angle * 180/CV_PI)-180);

            cout << angle << endl;

            // object with largest eigenval in this angle range will be pipe
            if (angle > 165 && angle < 195) {
                // Draw the principal components
                circle(image, pos, 3, CV_RGB(255, 0, 255), 2);
                line(image, pos, pos + 0.02 * Point(eigen_vecs[0].x * eigen_val[0], eigen_vecs[0].y * eigen_val[0]) , CV_RGB(0, 255, 0)); // minor axis
                line(image, pos, pos + 0.02 * Point(eigen_vecs[1].x * eigen_val[1], eigen_vecs[1].y * eigen_val[1]) , CV_RGB(0, 255, 0)); // major axis
            }

            // ---------
        }
    }

    imshow("Pipe", image);

}

Mat crop(Mat image)
{

    line(image, Point(var.x, var.y), Point(var.x, var.y+var.height), Scalar(0,255,255), 2);
    line(image, Point(var.x+var.width, var.y), Point(var.x+var.width, var.y+var.height), Scalar(0,255,255), 2);
    line(image, Point(var.x, var.y), Point(var.x+var.width, var.y), Scalar(0,255,255), 2);
    line(image, Point(var.x, var.y+var.height), Point(var.x+var.width, var.y+var.height), Scalar(0,255,255), 2);
    imshow("Crop", image);
    Mat crop = image.clone()(Rect(var.x, var.y, var.width, var.height));
    return crop;
}

Mat detectLeak(Mat &frame)
{
    Mat frameClone = frame.clone();
    cvtColor(frame, frame, CV_BGR2GRAY);

    Mat dst;

    //sharpen
    GaussianBlur(frame, dst, Size(var.size, var.size), (double)var.sigmaX/1000);
    cv::addWeighted(frame, (double)var.alpha/1000 , dst, (double)-var.beta/1000, (double)var.gamma/1000, dst);

    imshow("Before", dst);
    dst = crop(dst);
    //contrast
    Mat frameC;
    //frame.convertTo(frameC, -1, (double)var.scale/1000, (double)var.betaC/1000); //increase the contrast (double)

    Mat equil;
    equalizeHist(dst, equil); //equalize the histogram
    equil.convertTo(frameC, -1, (double)var.scale/1000, (double)var.betaC/1000);
    imshow("Contrast", frameC);

    threshold(frameC, dst, var.thresh, 255, CV_THRESH_BINARY_INV);

    Mat element = getStructuringElement( MORPH_RECT, Size( var.sSize, var.sSize ), Point( -1, -1 ) );
    morphologyEx( dst, dst, MORPH_CLOSE, element );

    imshow("Result", dst);

    findPipe(dst.clone());
    element = getStructuringElement( MORPH_RECT, Size(var.ssSizeX, var.ssSizeY ), Point(-1, -1) );
    morphologyEx(dst, dst, MORPH_OPEN, element);
    imshow("morph open", dst);
    // 15 / 4
    // 15/6
    //adaptiveThreshold(frameC, dst, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, 15, 6);

    vector< vector<Point>> contours;

    findContours(dst.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    if(contours.size() > 0)
    {
        for(int i = 0; i<contours.size(); i++)
        {
            if(contourArea(contours[i]) < 100)
            {
                Rect boundRect = boundingRect(contours[i]);
                rectangle( frameClone, Point(boundRect.tl().x+var.x, boundRect.tl().y+var.y)
                           , Point(boundRect.br().x + var.x, boundRect.br().y+var.y), Scalar(255, 255, 0), 2, 8, 0 );

            }
        }

    }
    imshow("LEAK?", frameClone);
    return dst;
}

int main(int argc, char *argv[])
{

    VideoCapture vc;
    Mat frame;
    Mat result;

    vc.open("/home/zzaj/Videos/Nova Leak Videos/Latest/1.asf");
    if(!vc.isOpened()) {
        qDebug() << "Did not open";
    }


    namedWindow("Before", CV_WINDOW_NORMAL);
    namedWindow("Result", CV_WINDOW_NORMAL);
    namedWindow("Contrast", CV_WINDOW_NORMAL);
    namedWindow("morph open", CV_WINDOW_NORMAL);
    namedWindow("Crop", CV_WINDOW_NORMAL);

    createTrackbar("thresh", "Result", &var.thresh, 255);
    createTrackbar("contrast", "Contrast", &var.scale, 5000);
    createTrackbar("beta", "Contrast", &var.betaC, 100000);
    createTrackbar("Morph ele size", "Result", &var.sSize, 20);

    createTrackbar("alpha", "Before", &var.alpha, 10000);
    createTrackbar("beta", "Before", &var.beta, 10000);
    createTrackbar("gamma", "Before", &var.gamma, 100000);

    createTrackbar("ele x", "morph open", &var.ssSizeX, 20);
    createTrackbar("ele y", "morph open", &var.ssSizeY, 20);


    vc >> frame;
    if(frame.rows <= 0) {
        vc.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
    } else {
        var.imRows = frame.rows;
        var.imCols = frame.cols;
    }
    createTrackbar("x", "Crop", &var.x, var.imCols);
    createTrackbar("y", "Crop", &var.y, var.imRows);
    createTrackbar("width", "Crop", &var.width, var.imCols);
    createTrackbar("height", "Crop", &var.height, var.imRows);


    while( waitKey(20) != 1048586) { // enter
        vc >> frame;
        if(frame.rows <= 0) {
            vc.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
            continue;
        } else {
            var.imRows = frame.rows;
            var.imCols = frame.cols;
        }

        result = detectLeak(frame);


    }
    return -1;


}
