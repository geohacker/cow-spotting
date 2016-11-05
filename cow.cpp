#include </Users/geohacker/projects/opencv_contrib/modules/dpm/include/opencv2/dpm.hpp>
#include </Users/geohacker/projects/opencv/include/opencv2/opencv.hpp>
#include "/usr/local/include/boost/filesystem.hpp"
#include "/usr/local/include/boost/filesystem/operations.hpp"
#include "/usr/local/include/boost/filesystem/path.hpp"
#include <iostream>
#include <string>

namespace fs = boost::filesystem;
using namespace cv;
using namespace cv::dpm;
using namespace std;

int main( int argc, char** argv )
{
    vector<string> models;
    vector<string> names;
    names.push_back("cow");
    models.push_back("cascade.xml");

    cv::Ptr<DPMDetector> detector = DPMDetector::create(models,names);


    double t = (double) getTickCount();

    fs::path full_path( fs::initial_path<fs::path>() );
    full_path = fs::system_complete( fs::path( argv[1] ) );

    // read the directory
    fs::directory_iterator end_iter;
    for ( fs::directory_iterator dir_itr( full_path );
          dir_itr != end_iter;
          ++dir_itr )
    {
        std::cout << dir_itr->path().filename() << "\n";

        // use the image and run through the model
        Mat frame = imread(dir_itr->path().string());
        resize(frame,frame,Size(),2,2);
        namedWindow("DPM Cascade Detection", 1);   
        vector<DPMDetector::ObjectDetection> ds;
        detector->detect(frame, ds);

        t = ((double) getTickCount() - t)/getTickFrequency();
        cerr << "dpm took " << t << " seconds" << endl;

        // look at the score and only write the image if confident.
        for (unsigned int i = 0; i < ds.size(); i++)
        {
            int id = ds[i].classID;
            cerr << names[id] << "\t" << ds[i].score << "\t" << ds[i].rect << endl;
            if (ds[i].score > -0.90) {
                rectangle(frame, ds[i].rect, Scalar(255, 255, 255), 1);
                imwrite(dir_itr->path().filename().string(), frame);
            }
        }
    }

    return 0;
}
