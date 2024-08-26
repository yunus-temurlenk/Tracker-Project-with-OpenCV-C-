#include <opencv2/opencv.hpp>
#include <opencv2/optflow/rlofflow.hpp>


cv::Mat drawTrackingPath(std::vector<cv::Point> pts_p, cv::Mat frame_p)
{
    cv::Point prevPoint;
    for(int i=0; i<(int) pts_p.size(); i++)
    {
        if(i!=0)
        {
            cv::line(frame_p,prevPoint, pts_p[i], cv::Scalar(0,255,255), 2);
        }
        prevPoint = pts_p[i];
    }

    return frame_p;
}

int main()
{
    cv::VideoCapture cap("/home/cvlab/Videos/Screencasts/temp/billard.mp4");

    cv::Mat frame, prevFrame, flowOut;

    cv::namedWindow("Frame",0);
    cv::namedWindow("ROI selector",0);
    cv::namedWindow("Result",0);

    cv::Ptr<cv::FarnebackOpticalFlow> optFlow = cv::FarnebackOpticalFlow::create();
    bool roiSelected = false;

    cv::Rect targetRect;

    std::vector<cv::Point> trackingPoints;

    cv::KalmanFilter KF(4, 2, 0);

    KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
                           0, 1, 0, 1,
                           0, 0, 1, 0,
                           0, 0, 0, 1);

    KF.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0,
                            0, 1, 0, 0);

    setIdentity(KF.processNoiseCov, cv::Scalar(1e-4));

    setIdentity(KF.measurementNoiseCov, cv::Scalar(1e-1));

    setIdentity(KF.errorCovPost, cv::Scalar(1));

    KF.statePost = (cv::Mat_<float>(4, 1) << 0,0,0,0);

    cv::Point2f predictedPosition, futurePredictPt;

    while (cap.read(frame))
    {
        cv::Mat result = frame.clone();
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::medianBlur(frame, frame, 3);
        if(roiSelected)
        {
            cv::Mat keepPrevFrame = frame.clone();
            if(prevFrame.data)
            {
                int xCoordSum = 0;
                int yCoordSum = 0;
                int vectorCnt = 0;
                optFlow->calc(prevFrame, frame, flowOut);

                for(int y=targetRect.y; y<targetRect.y + targetRect.height; y++)
                {
                    for(int x=targetRect.x; x<targetRect.x + targetRect.width; x++)
                    {
                        const cv::Point2f distanceVector = flowOut.at<cv::Point2f>(y, x);
                        double lengthOfVector = cv::norm(cv::Point(x,y) - cv::Point(x + distanceVector.x, y + distanceVector.y));
                        if(lengthOfVector > 1)
                        {
                            vectorCnt++;
                            xCoordSum += cvRound(x + distanceVector.x);
                            yCoordSum += cvRound(y + distanceVector.y);

                            cv::line(frame, cv::Point(x, y), cv::Point(x + distanceVector.x, y + distanceVector.y), cv::Scalar(0,0,0),1);
                        }
                    }
                }

                if(vectorCnt != 0)
                {
                    targetRect.x = xCoordSum / vectorCnt - targetRect.width/2;
                    targetRect.y = yCoordSum / vectorCnt - targetRect.height/2;

                    cv::Mat prediction = KF.predict();
                    predictedPosition = cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
                    cv::Mat measurement = (cv::Mat_<float>(2, 1) << xCoordSum / vectorCnt, yCoordSum / vectorCnt);
                    KF.correct(measurement);
                    cv::Mat future_state = KF.statePost.clone();
                    for (int i = 0; i < 10; ++i) { // predicting 10 steps ahead
                        future_state = KF.transitionMatrix * future_state;
                    }

                    futurePredictPt = cv::Point2f(future_state.at<float>(0), future_state.at<float>(1));

                    trackingPoints.push_back(cv::Point(xCoordSum / vectorCnt, yCoordSum / vectorCnt));
                }
            }

            prevFrame = keepPrevFrame.clone();
            cv::rectangle(frame, targetRect, cv::Scalar(255),2);
            cv::circle(result, predictedPosition, 5, cv::Scalar(0, 255, 0), -1); // Predicted position in green
            cv::circle(result, futurePredictPt, 5, cv::Scalar(0, 0, 255), -1); // Predicted position in green
            cv::imshow("Frame", frame);
            cv::imshow("Result", drawTrackingPath(trackingPoints, result));

            cv::waitKey(0);
        }
        else
        {
            targetRect = cv::selectROI(frame);
            roiSelected = true;
            cv::destroyWindow("ROI selector");
        }


    }

    return 0;
}
