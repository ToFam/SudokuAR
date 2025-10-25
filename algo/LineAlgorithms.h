#pragma once

#include <opencv2/core.hpp>


void getPolar(cv::Vec4i line, cv::Point2f origin, double& rho, double& theta);

/** \brief Calculate histogram of line angles
* \param lines input lines array. If 2 columns, polar coords are assumed, 
*                    otherwise use 4 columns with start and end points
* \param outHistogram output histogram of type CV_8U and dimensions 1x(360/angleResolution)
* \param angleResolution resolution of histogram in degree
*/
bool angleHistogram(cv::Mat lines, cv::Mat& outHistogram, float angleResolution = 1.0f);

/** \brief Non-Maximum supression for found hough lines that coincide in a specific theta and rho range */
void filterHoughLinesNonMax(std::vector<cv::Vec3f>& lines, double deltaRho = 0.1, double deltaTheta = 0.1);

/** \brief Filter hough lines that coincide in a specific theta and rho range,
 *  leaving only the one that lies nearest to \e targetTheta */
void filterHoughLinesAngle(std::vector<cv::Vec3f>& lines, double targetAngle, double deltaRho = 0.1, double deltaTheta = 0.1);

/** \brief Overwrite of filterHoughLinesAngle() */
void filterHoughLinesAngle(std::vector<cv::Vec2f>& lines, double targetAngle, double deltaRho = 0.1, double deltaTheta = 0.1);


/** \brief Hough line transform
* \param targetTheta median target angle for lines to be found, should be within [0, 2*PI), if negative the full range [0, PI) is searched
* \param angleTolerance tolerance margin around \e targetTheta in radians
*/
void houghLines(cv::Mat inputImage, std::vector<cv::Vec3f>& outLines, double rhoResolution, double thetaResolution, double threshold,
                double targetTheta = -1.0, double angleTolerance = 0.05);

/** \brief overwrite of Hough line transform that returns two-elem vector (just rho, theta no accumulator score) */
void houghLines(cv::Mat inputImage, std::vector<cv::Vec2f>& outLines, double rhoResolution, double thetaResolution, double threshold,
                double targetTheta = -1.0, double angleTolerance = 0.05);
