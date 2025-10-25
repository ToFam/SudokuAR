#pragma once

#include <math.h>

#include <opencv2/opencv.hpp>

#define PI 3.14159265f
#define EPS 1e-8


std::vector<float> localMax(cv::Mat data);

void dualAnglePeak(cv::Mat histogram, int minDist, int& maxPeak, int& secondPeak);

template<typename TV, typename T>
bool sortGridYX(const TV& left, const TV& right, T margin)
{
    if (abs(left.y - right.y) > margin)
    {
        return left.y < right.y;
    }
    else
    {
        return left.x < right.x;
    }

}
template<typename TV, typename T>
bool sortGridXY(const TV& left, const TV& right, T margin)
{
    if (abs(left.x - right.x) > margin)
    {
        return left.x < right.x;
    }
    else
    {
        return left.y < right.y;
    }
}

template<typename T>
inline T avgMean(typename std::vector<T>::iterator const& s, typename std::vector<T>::iterator const& t, int avgWindow = 0)
{
    std::sort(s, t);

    int size = t - s + 1;
    int m = size / 2;
    int l = std::max(0, m - avgWindow);
    int r = std::min(size - 1, m + avgWindow);
    int croppedSize = r - l + 1;

    T avg = 0;
    for (int i = l; i <= r; ++i)
    {
        avg += *(s + i);
    }
    return avg / croppedSize;
}

inline double toRad(double degree)
{
    return PI * (degree / 180);
}

inline double toDegree(double radians)
{
    return (radians / PI) * 180;
}

inline double safeAcos (double x)
{
    if (x < -1.0) x = -1.0 ;
    else if (x > 1.0) x = 1.0 ;
    return acos (x) ;
}

inline float safeAcos (float x)
{
    if (x < -1.0f) x = -1.0f ;
    else if (x > 1.0f) x = 1.0f ;
    return acos (x) ;
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, d1) and (o2, d2).
// adapted from https://stackoverflow.com/a/7448287
inline bool intersect2D(cv::Point2f o1, cv::Point2f d1, cv::Point2f o2, cv::Point2f d2,
                   cv::Point2f &r, double &t1)
{
    cv::Point2f x = o2 - o1;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < EPS)
     return false;

    t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return true;
}

inline bool intersect2D(float r1, float t1, float r2, float t2, cv::Point2f& intersection)
{
    float a = cos(t1);
    float b = sin(t1);
    float c = cos(t2);
    float d = sin(t2);
    float det = a * d - b * c;
    if (abs(det) < EPS)
    {
        return false;
    }

    intersection.y = (a * r2 - c * r1) / det;
    intersection.x = (d * r1 - b * r2) / det;
    return true;
}


inline bool intersect2D(cv::Vec4i line1, cv::Vec4i line2, float tolerance, cv::Point2f& intersection)
{
    cv::Point2f start1 = cv::Point2f(line1[0], line1[1]);
    cv::Point2f end1 = cv::Point2f(line1[2], line1[3]);
    cv::Point2f d1 = (end1 - start1);
    d1 /= cv::norm(d1);

    cv::Point2f start2 = cv::Point2f(line2[0], line2[1]);
    cv::Point2f end2 = cv::Point2f(line2[2], line2[3]);
    cv::Point2f d2 = end2 - start2;
    d2 /= cv::norm(d2);
    double t;

    double t1Max;
    if (d1.x > EPS)
    {
        t1Max = (end1.x - start1.x) / d1.x;
    }
    else
    {
        t1Max = (end1.y - start1.y) / d1.y;
    }

    double t2Max;
    if (d2.x > EPS)
    {
        t2Max = (end2.x - start2.x) / d2.x;
    }
    else
    {
        t2Max = (end2.y - start2.y) / d2.y;
    }

    if (intersect2D(start1, d1, start2, d2, intersection, t))
    {
        double t2;
        if (d2.x > EPS)
        {
            t2 = (intersection.x - start2.x) / d2.x;
        }
        else
        {
            t2 = intersection.y - start2.y / d2.y;
        }


        if (t < -tolerance || t > t1Max + tolerance || t2 < tolerance || t2 > t2Max + tolerance)
        {
            return false;
        }

        return true;
    }

    return false;
}

/*
   Calculate the line segment PaPb that is the shortest route between
   two lines P1P2 and P3P4. Calculate also the values of mua and mub where
      Pa = P1 + mua (P2 - P1)
      Pb = P3 + mub (P4 - P3)
   Return FALSE if no solution exists.

   adapted from http://paulbourke.net/geometry/pointlineplane/
*/
inline bool intersect3D (cv::Point3f p1, cv::Point3f p2, cv::Point3f p3, cv::Point3f p4, cv::Point3f& outA, cv::Point3f& outB, double &muA, double &muB)
{
   cv::Point3f p13,p43,p21;
   double d1343,d4321,d1321,d4343,d2121;
   double numer,denom;

   p13.x = p1.x - p3.x;
   p13.y = p1.y - p3.y;
   p13.z = p1.z - p3.z;
   p43.x = p4.x - p3.x;
   p43.y = p4.y - p3.y;
   p43.z = p4.z - p3.z;
   if (abs(p43.x) < EPS && abs(p43.y) < EPS && abs(p43.z) < EPS)
      return false;
   p21.x = p2.x - p1.x;
   p21.y = p2.y - p1.y;
   p21.z = p2.z - p1.z;
   if (abs(p21.x) < EPS && abs(p21.y) < EPS && abs(p21.z) < EPS)
       return false;

   d1343 = p13.x * p43.x + p13.y * p43.y + p13.z * p43.z;
   d4321 = p43.x * p21.x + p43.y * p21.y + p43.z * p21.z;
   d1321 = p13.x * p21.x + p13.y * p21.y + p13.z * p21.z;
   d4343 = p43.x * p43.x + p43.y * p43.y + p43.z * p43.z;
   d2121 = p21.x * p21.x + p21.y * p21.y + p21.z * p21.z;

   denom = d2121 * d4343 - d4321 * d4321;
   if (abs(denom) < EPS)
       return false;
   numer = d1343 * d4321 - d1321 * d4343;

   muA = numer / denom;
   muB = (d1343 + d4321 * muA) / d4343;

   outA.x = p1.x + muA * p21.x;
   outA.y = p1.y + muA * p21.y;
   outA.z = p1.z + muA * p21.z;
   outB.x = p3.x + muB * p43.x;
   outB.y = p3.y + muB * p43.y;
   outB.z = p3.z + muB * p43.z;

   return true;
}

inline bool intersectRect(float rho, float theta, float width, float height, cv::Point2f& start, cv::Point2f& end)
{
    // LEFT, TOP, RIGHT, BOTTOM
    float boundariesR[] = {0.f, 0.f, width, height};
    float boundariesT[] = {0.f, CV_PI*1.f/2.f, 0.f, CV_PI*1.f/2.f};


    float leftY, topX, rightY, botX;

    bool ok = true;

    bool vertical = false, horizontal = false;
    cv::Point2f intersect;
    if (!intersect2D(rho, theta, boundariesR[0], boundariesT[0], intersect))
        vertical = true;

    leftY = intersect.y;
    if (!intersect2D(rho, theta, boundariesR[1], boundariesT[1], intersect))
        horizontal = true;

    topX = intersect.x;
    ok &= intersect2D(rho, theta, boundariesR[2], boundariesT[2], intersect);
    rightY = intersect.y;
    ok &= intersect2D(rho, theta, boundariesR[3], boundariesT[3], intersect);
    botX = intersect.x;

    if (vertical && horizontal)
    {
        return false;
    }

    if (vertical)
    {
        start.x = topX;
        start.y = 0.f;
        end.x = botX;
        end.y = height;
    }
    else if (horizontal)
    {
        start.x = 0.f;
        start.y = leftY;
        end.x = width;
        end.y = rightY;
    }
    else
    {
        if (leftY < 0.f)
        {
            // Top
            start.x = topX;
            start.y = 0.f;

            if (rightY > height)
            {
                // Bottom
                end.x = botX;
                end.y = height;
            }
            else
            {
                // Right
                end.x = width;
                end.y = rightY;
            }
        }
        else
        {
            if (leftY < height)
            {
                // Left
                start.y = leftY;
                start.x = 0.f;

                if (topX < 0.f)
                {
                    if (botX > width)
                    {
                        // right
                        end.x = width;
                        end.y = rightY;
                    }
                    else
                    {
                        // bottom
                        end.x = botX;
                        end.y = height;
                    }
                }
                else
                {
                    if (rightY < 0.f)
                    {
                        // Top
                        end.x = topX;
                        end.y = 0.f;
                    }
                    else
                    {
                        // Right
                        end.x = width;
                        end.y = rightY;
                    }
                }
            }
            else
            {
                // Bottom
                start.y = height;
                start.x = botX;

                if (topX < width)
                {
                    // Top
                    end.x = topX;
                    end.y = 0.f;
                }
                else
                {
                    // Right
                    end.x = width;
                    end.y = rightY;
                }
            }
        }
    }

    return true;
}

inline bool isParallel(float theta1, float theta2, float toleranceRadians)
{
    float t1 = fmod(theta1, CV_PI);
    float t2 = fmod(theta2, CV_PI);

    return abs(t1 - t2) < toleranceRadians || abs(t1 - CV_PI - t2) < toleranceRadians || abs(t2 - CV_PI - t1) < toleranceRadians;
}

inline bool isDirection(float theta1, float theta2, float toleranceRadians)
{
    float t1 = fmod(theta1, 2*CV_PI);
    float t2 = fmod(theta2, 2*CV_PI);

    return abs(t1 - t2) < toleranceRadians || abs(t1 - 2*CV_PI - t2) < toleranceRadians || abs(t2 - 2*CV_PI - t1) < toleranceRadians;
}

/** \brief convert x, y value to polar coordinates
*   rho is distance to origin, always positive
*   theta is angle on unit circle in range [0, 2*pi)
*/
inline void cartesianToPolar(double x, double y, double& rho, double& theta)
{
    rho = sqrt(x * x + y * y);

    if (abs(x) < EPS)
    {
        y < 0 ? theta = 1.5*PI : theta = 0.5*PI;
    }
    else if (abs(y) < EPS)
    {
        x < 0 ? theta = PI : theta = 0.0;
    }
    else
    {
        double A = atan(abs(y/x));
        if (x >= 0.0 && y >= 0.0)
            theta = A;
        else if (x < 0.0 && y >= 0.0)
            theta = PI - A;
        else if (x < 0.0 && y < 0.0)
            theta = PI + A;
        else
            theta = 2 * PI - A;
    }
}

/** \brief convert polar coordinates to cartesian
*   rho is distance to origin, always positive
*   theta is angle on unit circle in range [0, 2*pi)
*/
inline void polarToCartesian(double rho, double theta, double& x, double& y)
{
    if (theta < 0.5 * CV_PI)
    {
        x = cos(theta) * rho;
        y = sin(theta) * rho;
    }
    else if (theta < CV_PI)
    {
        x = -cos(CV_PI - theta) * rho;
        y = sin(CV_PI - theta) * rho;
    }
    else if (theta < 1.5 * CV_PI)
    {
        x = -cos(theta - CV_PI) * rho;
        y = -sin(theta - CV_PI) * rho;
    }
    else
    {
        x = cos(2 * CV_PI - theta) * rho;
        y = -sin(2 * CV_PI - theta) * rho;
    }
}

/** \brief cap the polar angle theta to [0, pi) and flip the sign of rho if needed */
inline void polarToBoundedPolar(double& rho, double& theta)
{
    if (theta > CV_PI)
    {
        theta -= CV_PI;
        rho = -rho;
    }
}

/** \brief expand bounded polar coordinates with theta in [0, pi) to full spectrum [0, 2*pi) and flip rho if needed */
inline void boundedPolarToPolar(double& rho, double& theta)
{
    if (rho < 0.0)
    {
        rho = -rho;
        theta += CV_PI;
    }
}

inline void arrangePoints(cv::Point2f* src)
{
    auto swap = [](cv::Point2f *arr, int idx1, int idx2){
        cv::Point2f d = arr[idx1];
        arr[idx1] = arr[idx2];
        arr[idx2] = d;
    };

    for (int j = 0; j < 4; j++)
    {
        for (int i = j - 1; i >= 0; i--)
        {
            if (src[i].y < src[i + 1].y)
            {
                swap(src, i, i + 1);
            }
            else
                break;
        }
    }

    if (src[0].x < src[1].x)
        swap(src, 0, 1);

    if (src[2].x > src[3].x)
        swap(src, 2, 3);
}
