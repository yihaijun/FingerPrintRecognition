package com.company;

import org.opencv.core.*;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;

import java.util.LinkedList;
import java.util.List;

public class Matching {

    // variable

    //// method
    public void FetureExtract(Mat matInput)
    {

    }

    public int Match(Mat image1,Mat image2)
    {
        List<DMatch> matchesList;
        List<DMatch> goodMatchesList = new LinkedList<DMatch>();
        MatOfPoint2f goodPoints1 = new MatOfPoint2f();
        MatOfPoint2f goodPoints2 = new MatOfPoint2f();
        List<Point> goodPointsList1 = new LinkedList<Point>();
        List<Point> goodPointsList2 = new LinkedList<Point>();

        Mat descriptor1 = new Mat();
        Mat descriptor2 = new Mat();

        FeatureDetector detector = FeatureDetector.create(FeatureDetector.SIFT);
        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.SIFT);
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);

        MatOfKeyPoint keyPoints1 = new MatOfKeyPoint();
        MatOfKeyPoint keyPoints2 = new MatOfKeyPoint();
        MatOfDMatch matches = new MatOfDMatch();
        MatOfDMatch goodMatches = new MatOfDMatch();

        // detect features
        detector.detect(image1, keyPoints1);
        detector.detect(image2, keyPoints2);

        // extract features
        extractor.compute(image1, keyPoints1, descriptor1);
        extractor.compute(image2, keyPoints2, descriptor2);

        // match features
        matcher.match(descriptor1, descriptor2, matches);
        matchesList = matches.toList();


        double MatchingThreshold = 45;
        // find good matches
        double minDist = MatchingThreshold;
        double min = 1000000;
        double max = 0;
        double distance;
        for (int i = 0; i < descriptor1.rows(); i++) {
            distance = matchesList.get(i).distance;
            if (distance > max) max = distance;
            if (distance < min) min = distance;
            if (distance < minDist) {
                goodMatchesList.add(matchesList.get(i));
            }
        }
        goodMatches.fromList(goodMatchesList);

        // keyPoints of good matches
        List<KeyPoint> keyPointsList1 = keyPoints1.toList();
        List<KeyPoint> keyPointsList2 = keyPoints2.toList();
        DMatch m;
        for (int i = 0; i < goodMatchesList.size(); i++) {

            m = goodMatchesList.get(i);
            goodPointsList1.add(keyPointsList1.get(m.queryIdx).pt);
            goodPointsList2.add(keyPointsList2.get(m.trainIdx).pt);
        }
        goodPoints1.fromList(goodPointsList1);
        goodPoints2.fromList(goodPointsList2);

        // get homography
        // Mat homography = Calib3d.findHomography( goodPoints1, goodPoints2, Calib3d.RANSAC, 1.0);

        // draw result

        Mat result = new Mat();
        Scalar green = new Scalar(0, 255, 0);
        Scalar yellow = new Scalar(255, 255, 0);
        Scalar blue = new Scalar(0, 0, 255);
        Scalar red = new Scalar(255, 0, 0);
        MatOfByte mask = new MatOfByte();
        int flag = Features2d.NOT_DRAW_SINGLE_POINTS;
        Features2d.drawMatches(image1, keyPoints1, image2, keyPoints2, goodMatches, result, red, blue, mask, flag);

        int score = goodMatchesList.size();
        return score;
    }
}
