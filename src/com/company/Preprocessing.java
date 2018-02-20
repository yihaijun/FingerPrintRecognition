package com.company;

//////////
// img 에 대하여 전처리를 진행하는 클래스 총 6개의 step 으로 진행한다.
// 나중에 deep learning 을 이용한 전처리도 들어 갈 수 있다.
// step0 : Resize image
// step1 : Gray scale
// step2 : Masking
// step3 : Histogram equalization
// step4 : Ridge-orientation filter
// step5 : Threshold
// step6 : Thin
/////

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class Preprocessing {

    //// property

    // public
    public static Mat matSrcImage;
    public static Mat matResult;

    // private
    private Mat matResize;
    private Mat matGrayScale;
    private Mat matMasking;
    private Mat matRidgeOrientation;
    private Mat matRidgeFilter;
    private Mat matEnhanced;



    //// method


    // GetMatrix
    // parameter   : Mat
    // explanation : input 값을 받아와서 SrcImage 에 넣는다.
    public void GetMatrix(Mat matInput)
    {
        matSrcImage = matInput;
    }

    /**
     *
     * @param void
     * @return Mat
     */
    public Mat SetMatrix()
    {
        return matSrcImage;
    }

    public void Resize()
    {
        matResize = new Mat();
        Size size = new Size(300,400);
        Imgproc.resize(matSrcImage, matResize, size, 0, 0, Imgproc.INTER_CUBIC);
        GetMatrix(matResize);
    }

    public void GrayScaling()
    {
        int rows = matSrcImage.rows();
        int cols = matSrcImage.cols();
        matGrayScale = new Mat(rows, cols, CvType.CV_8UC1);
        Imgproc.cvtColor(matSrcImage, matGrayScale, Imgproc.COLOR_RGB2GRAY);
        GetMatrix(matGrayScale);
    }

    public void Masking()
    {
        matMasking = matGrayScale;
        int rows = matSrcImage.rows();
        int cols = matSrcImage.cols();
        int width = 300;
        int height = 400;
        // crop using ellipse and masking
        Mat roi = new Mat(rows, cols, CvType.CV_8UC1);

        Point center = new Point(width / 2, height / 2);
        Size axes = new Size(100, 75);
        Scalar scalarWhite = new Scalar(255, 255, 255);
        Scalar scalarGray = new Scalar(100, 100, 100);
        Scalar scalarBlack = new Scalar(0, 0, 0);
        int thickness = -1;
        int lineType = 8;

        // method 2: fill with gray instead of while
        roi.setTo(scalarWhite);
         //ellipse -- Core 에서 Imgproc 로 바꿨더니 됨~
        Imgproc.ellipse(roi, center, axes, 0, 0, 360, scalarBlack, thickness, lineType, 0);
        matMasking.setTo(scalarGray, roi);
        GetMatrix(matMasking);
//      roi.release();
    }

    public void HistogramEqualize()
    {
        //
    }

    public void RidgeOrientationFilter()
    {
        //
    }

    public void Thresholding()
    {

    }

    public void Thining()
    {

    }
}
