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

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class Preprocessing {

    //// property

    // public
    public static Mat matSrcImage;
    public static Mat matResult;

    // private
    private Mat matResize;
    private Mat matGrayScale;
    private Mat matMask;
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
        Size size = new Size(200,200);
        Imgproc.resize(matSrcImage, matResize, size, 0, 0, Imgproc.INTER_CUBIC);
        GetMatrix(matResize);
    }

    public void GrayScaling()
    {

    }

    public void Masking()
    {

    }

}
