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
import org.opencv.features2d.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

public class Preprocessing {

    //// property

    // public
    public static Mat matSrcImage;
    public static Mat MatSnapShot;
    public static Mat MatSnapShotMask;
    public static Mat matResult;

    // private
    private Mat matResize;
    private Mat matGrayScale;
    private Mat matMasking;
    private Mat matRidgeOrientation;
    private Mat matRidgeFilter;
    private Mat matEnhanced;

    private int maskWidth = 260;
    private int maskHeight = 160;

    //// method


    // GetMatrix
    // parameter   : Mat
    // explanation : input 값을 받아와서 SrcImage 에 넣는다.
    public void GetMatrix(Mat matInput)
    {
        matSrcImage = matInput;
    }


    public Mat SetMatrix()
    {
        return matSrcImage;
    }

    public void Process()
    {
        Resize();
        GrayScaling();

        int rows = matSrcImage.rows();
        int cols = matSrcImage.cols();

        // apply histogram equalization
        Mat equalized = new Mat(rows, cols, CvType.CV_32FC1);
        Imgproc.equalizeHist(matSrcImage, equalized);

        // convert to float, very important
        Mat floated = new Mat(rows, cols, CvType.CV_32FC1);
        equalized.convertTo(floated, CvType.CV_32FC1);



        // normalise image to have zero mean and 1 standard deviation
        Mat normalized = new Mat(rows, cols, CvType.CV_32FC1);
        normalizeImage(floated, normalized);



        // step 1: get ridge segment by padding then do block process
        int blockSize = 24;
        double threshold = 0.05;
        Mat padded = imagePadding(floated, blockSize);
        int imgRows = padded.rows();
        int imgCols = padded.cols();
        Mat matRidgeSegment = new Mat(imgRows, imgCols, CvType.CV_32FC1);
        Mat segmentMask = new Mat(imgRows, imgCols, CvType.CV_8UC1);
        ridgeSegment(padded, matRidgeSegment, segmentMask, blockSize, threshold);




        // step 2: get ridge orientation
        int gradientSigma = 1;
        int blockSigma = 13;
        int orientSmoothSigma = 15;
        matRidgeOrientation = new Mat(imgRows, imgCols, CvType.CV_32FC1);
        ridgeOrientation(matRidgeSegment, matRidgeOrientation, gradientSigma, blockSigma, orientSmoothSigma);

        // step 3: get ridge frequency
        int fBlockSize = 36;
        int fWindowSize = 5;
        int fMinWaveLength = 5;
        int fMaxWaveLength = 25;
        Mat matFrequency = new Mat(imgRows, imgCols, CvType.CV_32FC1);
        double medianFreq = ridgeFrequency(matRidgeSegment, segmentMask, matRidgeOrientation, matFrequency, fBlockSize, fWindowSize, fMinWaveLength, fMaxWaveLength);

        // step 4: get ridge filter
        matRidgeFilter = new Mat(imgRows, imgCols, CvType.CV_32FC1);
        double filterSize = 1.9;
        ridgeFilter(matRidgeSegment, matRidgeOrientation, matFrequency, matRidgeFilter, filterSize, filterSize, medianFreq);



        // step 5: enhance image after ridge filter
        matEnhanced = new Mat(imgRows, imgCols, CvType.CV_8UC1);
        enhancement(matRidgeFilter, matEnhanced, blockSize);

        // set process result
        matResult = matEnhanced.clone();

        GetMatrix(matResult);
        /*
        Resize();
        GrayScaling();
        Masking();
        HistogramEqualize();
        RidgeOrientationFilter();
         */

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

        // opencv 3.x
        //Imgproc.ellipse(roi, center, axes, 0, 0, 360, scalarBlack, thickness, lineType, 0);
        // opencv 2.x
        Core.ellipse(roi, center, axes, 0, 0, 360, scalarBlack, thickness, lineType, 0);
        matMasking.setTo(scalarGray, roi);
        GetMatrix(matMasking);
//      roi.release();
    }

    public void HistogramEqualize()
    {
        int rows = matSrcImage.rows();
        int cols = matSrcImage.cols();

        // apply histogram equalization
        Mat equalized = new Mat(rows, cols, CvType.CV_32FC1);
        Imgproc.equalizeHist(matSrcImage, equalized);

        // convert to float, very important
        Mat floated = new Mat(rows, cols, CvType.CV_32FC1);
        equalized.convertTo(floated, CvType.CV_32FC1);

        GetMatrix(floated);

        // normalise image to have zero mean and 1 standard deviation
        Mat normalized = new Mat(rows, cols, CvType.CV_32FC1);
        normalizeImage(floated, normalized);
    }

    public void Thresholding()
    {

    }

    public void Thining()
    {

    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    private void ridgeSegment(Mat source, Mat result, Mat mask, int blockSize, double threshold) {

        // for each block, get standard deviation
        // and replace the block with it
        int widthSteps = source.width() / blockSize;
        int heightSteps = source.height() / blockSize;

        MatOfDouble mean = new MatOfDouble(0);
        MatOfDouble std = new MatOfDouble(0);
        Mat window;
        Scalar scalarBlack = Scalar.all(0);
        Scalar scalarWhile = Scalar.all(255);

        Mat windowMask = new Mat(source.rows(), source.cols(), CvType.CV_8UC1);

        Rect roi;
        double stdVal;

        for (int y = 1; y <= heightSteps; y++) {
            for (int x = 1; x <= widthSteps; x++) {

                roi = new Rect((blockSize) * (x - 1), (blockSize) * (y - 1), blockSize, blockSize);
                windowMask.setTo(scalarBlack);

                //opencv 2.x
                Core.rectangle(windowMask, new Point(roi.x, roi.y), new Point(roi.x + roi.width, roi.y + roi.height), scalarWhile, -1, 8, 0);

                //opencv 3.x
                //Imgproc.rectangle(windowMask, new Point(roi.x, roi.y), new Point(roi.x + roi.width, roi.y + roi.height), scalarWhile, -1, 8, 0);

                window = source.submat(roi);
                Core.meanStdDev(window, mean, std);
                stdVal = std.toArray()[0];
                result.setTo(Scalar.all(stdVal), windowMask);

                // mask used to calc mean and standard deviation later
                mask.setTo(Scalar.all(stdVal >= threshold ? 1 : 0), windowMask);
            }
        }

        // get mean and standard deviation
        Core.meanStdDev(source, mean, std, mask);
        Core.subtract(source, Scalar.all(mean.toArray()[0]), result);
        Core.meanStdDev(result, mean, std, mask);
        Core.divide(result, Scalar.all(std.toArray()[0]), result);
    }

    /**
     * Calculate ridge orientation.
     *
     * @param ridgeSegment
     * @param result
     * @param gradientSigma
     * @param blockSigma
     * @param orientSmoothSigma
     */
    private void ridgeOrientation(Mat ridgeSegment, Mat result, int gradientSigma, int blockSigma, int orientSmoothSigma) {

        int rows = ridgeSegment.rows();
        int cols = ridgeSegment.cols();

        // calculate image gradients
        int kSize = Math.round(6 * gradientSigma);
        if (kSize % 2 == 0) {
            kSize++;
        }
        Mat kernel = gaussianKernel(kSize, gradientSigma);

        Mat fXKernel = new Mat(1, 3, CvType.CV_32FC1);
        Mat fYKernel = new Mat(3, 1, CvType.CV_32FC1);
        fXKernel.put(0, 0, -1);
        fXKernel.put(0, 1, 0);
        fXKernel.put(0, 2, 1);
        fYKernel.put(0, 0, -1);
        fYKernel.put(1, 0, 0);
        fYKernel.put(2, 0, 1);

        Mat fX = new Mat(kSize, kSize, CvType.CV_32FC1);
        Mat fY = new Mat(kSize, kSize, CvType.CV_32FC1);
        Imgproc.filter2D(kernel, fX, CvType.CV_32FC1, fXKernel);
        Imgproc.filter2D(kernel, fY, CvType.CV_32FC1, fYKernel);

        Mat gX = new Mat(rows, cols, CvType.CV_32FC1);
        Mat gY = new Mat(rows, cols, CvType.CV_32FC1);
        Imgproc.filter2D(ridgeSegment, gX, CvType.CV_32FC1, fX);
        Imgproc.filter2D(ridgeSegment, gY, CvType.CV_32FC1, fY);

        // covariance data for the image gradients
        Mat gXX = new Mat(rows, cols, CvType.CV_32FC1);
        Mat gXY = new Mat(rows, cols, CvType.CV_32FC1);
        Mat gYY = new Mat(rows, cols, CvType.CV_32FC1);
        Core.multiply(gX, gX, gXX);
        Core.multiply(gX, gY, gXY);
        Core.multiply(gY, gY, gYY);

        // smooth the covariance data to perform a weighted summation of the data.
        kSize = Math.round(6 * blockSigma);
        if (kSize % 2 == 0) {
            kSize++;
        }
        kernel = gaussianKernel(kSize, blockSigma);
        Imgproc.filter2D(gXX, gXX, CvType.CV_32FC1, kernel);
        Imgproc.filter2D(gYY, gYY, CvType.CV_32FC1, kernel);
        Imgproc.filter2D(gXY, gXY, CvType.CV_32FC1, kernel);
        Core.multiply(gXY, Scalar.all(2), gXY);

        // analytic solution of principal direction
        Mat denom = new Mat(rows, cols, CvType.CV_32FC1);
        Mat gXXMiusgYY = new Mat(rows, cols, CvType.CV_32FC1);
        Mat gXXMiusgYYSquared = new Mat(rows, cols, CvType.CV_32FC1);
        Mat gXYSquared = new Mat(rows, cols, CvType.CV_32FC1);
        Core.subtract(gXX, gYY, gXXMiusgYY);
        Core.multiply(gXXMiusgYY, gXXMiusgYY, gXXMiusgYYSquared);
        Core.multiply(gXY, gXY, gXYSquared);
        Core.add(gXXMiusgYYSquared, gXYSquared, denom);
        Core.sqrt(denom, denom);

        // sine and cosine of doubled angles
        Mat sin2Theta = new Mat(rows, cols, CvType.CV_32FC1);
        Mat cos2Theta = new Mat(rows, cols, CvType.CV_32FC1);
        Core.divide(gXY, denom, sin2Theta);
        Core.divide(gXXMiusgYY, denom, cos2Theta);

        // smooth orientations (sine and cosine)
        // smoothed sine and cosine of doubled angles
        kSize = Math.round(6 * orientSmoothSigma);
        if (kSize % 2 == 0) {
            kSize++;
        }
        kernel = gaussianKernel(kSize, orientSmoothSigma);
        Imgproc.filter2D(sin2Theta, sin2Theta, CvType.CV_32FC1, kernel);
        Imgproc.filter2D(cos2Theta, cos2Theta, CvType.CV_32FC1, kernel);

        // calculate the result as the following, so the values of the matrix range [0, PI]
        //orientim = atan2(sin2theta,cos2theta)/360;
        atan2(sin2Theta, cos2Theta, result);
        Core.multiply(result, Scalar.all(Math.PI / 360.0), result);
    }

    /**
     * Calculate ridge frequency.
     *
     * @param ridgeSegment
     * @param segmentMask
     * @param ridgeOrientation
     * @param frequencies
     * @param blockSize
     * @param windowSize
     * @param minWaveLength
     * @param maxWaveLength
     * @return
     */
    private double ridgeFrequency(Mat ridgeSegment, Mat segmentMask, Mat ridgeOrientation, Mat frequencies, int blockSize, int windowSize, int minWaveLength, int maxWaveLength) {

        int rows = ridgeSegment.rows();
        int cols = ridgeSegment.cols();

        Mat blockSegment;
        Mat blockOrientation;
        Mat frequency;

        for (int y = 0; y < rows - blockSize; y += blockSize) {
            for (int x = 0; x < cols - blockSize; x += blockSize) {
                blockSegment = ridgeSegment.submat(y, y + blockSize, x, x + blockSize);
                blockOrientation = ridgeOrientation.submat(y, y + blockSize, x, x + blockSize);
                frequency = calculateFrequency(blockSegment, blockOrientation, windowSize, minWaveLength, maxWaveLength);
                frequency.copyTo(frequencies.rowRange(y, y + blockSize).colRange(x, x + blockSize));
            }
        }

        // mask out frequencies calculated for non ridge regions
        Core.multiply(frequencies, segmentMask, frequencies, 1.0, CvType.CV_32FC1);

        // find median frequency over all the valid regions of the image.
        double medianFrequency = medianFrequency(frequencies);

        // the median frequency value used across the whole fingerprint gives a more satisfactory result
        Core.multiply(segmentMask, Scalar.all(medianFrequency), frequencies, 1.0, CvType.CV_32FC1);

        return medianFrequency;
    }

    /**
     * Estimate fingerprint ridge frequency within image block.
     *
     * @param block
     * @param blockOrientation
     * @param windowSize
     * @param minWaveLength
     * @param maxWaveLength
     * @return
     */
    private Mat calculateFrequency(Mat block, Mat blockOrientation, int windowSize, int minWaveLength, int maxWaveLength) {

        int rows = block.rows();
        int cols = block.cols();

        Mat orientation = blockOrientation.clone();
        Core.multiply(orientation, Scalar.all(2.0), orientation);

        int orientLength = (int) (orientation.total());
        float[] orientations = new float[orientLength];
        orientation.get(0, 0, orientations);

        double[] sinOrient = new double[orientLength];
        double[] cosOrient = new double[orientLength];
        for (int i = 1; i < orientLength; i++) {
            sinOrient[i] = Math.sin((double) orientations[i]);
            cosOrient[i] = Math.cos((double) orientations[i]);
        }
        float orient = Core.fastAtan2((float) calculateMean(sinOrient), (float) calculateMean(cosOrient)) / (float) 2.0;

        // rotate the image block so that the ridges are vertical
        Mat rotated = new Mat(rows, cols, CvType.CV_32FC1);
        Point center = new Point(cols / 2, rows / 2);
        double rotateAngle = ((orient / Math.PI) * (180.0)) + 90.0;
        double rotateScale = 1.0;
        Size rotatedSize = new Size(cols, rows);
        Mat rotateMatrix = Imgproc.getRotationMatrix2D(center, rotateAngle, rotateScale);
        Imgproc.warpAffine(block, rotated, rotateMatrix, rotatedSize, Imgproc.INTER_NEAREST);

        // crop the image so that the rotated image does not contain any invalid regions
        // this prevents the projection down the columns from being mucked up
        int cropSize = (int) Math.round(rows / Math.sqrt(2));
        int offset = (int) Math.round((rows - cropSize) / 2.0) - 1;
        Mat cropped = rotated.submat(offset, offset + cropSize, offset, offset + cropSize);

        // get sums of columns
        float sum = 0;
        Mat proj = new Mat(1, cropped.cols(), CvType.CV_32FC1);
        for (int c = 1; c < cropped.cols(); c++) {
            sum = 0;
            for (int r = 1; r < cropped.cols(); r++) {
                sum += cropped.get(r, c)[0];
            }
            proj.put(0, c, sum);
        }

        // find peaks in projected grey values by performing a grayScale
        // dilation and then finding where the dilation equals the original values.
        Mat dilateKernel = new Mat(windowSize, windowSize, CvType.CV_32FC1, Scalar.all(1.0));
        Mat dilate = new Mat(1, cropped.cols(), CvType.CV_32FC1);
        Imgproc.dilate(proj, dilate, dilateKernel, new Point(-1, -1), 1);
        //Imgproc.dilate(proj, dilate, dilateKernel, new Point(-1, -1), 1, Imgproc.BORDER_CONSTANT, Scalar.all(0.0));

        double projMean = Core.mean(proj).val[0];
        double projValue;
        double dilateValue;
        final double ROUND_POINTS = 1000;
        ArrayList<Integer> maxind = new ArrayList<Integer>();
        for (int i = 0; i < cropped.cols(); i++) {

            projValue = proj.get(0, i)[0];
            dilateValue = dilate.get(0, i)[0];

            // round to maximize the likelihood of equality
            projValue = (double) Math.round(projValue * ROUND_POINTS) / ROUND_POINTS;
            dilateValue = (double) Math.round(dilateValue * ROUND_POINTS) / ROUND_POINTS;

            if (dilateValue == projValue && projValue > projMean) {
                maxind.add(i);
            }
        }

        // determine the spatial frequency of the ridges by dividing the distance between
        // the 1st and last peaks by the (No of peaks-1). If no peaks are detected
        // or the wavelength is outside the allowed bounds, the frequency image is set to 0
        Mat result = new Mat(rows, cols, CvType.CV_32FC1, Scalar.all(0.0));
        int peaks = maxind.size();
        if (peaks >= 2) {
            double waveLength = (maxind.get(peaks - 1) - maxind.get(0)) / (peaks - 1);
            if (waveLength >= minWaveLength && waveLength <= maxWaveLength) {
                result = new Mat(rows, cols, CvType.CV_32FC1, Scalar.all((1.0 / waveLength)));
            }
        }

        return result;
    }

    /**
     * Enhance fingerprint image using oriented filters.
     *
     * @param ridgeSegment
     * @param orientation
     * @param frequency
     * @param result
     * @param kx
     * @param ky
     * @param medianFreq
     * @return
     */
    private void ridgeFilter(Mat ridgeSegment, Mat orientation, Mat frequency, Mat result, double kx, double ky, double medianFreq) {

        int angleInc = 3;
        int rows = ridgeSegment.rows();
        int cols = ridgeSegment.cols();

        int filterCount = 180 / angleInc;
        Mat[] filters = new Mat[filterCount];

        double sigmaX = kx / medianFreq;
        double sigmaY = ky / medianFreq;

        //mat refFilter = exp(-(x. ^ 2 / sigmaX ^ 2 + y. ^ 2 / sigmaY ^ 2) / 2). * cos(2 * pi * medianFreq * x);
        int size = (int) Math.round(3 * Math.max(sigmaX, sigmaY));
        size = (size % 2 == 0) ? size : size + 1;
        int length = (size * 2) + 1;
        Mat x = meshGrid(size);
        Mat y = x.t();

        Mat xSquared = new Mat(length, length, CvType.CV_32FC1);
        Mat ySquared = new Mat(length, length, CvType.CV_32FC1);
        Core.multiply(x, x, xSquared);
        Core.multiply(y, y, ySquared);
        Core.divide(xSquared, Scalar.all(sigmaX * sigmaX), xSquared);
        Core.divide(ySquared, Scalar.all(sigmaY * sigmaY), ySquared);

        Mat refFilterPart1 = new Mat(length, length, CvType.CV_32FC1);
        Core.add(xSquared, ySquared, refFilterPart1);
        Core.divide(refFilterPart1, Scalar.all(-2.0), refFilterPart1);
        Core.exp(refFilterPart1, refFilterPart1);

        Mat refFilterPart2 = new Mat(length, length, CvType.CV_32FC1);
        Core.multiply(x, Scalar.all(2 * Math.PI * medianFreq), refFilterPart2);
        refFilterPart2 = matCos(refFilterPart2);

        Mat refFilter = new Mat(length, length, CvType.CV_32FC1);
        Core.multiply(refFilterPart1, refFilterPart2, refFilter);

        // Generate rotated versions of the filter.  Note orientation
        // image provides orientation *along* the ridges, hence +90
        // degrees, and the function requires angles +ve anticlockwise, hence the minus sign.
        Mat rotated;
        Mat rotateMatrix;
        double rotateAngle;
        Point center = new Point(length / 2, length / 2);
        Size rotatedSize = new Size(length, length);
        double rotateScale = 1.0;
        for (int i = 0; i < filterCount; i++) {
            rotateAngle = -(i * angleInc);
            rotated = new Mat(length, length, CvType.CV_32FC1);
            rotateMatrix = Imgproc.getRotationMatrix2D(center, rotateAngle, rotateScale);
            Imgproc.warpAffine(refFilter, rotated, rotateMatrix, rotatedSize, Imgproc.INTER_LINEAR);
            filters[i] = rotated;
        }

        // convert orientation matrix values from radians to an index value
        // that corresponds to round(degrees/angleInc)
        Mat orientIndexes = new Mat(orientation.rows(), orientation.cols(), CvType.CV_8UC1);
        Core.multiply(orientation, Scalar.all((double) filterCount / Math.PI), orientIndexes, 1.0, CvType.CV_8UC1);

        Mat orientMask;
        Mat orientThreshold;

        orientMask = new Mat(orientation.rows(), orientation.cols(), CvType.CV_8UC1, Scalar.all(0));
        orientThreshold = new Mat(orientation.rows(), orientation.cols(), CvType.CV_8UC1, Scalar.all(0.0));
        Core.compare(orientIndexes, orientThreshold, orientMask, Core.CMP_LT);
        Core.add(orientIndexes, Scalar.all(filterCount), orientIndexes, orientMask);

        orientMask = new Mat(orientation.rows(), orientation.cols(), CvType.CV_8UC1, Scalar.all(0));
        orientThreshold = new Mat(orientation.rows(), orientation.cols(), CvType.CV_8UC1, Scalar.all(filterCount));
        Core.compare(orientIndexes, orientThreshold, orientMask, Core.CMP_GE);
        Core.subtract(orientIndexes, Scalar.all(filterCount), orientIndexes, orientMask);

        // finally, find where there is valid frequency data then do the filtering
        Mat value = new Mat(length, length, CvType.CV_32FC1);
        Mat subSegment;
        int orientIndex;
        double sum;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (frequency.get(r, c)[0] > 0
                        && r > (size + 1)
                        && r < (rows - size - 1)
                        && c > (size + 1)
                        && c < (cols - size - 1)) {
                    orientIndex = (int) orientIndexes.get(r, c)[0];
                    subSegment = ridgeSegment.submat(r - size - 1, r + size, c - size - 1, c + size);
                    Core.multiply(subSegment, filters[orientIndex], value);
                    sum = Core.sumElems(value).val[0];
                    result.put(r, c, sum);
                }
            }
        }
    }

    /**
     * Enhance the image after ridge filter.
     * Apply mask, binary threshold, thinning, ..., etc.
     *
     * @param source
     * @param result
     * @param blockSize
     */
    private void enhancement(Mat source, Mat result, int blockSize) {

        MatSnapShotMask = snapShotMask(matSrcImage.rows(),matSrcImage.cols(),10);

        Mat paddedMask = imagePadding(MatSnapShotMask, blockSize);

        // apply the original mask to get rid of extras
        Core.multiply(source, paddedMask, result, 1.0, CvType.CV_8UC1);

        // apply binary threshold
        Imgproc.threshold(result, result, 0, 255, Imgproc.THRESH_BINARY);

        // apply thinning
        //int thinIterations = 2;
        //thin(result, thinIterations);

        //// normalize the values to the binary scale [0, 255]
        //Core.normalize(result, result, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);

        // apply morphing (erosion, opening, ... )
        //int erosionSize = 1;
        //int erosionLength = 2 * erosionSize + 1;
        //Mat erosionKernel = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(erosionLength, erosionLength), new Point(erosionSize, erosionSize));
        //Imgproc.erode(result, result, erosionKernel);
    }

    /**
     * Thinning the given matrix.
     *
     * @param source
     * @param iterations
     * @return
     */
    private void thin(Mat source, int iterations) {

        int rows = source.rows();
        int cols = source.cols();

        Mat thin = new Mat(rows, cols, CvType.CV_8UC1, Scalar.all(0.0));

        for (int i = 0; i < iterations; i++) {
            thinSubIteration(source, thin);
            thin.copyTo(source);
        }
    }

    /**
     * Iteration for thinning.
     *
     * @param pSrc
     * @param pDst
     */
    private void thinSubIteration(Mat pSrc, Mat pDst) {
        int rows = pSrc.rows();
        int cols = pSrc.cols();
        pSrc.copyTo(pDst);
        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                if (pSrc.get(i, j)[0] == 1.0) {
                    /// get 8 neighbors
                    /// calculate C(p)
                    int n0 = (int) pSrc.get(i - 1, j - 1)[0];
                    int n1 = (int) pSrc.get(i - 1, j)[0];
                    int n2 = (int) pSrc.get(i - 1, j + 1)[0];
                    int n3 = (int) pSrc.get(i, j + 1)[0];
                    int n4 = (int) pSrc.get(i + 1, j + 1)[0];
                    int n5 = (int) pSrc.get(i + 1, j)[0];
                    int n6 = (int) pSrc.get(i + 1, j - 1)[0];
                    int n7 = (int) pSrc.get(i, j - 1)[0];
                    int C = (~n1 & (n2 | n3)) + (~n3 & (n4 | n5)) + (~n5 & (n6 | n7)) + (~n7 & (n0 | n1));
                    if (C > 0) {
                        /// calculate N
                        int N1 = (n0 | n1) + (n2 | n3) + (n4 | n5) + (n6 | n7);
                        int N2 = (n1 | n2) + (n3 | n4) + (n5 | n6) + (n7 | n0);
                        int N = Math.min(N1, N2);
                        if ((N == 2) || (N == 3)) {
                            /// calculate criteria 3
                            int c3 = (n1 | n2 | ~n4) & n3;
                            if (c3 == 0) {
                                pDst.put(i, j, 0.0);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Function for thinning the given binary image
     *
     * @param im
     */
    private void thinningGuoHall(Mat im) {

        int rows = im.rows();
        int cols = im.cols();

        Core.divide(im, Scalar.all(255.0), im);
        Mat prev = new Mat(rows, cols, CvType.CV_8UC1, Scalar.all(0.0));
        Mat diff = new Mat(rows, cols, CvType.CV_8UC1);

        do {
            thinningGuoHallIteration(im, 0);
            thinningGuoHallIteration(im, 1);
            Core.absdiff(im, prev, diff);
            im.copyTo(prev);
        }
        while (Core.countNonZero(diff) > 0);

        Core.multiply(im, Scalar.all(255.0), im);
    }

    /**
     * Perform one thinning iteration.
     * Normally you wouldn't call this function directly from your code.
     *
     * @param im         Binary image with range = 0-1
     * @param iterations 0=even, 1=odd
     */
    private void thinningGuoHallIteration(Mat im, int iterations) {

        int rows = im.rows();
        int cols = im.cols();

        Mat marker = new Mat(rows, cols, CvType.CV_8UC1, Scalar.all(0.0));

        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                byte p2 = (byte) im.get(i - 1, j)[0];
                byte p3 = (byte) im.get(i - 1, j + 1)[0];
                byte p4 = (byte) im.get(i, j + 1)[0];
                byte p5 = (byte) im.get(i + 1, j + 1)[0];
                byte p6 = (byte) im.get(i + 1, j)[0];
                byte p7 = (byte) im.get(i + 1, j - 1)[0];
                byte p8 = (byte) im.get(i, j - 1)[0];
                byte p9 = (byte) im.get(i - 1, j - 1)[0];

                int C = (~p2 & (p3 | p4)) + (~p4 & (p5 | p6)) + (~p6 & (p7 | p8)) + (~p8 & (p9 | p2));
                int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
                int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
                int N = N1 < N2 ? N1 : N2;
                int m = iterations == 0 ? ((p6 | p7 | ~p9) & p8) : ((p2 | p3 | ~p5) & p4);

                if (C == 1 && (N >= 2 && N <= 3) & m == 0) {
                    marker.put(i, j, 1);
                }
            }
        }

        Core.bitwise_not(marker, marker);
        Core.add(im, marker, im);
    }

    /**
     * Function for thinning the given binary image
     *
     * @param im Binary image with range = 0-255
     */
    void thinning(Mat im) {

        int rows = im.rows();
        int cols = im.cols();

        Core.divide(im, Scalar.all(255.0), im);

        Mat prev = new Mat(rows, cols, CvType.CV_8UC1, Scalar.all(0.0));
        Mat diff = new Mat(rows, cols, CvType.CV_8UC1);

        do {
            thinningIteration(im, 0);
            thinningIteration(im, 1);
            Core.absdiff(im, prev, diff);
            im.copyTo(prev);
        }
        while (Core.countNonZero(diff) > 0);

        Core.multiply(im, Scalar.all(255.0), im);
    }

    /**
     * Perform one thinning iteration.
     * Normally you wouldn't call this function directly from your code.
     *
     * @param im         Binary image with range = 0-1
     * @param iterations 0=even, 1=odd
     */
    private void thinningIteration(Mat im, int iterations) {
        int rows = im.rows();
        int cols = im.cols();

        Mat marker = new Mat(rows, cols, CvType.CV_8UC1, Scalar.all(0.0));

        for (int i = 1; i < rows - 1; i++) {
            for (int j = 1; j < cols - 1; j++) {
                byte p2 = (byte) im.get(i - 1, j)[0];
                byte p3 = (byte) im.get(i - 1, j + 1)[0];
                byte p4 = (byte) im.get(i, j + 1)[0];
                byte p5 = (byte) im.get(i + 1, j + 1)[0];
                byte p6 = (byte) im.get(i + 1, j)[0];
                byte p7 = (byte) im.get(i + 1, j - 1)[0];
                byte p8 = (byte) im.get(i, j - 1)[0];
                byte p9 = (byte) im.get(i - 1, j - 1)[0];

                boolean a = (p2 == 0 && p3 == 1) || (p3 == 0 && p4 == 1) || (p4 == 0 && p5 == 1) || (p5 == 0 && p6 == 1) || (p6 == 0 && p7 == 1) || (p7 == 0 && p8 == 1) || (p8 == 0 && p9 == 1) || (p9 == 0 && p2 == 1);
                int A = a ? 1 : 0;
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int m1 = iterations == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
                int m2 = iterations == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

                if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
                    marker.put(i, j, 1);
                }
            }
        }

        Core.bitwise_not(marker, marker);
        Core.add(im, marker, im);
    }

    /**
     * Create mesh grid.
     *
     * @param size
     * @return
     */
    private Mat meshGrid(int size) {

        int l = (size * 2) + 1;
        int value = -size;

        Mat result = new Mat(l, l, CvType.CV_32FC1);
        for (int c = 0; c < l; c++) {
            for (int r = 0; r < l; r++) {
                result.put(r, c, value);
            }
            value++;
        }
        return result;
    }

    /**
     * Round the values of the given mat to
     *
     * @param source
     * @param points
     * @return
     */
    private Mat roundMat(Mat source, double points) {

        int cols = source.cols();
        int rows = source.rows();

        Mat doubleMat = new Mat(rows, cols, CvType.CV_32FC1);
        Mat intMat = new Mat(rows, cols, CvType.CV_8UC1);

        Core.multiply(source, Scalar.all(points), doubleMat);
        doubleMat.convertTo(intMat, CvType.CV_8UC1);
        intMat.convertTo(doubleMat, CvType.CV_32FC1);
        Core.divide(doubleMat, Scalar.all(points), doubleMat);

        return doubleMat;
    }

    /**
     * Get unique items in the given mat using the given mask.
     *
     * @param source
     * @param mask
     * @return
     */
    private float[] uniqueValues(Mat source, Mat mask) {

        Mat result = new Mat(source.cols(), source.rows(), CvType.CV_32FC1);
        Core.multiply(source, mask, result, 1.0, CvType.CV_32FC1);

        int length = (int) (result.total());
        float[] values = new float[length];
        result.get(0, 0, values);

        return values;
    }

    /**
     * Apply sin to each element of the matrix.
     *
     * @param source
     * @return
     */
    private Mat matSin(Mat source) {

        int cols = source.cols();
        int rows = source.rows();
        Mat result = new Mat(cols, rows, CvType.CV_32FC1);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.put(r, c, Math.sin(source.get(r, c)[0]));
            }
        }

        return result;
    }

    /**
     * Apply cos to each element of the matrix.
     *
     * @param source
     * @return
     */
    private Mat matCos(Mat source) {

        int rows = source.rows();
        int cols = source.cols();

        Mat result = new Mat(cols, rows, CvType.CV_32FC1);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.put(r, c, Math.cos(source.get(r, c)[0]));
            }
        }

        return result;
    }

    /**
     * Calculate the median of all values greater than zero.
     *
     * @param image
     * @return
     */
    private double medianFrequency(Mat image) {

        ArrayList<Double> values = new ArrayList<Double>();
        double value = 0;

        for (int r = 0; r < image.rows(); r++) {
            for (int c = 0; c < image.cols(); c++) {
                value = image.get(r, c)[0];
                if (value > 0) {
                    values.add(value);
                }
            }
        }

        Collections.sort(values);
        int size = values.size();
        double median = 0;

        if (size > 0) {
            int halfSize = size / 2;
            if ((size % 2) == 0) {
                median = (values.get(halfSize - 1) + values.get(halfSize)) / 2.0;
            } else {
                median = values.get(halfSize);
            }
        }
        return median;
    }

    /**
     * Apply padding to the image.
     *
     * @param source
     * @param blockSize
     * @return
     */
    private Mat imagePadding(Mat source, int blockSize) {

        int width = source.width();
        int height = source.height();

        int bottomPadding = 0;
        int rightPadding = 0;

        if (width % blockSize != 0) {
            bottomPadding = blockSize - (width % blockSize);
        }
        if (height % blockSize != 0) {
            rightPadding = blockSize - (height % blockSize);
        }
        //opencv 3.x version
        //Core.copyMakeBorder(source, source, 0, bottomPadding, 0, rightPadding, Core.BORDER_CONSTANT, Scalar.all(0));

        //opencv 2.x version
        Imgproc.copyMakeBorder(source, source, 0, bottomPadding, 0, rightPadding, Imgproc.BORDER_CONSTANT, Scalar.all(0));
        return source;
    }

    private void atan2(Mat src1, Mat src2, Mat dst) {

        int height = src1.height();
        int width = src2.width();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                dst.put(y, x, Core.fastAtan2((float) src1.get(y, x)[0], (float) src2.get(y, x)[0]));
            }
        }
    }

    /**
     * Normalize the image to have zero mean and unit standard deviation.
     */
    private void normalizeImage(Mat src, Mat dst) {

        MatOfDouble mean = new MatOfDouble(0.0);
        MatOfDouble std = new MatOfDouble(0.0);

        // get mean and standard deviation
        Core.meanStdDev(src, mean, std);
        Core.subtract(src, Scalar.all(mean.toArray()[0]), dst);
        Core.meanStdDev(dst, mean, std);
        Core.divide(dst, Scalar.all(std.toArray()[0]), dst);
    }

    /**
     * Create Gaussian kernel.
     *
     * @param sigma
     */
    private Mat gaussianKernel(int kSize, int sigma) {

        Mat kernelX = Imgproc.getGaussianKernel(kSize, sigma, CvType.CV_32FC1);
        Mat kernelY = Imgproc.getGaussianKernel(kSize, sigma, CvType.CV_32FC1);

        Mat kernel = new Mat(kSize, kSize, CvType.CV_32FC1);
        Core.gemm(kernelX, kernelY.t(), 1, Mat.zeros(kSize, kSize, CvType.CV_32FC1), 0, kernel, 0);
        return kernel;
    }

    /**
     * Create Gaussian kernel.
     *
     * @param sigma
     */
    private Mat gaussianKernel_(int kSize, int sigma) {

        Mat kernel = new Mat(kSize, kSize, CvType.CV_32FC1);

        double total = 0;
        int l = kSize / 2;
        double distance = 0;
        double value = 0;

        for (int y = -l; y <= l; y++) {
            for (int x = -l; x <= l; x++) {
                distance = ((x * x) + (y * y)) / (2 * (sigma * sigma));
                value = Math.exp(-distance);
                kernel.put(y + l, x + l, value);
                total += value;
            }
        }

        for (int y = 0; y < kSize; y++) {
            for (int x = 0; x < kSize; x++) {
                value = kernel.get(y, x)[0];
                value /= total;
                kernel.put(y, x, value);
            }
        }

        return kernel;
    }


    /**
     * Calculate mean of given array.
     *
     * @param m
     * @return
     */
    private double calculateMean(double[] m) {
        double sum = 0;
        for (int i = 0; i < m.length; i++) {
            sum += m[i];
        }
        return sum / m.length;
    }

    /**
     * Calculate mean of given array.
     *
     * @param m
     * @return
     */
    private double calculateMean(ArrayList<Double> data) {
        double sum = 0;
        for (int i = 0; i < data.size(); i++) {
            sum += data.get(i);
        }
        return sum / data.size();
    }

    /**
     * Calculate variance of a list.
     *
     * @param data
     * @param mean
     * @return
     */
    double getVariance(ArrayList<Double> data, double mean) {
        double temp = 0;
        for (double a : data) {
            temp += (mean - a) * (mean - a);
        }
        return Math.sqrt(temp / data.size());
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    private Mat snapShotMask(int rows, int cols, int offset) {

        Point center = new Point(cols / 2, rows / 2);
        Size axes = new Size(maskWidth - offset, maskHeight - offset);
        Scalar scalarWhite = new Scalar(255, 255, 255);
        Scalar scalarGray = new Scalar(100, 100, 100);
        Scalar scalarBlack = new Scalar(0, 0, 0);
        int thickness = -1;
        int lineType = 8;

        Mat mask = new Mat(rows, cols, CvType.CV_8UC1, scalarBlack);

        //opencv 2.x
        Core.ellipse(mask, center, axes, 0, 0, 360, scalarWhite, thickness, lineType, 0);

        //opencv 3.x
        //Imgproc.ellipse(mask, center, axes, 0, 0, 360, scalarWhite, thickness, lineType, 0);
        return mask;
    }


    //////////////////////////////////////////////matching/////////////////////////////////////////////////////
    public static int matching(Mat image1, Mat image2) {

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

        // find good matches
        double MatchingThreshold = 45;
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

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

}
