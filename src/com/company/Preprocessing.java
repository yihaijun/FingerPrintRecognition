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

import java.util.ArrayList;
import java.util.Collections;

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


    private void normalizeImage(Mat src, Mat dst) {

        MatOfDouble mean = new MatOfDouble(0.0);
        MatOfDouble std = new MatOfDouble(0.0);

        // get mean and standard deviation
        Core.meanStdDev(src, mean, std);
        Core.subtract(src, Scalar.all(mean.toArray()[0]), dst);
        Core.meanStdDev(dst, mean, std);
        Core.divide(dst, Scalar.all(std.toArray()[0]), dst);
    }

    public void RidgeOrientationFilter()
    {
        // step 1: get ridge segment by padding then do block process
        int blockSize = 24;
        double threshold = 0.05;
        Mat padded = imagePadding(matSrcImage, blockSize);                           ////1////
        int imgRows = padded.rows();
        int imgCols = padded.cols();

        Mat matRidgeSegment = new Mat(imgRows, imgCols, CvType.CV_32FC1);
        Mat segmentMask = new Mat(imgRows, imgCols, CvType.CV_8UC1);
        ridgeSegment(padded, matRidgeSegment, segmentMask, blockSize, threshold);    ////2////



        // step 2: get ridge orientation
        int gradientSigma = 1;
        int blockSigma = 13;
        int orientSmoothSigma = 15;
        matRidgeOrientation = new Mat(imgRows, imgCols, CvType.CV_32FC1);
        ridgeOrientation(matRidgeSegment, matRidgeOrientation, gradientSigma, blockSigma, orientSmoothSigma);
        ////3////



        // step 3: get ridge frequency
        int fBlockSize = 36;
        int fWindowSize = 5;
        int fMinWaveLength = 5;
        int fMaxWaveLength = 25;
        Mat matFrequency = new Mat(imgRows, imgCols, CvType.CV_32FC1);
        double medianFreq = ridgeFrequency(matRidgeSegment, segmentMask, matRidgeOrientation, matFrequency, fBlockSize, fWindowSize, fMinWaveLength, fMaxWaveLength);

        ////4////

        // step 4: get ridge filter
        matRidgeFilter = new Mat(imgRows, imgCols, CvType.CV_32FC1);
        double filterSize = 1.9;
        ridgeFilter(matRidgeSegment, matRidgeOrientation, matFrequency, matRidgeFilter, filterSize, filterSize, medianFreq);

        ////5////

        //GetMatrix(matRidgeFilter);
    }

    ////1////
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


        //Imgproc.copyMakeBorder(source, source, 0, bottomPadding, 0, rightPadding, Imgproc.BORDER_CONSTANT, Scalar.all(0));
        return source;
    }

    ////2////
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
                //
                // Core.rectangle(windowMask, new Point(roi.x, roi.y), new Point(roi.x + roi.width, roi.y + roi.height), scalarWhile, -1, 8, 0);

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

    ////3////

    private void ridgeOrientation(Mat ridgeSegment, Mat result, int gradientSigma, int blockSigma, int orientSmoothSigma) {

        int rows = ridgeSegment.rows();
        int cols = ridgeSegment.cols();

        // calculate image gradients
        int kSize = Math.round(6 * gradientSigma);
        if (kSize % 2 == 0) {
            kSize++;
        }
        Mat kernel = gaussianKernel(kSize, gradientSigma);
        ////3-1////

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
        ////3-2////
        Core.multiply(result, Scalar.all(Math.PI / 360.0), result);
    }

    ////3 - 1////
    private Mat gaussianKernel(int kSize, int sigma) {

        Mat kernelX = Imgproc.getGaussianKernel(kSize, sigma, CvType.CV_32FC1);
        Mat kernelY = Imgproc.getGaussianKernel(kSize, sigma, CvType.CV_32FC1);

        Mat kernel = new Mat(kSize, kSize, CvType.CV_32FC1);
        Core.gemm(kernelX, kernelY.t(), 1, Mat.zeros(kSize, kSize, CvType.CV_32FC1), 0, kernel, 0);
        return kernel;
    }

    ////3-2//// atan
    private void atan2(Mat src1, Mat src2, Mat dst) {

        int height = src1.height();
        int width = src2.width();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                dst.put(y, x, Core.fastAtan2((float) src1.get(y, x)[0], (float) src2.get(y, x)[0]));
            }
        }
    }

    ////4////
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
                ////4-1////
                frequency.copyTo(frequencies.rowRange(y, y + blockSize).colRange(x, x + blockSize));
            }
        }

        // mask out frequencies calculated for non ridge regions
        Core.multiply(frequencies, segmentMask, frequencies, 1.0, CvType.CV_32FC1);

        // find median frequency over all the valid regions of the image.
        double medianFrequency = medianFrequency(frequencies);
        ////4-2////

        // the median frequency value used across the whole fingerprint gives a more satisfactory result
        Core.multiply(segmentMask, Scalar.all(medianFrequency), frequencies, 1.0, CvType.CV_32FC1);

        return medianFrequency;
    }

    ////4-1////
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
        ////4-1-1////

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
    ////4-1-1////

    private double calculateMean(double[] m) {
        double sum = 0;
        for (int i = 0; i < m.length; i++) {
            sum += m[i];
        }
        return sum / m.length;
    }



    ////4-2////
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


    ////5////
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
        ////5-0////
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
        ////5-1////

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

    ////5-0////
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


    ////5-1////
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


    public void Thresholding()
    {

    }

    public void Thining()
    {

    }
}
