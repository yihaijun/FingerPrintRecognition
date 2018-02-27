package com.company;

import javafx.collections.ObservableList;
import javafx.scene.control.TextField;
import javafx.scene.layout.HBox;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
//import org.opencv.imgcodecs.Imgcodecs;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.stage.Stage;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Group;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.VBox;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import java.io.ByteArrayInputStream;

public class Main extends Application {

    @Override

    public void start(Stage primaryStage) throws Exception
    {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Preprocessing pp = new Preprocessing();
        Preprocessing pp1 = new Preprocessing();

        Stage stage = new Stage();
        Group root = new Group();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat in_image = new Mat();
        Mat in_image1 = new Mat();
        // 이미지 불로오는 부분

        Mat matchImg1 = new Mat();
        Mat matchImg2 = new Mat();

        //opencv 2.x
        in_image = Highgui.imread("C:\\Users\\sungmin\\Desktop\\projectjava\\firstProject-test\\AF1.jpg");
        in_image1 = Highgui.imread("C:\\Users\\sungmin\\Desktop\\projectjava\\firstProject-test\\AF6.jpg");

        //opencv 3.x
        //in_image = Imgcodecs.imread("C:\\Users\\sungmin\\Desktop\\projectjava\\firstProject-test\\AF1.jpg");
        //in_image1 = Imgcodecs.imread("C:\\Users\\sungmin\\Desktop\\projectjava\\firstProject-test\\AF2.jpg");
        //in_image = Imgcodecs.imread("C:\\Users\\sungmin\\Desktop\\projectjava\\firstProject-test\\test02.png");

        //preprocessing 하는부분
        pp.GetMatrix(in_image);
        pp.Process();

        pp1.GetMatrix(in_image1);
        pp1.Process();

        matchImg1 = pp.SetMatrix();
        matchImg2 = pp1.SetMatrix();

        Matching m = new Matching();
        int score = m.Match(matchImg1,matchImg2);

        //pp.GetMatrix(score);
        System.out.println("점수는요 : " + score);

        // 문제는 바로 이 부분 Mat 으로 된 in_image의 경우 Javafx나 swt 등에서 바로 ImageView를 할 수 없기에
        // 다음과 같이 바꿔서 하시면 됩니다. 아래 2줄은 stackOverflow에서 참고하였음.

        MatOfByte byteMat = new MatOfByte();
        MatOfByte byteMat1 = new MatOfByte();

        //opencv 2.x
        Highgui.imencode(".png", pp.SetMatrix(), byteMat);
        //opencv 3.x
        //Imgcodecs.imencode(".png", pp.SetMatrix(), byteMat);
        // SetMatrix()로 넣어주었다.

        Highgui.imencode(".png", pp1.SetMatrix(), byteMat1);

        Image img = new Image(new ByteArrayInputStream(byteMat.toArray()));
        ImageView imageView = new ImageView();

        imageView.setImage(img);

        Image img1 = new Image(new ByteArrayInputStream(byteMat1.toArray()));
        ImageView imageView1 = new ImageView();

        imageView1.setImage(img1);

        // -_- 귀찮아서 VBox 하나 만들어 이미지 올렸음돠..
        HBox vbox = new HBox();
        vbox.setAlignment(Pos.CENTER);

        ObservableList list = vbox.getChildren(); //HBox의 ObservableList 얻기
        list.add(imageView);     //TextField 컨트롤 배치
        list.add(imageView1);      //Button의 컨트롤 배치

        // 그래도 GUI 인데 EXIT 버튼 하나 달아주는 센스.
        Button btnExit = new Button("Exit");
        btnExit.setPrefHeight(30);
        btnExit.setMaxWidth(Double.MAX_VALUE);
        btnExit.setPadding(new Insets(10));
        btnExit.setOnAction(e -> System.exit(0));
        vbox.getChildren().add(btnExit);

        root.getChildren().add(vbox);

        Double width = img.getWidth() * 2 + 50;
        Double height = img.getHeight() + btnExit.getPrefHeight() + 5;
        height /= 1;

        // width 하고 Height에 값이 잘들어 있나 할려고 넣었음 없어도 되는 부분.
        System.out.println(width + " : " + height);
        Scene scene = new Scene(root, width, height);
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
