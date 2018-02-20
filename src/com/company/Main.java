package com.company;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;

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
import org.opencv.imgcodecs.Imgcodecs;
import java.io.ByteArrayInputStream;

public class Main extends Application {

    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }


    @Override

    public void start(Stage primaryStage) throws Exception{

        Preprocessing pp = new Preprocessing();

        Stage stage = new Stage();
        Group root = new Group();
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat in_image = new Mat();
        // 이미지 불로오는 부분

        in_image = Imgcodecs.imread("C:\\Users\\sungmin\\Desktop\\projectjava\\firstProject-test\\AF1.jpg");


        //preprocessing 하는부분
        pp.GetMatrix(in_image);
        pp.Resize();
        pp.GrayScaling();
        pp.Masking();


        // 문제는 바로 이 부분 Mat 으로 된 in_image의 경우 Javafx나 swt 등에서 바로 ImageView를 할 수 없기에
        // 다음과 같이 바꿔서 하시면 됩니다. 아래 2줄은 stackOverflow에서 참고하였음.

        MatOfByte byteMat = new MatOfByte();
        Imgcodecs.imencode(".png", pp.SetMatrix(), byteMat);
        // SetMatrix()로 넣어주었다.

        Image img = new Image(new ByteArrayInputStream(byteMat.toArray()));
        ImageView imageView = new ImageView();

        imageView.setImage(img);

        // -_- 귀찮아서 VBox 하나 만들어 이미지 올렸음돠..
        VBox vbox = new VBox();
        vbox.setAlignment(Pos.CENTER);
        vbox.getChildren().add(imageView);
        // 그래도 GUI 인데 EXIT 버튼 하나 달아주는 센스.
        Button btnExit = new Button("Exit");
        btnExit.setPrefHeight(30);
        btnExit.setMaxWidth(Double.MAX_VALUE);
        btnExit.setPadding(new Insets(10));
        btnExit.setOnAction(e -> System.exit(0));
        vbox.getChildren().add(btnExit);

        root.getChildren().add(vbox);

        Double width = img.getWidth();
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
