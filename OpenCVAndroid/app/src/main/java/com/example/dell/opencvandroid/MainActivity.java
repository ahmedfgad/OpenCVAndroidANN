package com.example.dell.opencvandroid;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.ml.ANN_MLP;

import static org.opencv.core.CvType.CV_32F;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        OpenCVLoader.initDebug();
    }

    public void predictANN(View v){
        double[][] XORTrainArray = {
                {0.0, 0.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 1.0}
        };
        Mat XORTrain = new Mat(4, 2, CV_32F);
        XORTrain.put(0, 0, XORTrainArray[0]);
        XORTrain.put(1, 0, XORTrainArray[1]);
        XORTrain.put(2, 0, XORTrainArray[2]);
        XORTrain.put(3, 0, XORTrainArray[3]);
        System.out.println("Train Inputs : \n" + XORTrain.dump());

        double[][] XORTrainOutArray = {
                {0.0},
                {1.0},
                {1.0},
                {0.0}
        };
        Mat XORTrainOut = new Mat(4, 1, CV_32F);
        XORTrainOut.put(0, 0, XORTrainOutArray[0]);
        XORTrainOut.put(1, 0, XORTrainOutArray[1]);
        XORTrainOut.put(2, 0, XORTrainOutArray[2]);
        XORTrainOut.put(3, 0, XORTrainOutArray[3]);
        System.out.println("Train Labels : \n" + XORTrainOut.dump());

        EditText modelPath = findViewById(R.id.modelPath);
        ANN_MLP ANN = ANN_MLP.load(modelPath.getText().toString());

        double num_correct_predictions = 0;
        for (int i = 0; i < XORTrain.rows(); i++) {
            Mat sample = XORTrain.row(i);
            double correct_label = XORTrainOut.get(i, 0)[0];

            Mat results = new Mat();
            ANN.predict(sample, results, 0);

            double response = results.get(0, 0)[0];
            double predicted_label = 0.0;

            if (response >= 0.5) {
                predicted_label = 1.0;
            } else {
                predicted_label = 0.0;
            }

            System.out.println("Input Sample : " + sample.dump() + ", Predicted Score : " + response + ", Predicted Label : " + predicted_label + ", Correct Label : " + correct_label);

            if (predicted_label == correct_label) {
                num_correct_predictions += 1;
            }
        }

        double accuracy = (num_correct_predictions / XORTrain.rows()) * 100;
        Toast.makeText(getApplicationContext(), "Accuracy : " + accuracy, Toast.LENGTH_LONG).show();

    }
}
