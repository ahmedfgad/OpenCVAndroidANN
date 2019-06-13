# OpenCVAndroidANN
Running Artificial Neural Network (ANN) in Android using OpenCV

The project has 2 folders:
1. **OpenCVApp**: This is the NetBeans project in which the ANN is created, trained, and saved in an YML file.
2. **OpenCVAndroid**: This is the Android Studio project in which a simple Android app is created for loading the YML file of the trained ANN and then makes prediction.

Because the goal of this project is to discuss how to create an ANN and load it into Andorid, there is no need to complicate the process. This is why a simple ANN created for building an XOR gate.

The interface of the Android app main activity is shown below.
![Fig 24](https://user-images.githubusercontent.com/16560492/59463142-92d6de00-8e25-11e9-9d38-c668fb30cc4f.png)

There is an EditText view which accepts the path of the YML file. This means you have to send the trained ANN (YML file) to your Android device before using it.

After making sure the YML file is sent to your device, enter its path inside the EditText view and click the button above it. This loads the ANN and makes predictions.

The classification accuracy is displayed in a Toast message.
