# TrackIt AI - AI-Powered Attendance System

## Overview

TrackIt AI is a smart attendance system that recognizes faces to mark attendance automatically. It’s built to be simple and user-friendly, helping you manage attendance without any hassle. You can add images either by capturing them with a camera or uploading them from your computer. The system recognizes registered users and marks their attendance, storing the data in an Excel sheet for easy access and download.

## Demo Video

Click the video below to see how to use TrackIt AI:

[![TrackIt AI Demo](https://img.youtube.com/vi/your-video-id/maxresdefault.jpg)](https://www.youtube.com/watch?v=your-video-id)


This is just the beginning, and I plan to make TrackIt AI even better with future enhancements, aiming to improve accuracy and add more features.

## How to Use TrackIt AI

### 1. Adding Image Data

#### Using Camera:

1. **Enter Name**: Type your name in the provided field.
2. **Start Camera**: Click the **Camera** button to turn on your camera.
3. **Capture Images**: Click the **Capture** button to take pictures. The app will capture multiple images automatically.
4. **Stop Camera**: Click the **Stop** button to turn off the camera. These images will be used to train the model.
5. **View Data**: Click the **Show** button to see your data and information.
6. **Delete Data**: Click the **Delete** button to remove all your image data.

#### Uploading Files:

1. **Prepare Images**: Name each image file with the person’s name followed by a dot [exp:- Mohit.].
2. **Upload Images**: Select and upload the images from your computer.
3. **Optimal Performance**: For Using TrackIt AI, add at least 5 images per user and ensure a balanced number of images for each user.
4. **Train Model**: Click the **Start Training** button to train the model. Training will take some time.

### 2. Recording Attendance

1. **Start Camera**: After training, click the **Start** button to turn on the camera.
2. **Face Detection**: The system will detect and recognize the faces of registered users and mark their attendance in an Excel sheet. It will detect but not mark unknown users.
3. **Stop Camera**: Click the **End** button to turn off the camera.
4. **View Attendance Data**: Click the **Show Data** button to see the attendance data in the Excel sheet.
5. **Download CSV**: Click the **Download CSV** button to download the attendance data as a CSV file.

### Note:

If the model takes time to detect your face, make sure:

- Your webcam is clean.
- Your pose is correct.
- You are close enough to the camera.
- The lighting is good.
- Your face is clearly visible.

By following these steps, you can effectively use the TrackIt AI Attendance System to manage attendance through facial recognition.

## Installation

You can use this application by downloading and running it on your localhost. Follow these steps:

1. **Download the Files**: Download `app.py` and `haarcascade_frontalface_default.xml` and save them in the same directory.
2. **Set Up Environment**: Set up your Python environment with the necessary libraries.

## Contributing

I welcome contributions from the community. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.


**Thank you for using TrackIt AI!**

   
