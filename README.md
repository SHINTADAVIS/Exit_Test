# Automated Emotion Detection from a video

## Emotion Detection from Video using DeepFace and OpenCV

This project demonstrates **automatic emotion recognition from video frames** using **DeepFace** and **OpenCV**.  
The script analyzes each frame of a video, detects faces, predicts the dominant emotion for each detected face, overlays the emotion label, and saves the annotated video.

## Project Overview

The goal of this project is to:  
1. **Extract frames** from a video file    
2. **Detect faces** using a Haar Cascade classifier    
3. **Analyze emotions** using the DeepFace library    
4. **Overlay predictions** (emotion labels) on detected faces    
5. **Save the output video** with all annotations  

This implementation is designed to run smoothly in **Google Colab** with Google Drive integration.  

### Tools and Libraries Used

| Library                             | Description |
|----------                           |-------------|
| **DeepFace**                        | Deep learning-based face analysis framework for emotion, age, gender, and race detection |
| **OpenCV (cv2)**                    | Used for video frame extraction, face detection, and annotation |
| **Google Colab**                    | Cloud environment for running the notebook |
| **Google Drive**                    | Used to store and access input videos |
| **Haar Cascade Classifier**         | Pre-trained model for frontal face detection |

### Workflow

#### Mount Google Drive
Mount Google Drive to access the input video:  

    from google.colab import drive
    drive.mount('/content/drive')
    
#### Install and Import Dependencies
    !pip install deepface
    from deepface import DeepFace
    import cv2
    from google.colab.patches import cv2_imshow
#### Define File Paths
Set your input video path: 

    video_path = "/content/drive/MyDrive/Colab Notebooks/Ml_IA/Video/video.mp4"
#### Load Face Detection Model
    face_model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#### Process Video Frames
Loop through video frames and detect emotions:  

    frame_list = []
    capture = cv2.VideoCapture(video_path)

    for i in range(5000):
        _, frame = capture.read()
        face = face_model.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)

        for x, y, width, height in face:
            emotion = DeepFace.analyze(frame, actions=["emotion"])
            cv2.putText(frame, str(emotion["dominant_emotion"]),(x, y + height), cv2.FONT_HERSHEY_COMPLEX,0.9, (255, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
            frame_list.append(frame)

        height, width, colors = frame.shape
        size = (width, height)
        
#### Save Output Video
    output_path = "Emotions.avi"
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"DIVX"), 10, size)
    for frame in frame_list:
        output.write(frame)

    output.release()
 #### Output

    The processed video (Emotions.avi) contains:

    Bounding boxes drawn around each detected face

    The predicted dominant emotion displayed above or below the face

Example emotions detected:
Happy
Sad
Angry
Surprised
Neutral

#### Input and Output Details
Input Video:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.mp4 file&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Original video file    
Output Video:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.avi	file&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Annotated video with detected emotions  

#### Requirements

Install the following dependencies: 

    !pip install deepface opencv-python
    
DeepFace automatically installs required backend frameworks (TensorFlow, Keras, etc.) if not already present.  

#### Notes and Recommendations

If the video has more than 5000 frames, adjust the loop limit:  

    for i in range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))):
    
To speed up processing, consider analyzing every Nth frame instead of all frames.    
GPU acceleration is recommended for faster DeepFace inference (available in Google Colab)  

