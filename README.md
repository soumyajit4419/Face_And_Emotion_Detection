# Facial Recognition and Emotion Detection
___________

### Emotion Detection

Humans are used to non verbal communication. The emotions expressed increases the clarity of any thoughts and ideas. It becoms quite interesting when a computer can capture this complex feature of humans, ie emotions. This topic talks about building a model which can detect an emotion from an image. There key points to be followed are:

1. Data gathering and  augmentation

   The dataset taken was **"fer2013"**. It can be downloaded through the link "https://github.com/npinto/fer2013". Image augmentation was performed on this data.

2. Model building

   The model architecture consists of CNN Layer, Max Pooling, Flatten and Dropout Layers.

3. Training

   The model was trained  by  using variants of above layers mentioned in model building and by varying hyperparameters. The best model was able to achieve 60.1% of validation accuracy.

4. Testing

   The model was tested with sample images. It can be seen below:

   <img src="./Test_Images/happy.jpg" alt="index1" height="300px"/>
   <img src="./Test_Images/neutral.jpg" alt="index2" height="300px"/>
    <img src="./Test_Images/suprise.jpg" alt="index3" height="300px"/>

#### The model will be able to detect 7 types of emotions:-
 #####  Angry , Sad ,  Neutral ,  Disgust ,  Surprise ,  Fear  , and   Happy

## Usage:

### For  Face Detection, and Emotion Detection Code

Refer to the notebook /Emotion_Detection.ipynb.<br/>
I have trained an emotion detection model and put its trained weights at /Models

### Train your Emotion Detection Model
To train your own emotion detection model, Refer to the notebook /facial_emotion_recognition.ipynb

### For Emotion Detection  using Webcam 
` python Emotion_Detection.py`


