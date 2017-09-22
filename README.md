# EmotionDetector

Python project that uses OpenCV, Sci-kit Learn, NumPy, TensorFlow and Kivy to implement machine learning and computer vision to detect emotions in faces and categorize them into seven different categories [Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral].  
This project consists of three modules: 
1. The module that runs a GUI through which a picture can be snapped or uploaded to detect emotions.
2. The module that handles facial recognition, image extraction and manipulation.
3. The module that handles training of the CNN as well as emotion classification.

Dataset: 35,000 48x48 grayscale images downloaded from Kaggle.  
CNN consists of two convolutional layers, two activation layyers, two max pool layers and three fully connected layers and makes use of normalization techniques to increase confidence score.   
Lanczos interpolation and fast non local means denoising used to smoothen the cropped face and remove Gaussian white noise.
