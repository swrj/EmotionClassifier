import tensorflow as tf
import cv2
import numpy as np
import sys
import getopt

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
emotion_to_be_displayed = ''
image_path = ''
def main(argv):
    try:
        opts, args = getopt.getopt(argv, "e:i:", ["emotion=", "image="])
        print (opts)
        print (args)
    except getopt.GetoptError:
        print ('usage: python test.py -e <emotion> -i <image_path>')
    for opt, arg in opts:
        if opt in ("-e", "--emotion"):
            if arg in emotions:
                emotion_to_be_displayed=arg
                index_of_emotion =  emotions.index(emotion_to_be_displayed)
        elif opt in ("-i", "--image"):
            image_path=arg
    print ("Emotion: " + emotion_to_be_displayed)
    print ("Image path: " + image_path)
    img = cv2.imread(image_path)
    if cv2:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        sub_face = img[y:y + h, x:x + w]
    gray_face = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_face, (48, 48))
    print(resized_image.shape)
    feature_set = np.multiply(resized_image.shape[0], resized_image.shape[1])
    test_set = np.reshape(resized_image, [1, feature_set])
    hist = np.histogram(resized_image.flatten(), 256, [0, 256])[0]
    print(hist.shape)
    width, height = resized_image.shape[:2]
    print(width)
    print(height)
    cv2.imshow('gray_face', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sess = tf.Session()
    saver = tf.train.import_meta_graph('my_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y_:0")
    predict = graph.get_tensor_by_name("prediction:0")
    max_pred = graph.get_tensor_by_name("max_prediction:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    classification = sess.run(predict, feed_dict={x: test_set, keep_prob: 1.0})
    max = sess.run(max_pred,feed_dict={x:test_set,keep_prob:1.0})
    index = classification[0]
    emotion_displayed = emotions[index]
    if emotion_displayed is emotion_to_be_displayed:
        print("---------------------------")
        print("This person is showing "+ emotion_displayed + ". The score is "+(max*100))
        print("---------------------------")
    else:
        print("---------------------------")
        print("This person is NOT showing " + emotion_displayed + ". The emotion displayed is " + (max*100) + " with a score of " + (max*100))
        print("---------------------------")

main(sys.argv[1:])
