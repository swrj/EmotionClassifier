import tensorflow as tf
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if cv2:
    img =  cv2.imread("happy.jpg")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    sub_face = img[y:y+h, x:x+w]
gray_face = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
def grayface():
    return gray_face
resized_image = cv2.resize(gray_face, (48,48))
print(resized_image.shape)
feature_set = np.multiply(resized_image.shape[0],resized_image.shape[1])
test_set = np.reshape(resized_image,[1,feature_set] )
hist = np.histogram(resized_image.flatten(), 256, [0,256])[0]
print (hist.shape)
width, height = resized_image.shape[:2]
print (width)
print (height)
cv2.imshow('gray_face', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
sess = tf.Session()
saver = tf.train.import_meta_graph('my_model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
y = graph.get_tensor_by_name("y_:0")
predict = graph.get_tensor_by_name("prediction:0")
max_pred = graph.get_tensor_by_name("max_prediction:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
classification = sess.run(predict,feed_dict = {x:test_set,keep_prob: 1.0})
max = sess.run(max_pred,feed_dict={x:test_set,keep_prob:1.0})
print(classification)
print(max)
