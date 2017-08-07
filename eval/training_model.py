import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# reading Data
data = pd.read_csv("C:\\Users\\Nikhil Kannan\\Documents\\fer2013.csv")
# splitting pixels at newline character
pixels_values = data.pixels.str.split(" ").tolist()
# storing the features(pixel_values) into a DataFrame
pixels_values = pd.DataFrame(pixels_values, dtype=int)
print(pixels_values)
# storing features as float
images = pixels_values.values.astype(np.float32)
print(images)

# Normalization of images(training data)
images = images - images.mean(axis=1).reshape(-1,1)
images = np.multiply(images,100.0/255.0)
each_pixel_mean = images.mean(axis=0)
each_pixel_std = np.std(images, axis=0)
images = np.divide(np.subtract(images,each_pixel_mean), each_pixel_std)

# Method to display images
def show_image(img):
    image = img.reshape([48,48])
    plt.imshow(image,cmap= "gray")

show_image(images[7])

# storing the number of features
image_pixels = images.shape[1]
# image width and image height
image_width = image_height = np.ceil(np.sqrt(image_pixels)).astype(np.uint8)
# labels for the data
labels_flat = data["emotion"].values.reshape([-1,1])
# number of instances
labels_count = labels_flat.shape[0]
# converting labels into One-Hot vectors
Encoder = OneHotEncoder()
labels_encoder = Encoder.fit_transform(labels_flat)
labels_one_hot = labels_encoder.toarray()
labels = labels_one_hot.astype(np.uint8)
# storing the number of labels
label_classes = labels.shape[1]

# splitting data into a training and testing set
X_train,X_test,Y_train,Y_test = train_test_split(images,labels,test_size=0.2,random_state=42)
print(X_test.shape)

# CNN model

# Xavier initializer for regularization purposes
Xint = tf.contrib.layers.xavier_initializer()
epsilon = 0.001
filter_size1 = 5
filter_size2 = 5
num_input_channels = 1
num_filters1 = 64
num_filters2 = 128

# method to create convolutional filters
def weight_variable(shape):
    Weights = tf.truncated_normal(shape, stddev=1e-4)
    tf.summary.histogram("Filter",Weights)
    return tf.Variable(Weights,Xint)
# method to create bias variable for the convolutional process
def bias_variable(shape):
    Bias = tf.constant(0.1, shape=shape)
    tf.summary.histogram("Filter_bias",Bias)
    return tf.Variable(Bias)
# convolutional process
def conv2d(x, W, padd):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padd)
# max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


# placeholder holds features
x = tf.placeholder('float', shape=[None, image_pixels])
# placeholder holds labels
y_ = tf.placeholder('float', shape=[None, label_classes])
# probability that the element is kept
keep_prob = tf.placeholder('float')

# creates convolutional filter
W_conv1 = weight_variable([filter_size1,filter_size1,num_input_channels, num_filters1])
b_conv1 = bias_variable([num_filters1])

# reshaping the x placeholder
image = tf.reshape(x, [-1,image_width , image_height,num_input_channels])
# Applying activation function after the convolution process
h_conv1 = tf.nn.relu(conv2d(image, W_conv1, "SAME") + b_conv1)
tf.summary.histogram("activation1",h_conv1)
# call the max pooling method
h_pool1 = max_pool_2x2(h_conv1)
# normalization on the pooling layer
h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
# second convolutional layer
W_conv2 = weight_variable([filter_size2,filter_size2,num_filters1,num_filters2])
b_conv2 = bias_variable([num_filters2])
# Applying activation function after the convolution process
h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2, "SAME") + b_conv2)
tf.summary.histogram("activation2",h_conv2)
# call the max pooling method
h_pool2 = max_pool_2x2(h_conv2)
# normalization on the pooling layer
h_norm2 = tf.nn.lrn(h_pool2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
# get the shape of the normalized pooling layer
h_norm2_shape = h_norm2.get_shape()
# extract the number of elements in the 4-d tensor
num_features = h_norm2_shape[1:4].num_elements()
print(num_features)
# number of neurons in the 1st Hidden_Layer
Hidden_Layer1 = 3072
# number of neurons in the 2nd Hidden_Layer
Hidden_Layer2 = 1536
# number of neurons in the 3rd Hidden_Layer
Hidden_Layer3  = 768


# creating fully_connected_Layers

# method to create weights and biases for every layer
def local_weight_variable(shape):
    weight = tf.truncated_normal(shape, stddev=0.04)
    tf.summary.histogram("Weight",weight)
    return tf.Variable(weight,Xint)

def local_bias_variable(shape):
    bias = tf.constant(0.0, shape=shape)
    tf.summary.histogram("Bias",bias)
    return tf.Variable(bias)

# generating weights and biases between the input and the 1st Hidden_Layer
W_fc1 = local_weight_variable([num_features, Hidden_Layer1])
b_fc1 = local_bias_variable([Hidden_Layer1])

# reshaping the 4-D input tensor into a 2-D tensor
h_norm2_flat = tf.reshape(h_norm2, [-1,num_features])

# Applying summation operation
fc_1 = tf.matmul(h_norm2_flat, W_fc1) + b_fc1

# Applying batch_normalization
# calculating mean, and variance for the 1st fully_connected layer
batch_mean1,batch_var1 = tf.nn.moments(fc_1,[0])
# normalization
fc_1_batch = (fc_1-batch_mean1)/tf.sqrt(batch_var1+epsilon)
# scaling factor
alpha1 = tf.Variable(tf.ones([Hidden_Layer1]))
#offset
beta1 = tf.Variable(tf.zeros([Hidden_Layer1]))
fc_1_BN = alpha1*fc_1_batch + beta1
# Applying activation function
hc_1 = tf.nn.elu(fc_1_BN)
tf.summary.histogram("Act",hc_1)
# dropout
Layer1_dropout = tf.nn.dropout(hc_1,keep_prob)


# generating weights and biases between the input and the 2nd Hidden_Layer
W_fc2 = local_weight_variable([Hidden_Layer1, Hidden_Layer2])
b_fc2 = local_bias_variable([Hidden_Layer2])

# Applying summation operation
fc_2 = tf.matmul(Layer1_dropout, W_fc2) + b_fc2

# Applying batch_normalization
batch_mean,batch_var = tf.nn.moments(fc_2,[0])
fc_2_batch = (fc_2-batch_mean)/tf.sqrt(batch_var+epsilon)
alpha = tf.Variable(tf.ones([Hidden_Layer2]))
beta = tf.Variable(tf.zeros([Hidden_Layer2]))
fc_2_BN = alpha*fc_2_batch + beta
hc_2 = tf.nn.elu(fc_2_BN)
tf.summary.histogram("Act1",hc_2)
# dropout
h_fc2_drop = tf.nn.dropout(hc_2, keep_prob)

# Layer 3
W_fc3 = local_weight_variable([Hidden_Layer2, Hidden_Layer3])
b_fc3 = local_bias_variable([Hidden_Layer3])

# Applying summation operation
fc_3 = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

# Applying batch_normalization
batch_mean3,batch_var3 = tf.nn.moments(fc_3,[0])
fc_3_batch = (fc_3-batch_mean3)/tf.sqrt(batch_var3+epsilon)
alpha3 = tf.Variable(tf.ones([Hidden_Layer3]))
beta3 = tf.Variable(tf.zeros([Hidden_Layer3]))
fc_3_BN = alpha3*fc_3_batch + beta3
hc_3 = tf.nn.elu(fc_3_BN)
tf.summary.histogram("Act2",hc_3)
h_fc3_drop = tf.nn.dropout(hc_3, keep_prob)

W_fc4 = weight_variable([Hidden_Layer3, label_classes])
b_fc4 = bias_variable([label_classes])

y = tf.matmul(h_fc3_drop, W_fc4) + b_fc4
batch_mean2,batch_var2 = tf.nn.moments(y,[0])
y_batch = (y - batch_mean2)/tf.sqrt(batch_var2+epsilon)
alpha2 = tf.Variable(tf.ones([label_classes]))
beta2 = tf.Variable(tf.zeros([label_classes]))
y_BN = alpha2*y_batch + beta2
y_pred = tf.nn.softmax(y_BN)

LEARNING_RATE = 0.001
log_path = "/tmp/face_recog"
# cost function
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits = y_pred))
tf.summary.scalar("Xentropy",cross_entropy)

# optimisation function
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
tf.summary.scalar("Accuracy",accuracy)
predict = tf.argmax(y_pred,1)
TRAINING_ITERATIONS = 3000
    
DROPOUT = 0.5
BATCH_SIZE = 50
VALIDATION_SIZE = 7178
epochs_completed = 0
index_in_epoch = 0
num_examples = X_train.shape[0]

# serve data by batches
def next_batch(batch_size):
    
    global X_train
    global Y_train
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        X_train = X_train[perm]
        Y_train = Y_train[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return X_train[start:end], Y_train[start:end]

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
Merged = tf.summary.merge_all()
Writer = tf.summary.FileWriter(log_path,graph=tf.get_default_graph())
sess.run(init)
train_accuracies = []
validation_accuracies = []
x_range = []
display_step=1
for i in range(TRAINING_ITERATIONS):

    #get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)        

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 
                                                  y_: batch_ys, 
                                                  keep_prob: 1.0})       
        if(VALIDATION_SIZE):
           # validation_accuracy = accuracy.eval(feed_dict={ x: X_test[0:BATCH_SIZE], 
                                                            #y_: Y_test[0:BATCH_SIZE], 
                                                            #keep_prob: 1.0})        
            validation_accuracy,summary = sess.run([accuracy,Merged],feed_dict={x:X_test[0:BATCH_SIZE],y_: Y_test[0:BATCH_SIZE],keep_prob:1.0})
            print('training_accuracy / validation_accuracy --> %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
            Writer.add_summary(summary,i)
            
            
            validation_accuracies.append(validation_accuracy)
            
        else:
             print('training_accuracy --> %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)
   
        # increase display_step
    if i%(display_step*10) == 0 and i and display_step<100:
            display_step *= 10
    # train on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})
    
    
