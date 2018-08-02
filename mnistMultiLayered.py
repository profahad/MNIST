import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pylab as plt
from keras.utils import to_categorical
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
import warnings
warnings.filterwarnings("always")

with open('/Users/muhammadfahad/Desktop/EML/MNIST/MNIST_data/train-images-idx3-ubyte.gz', 'rb') as f:
    train_images = extract_images(f)
with open('/Users/muhammadfahad/Desktop/EML/MNIST/MNIST_data/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_labels = extract_labels(f)
with open('/Users/muhammadfahad/Desktop/EML/MNIST/MNIST_data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_images = extract_images(f)
with open('/Users/muhammadfahad/Desktop/EML/MNIST/MNIST_data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_labels = extract_labels(f)

save_file = "model.ckpt"
encoded = to_categorical(train_labels)
flatted_images = train_images.reshape(60000, -1)
flatted_test_images = test_images.reshape(10000, -1)
encoded_labels = to_categorical(test_labels)
# print(flatted_images[0].shape)
# print(encoded[0])
# print(train_labels[0])

x = tf.placeholder(tf.float32, [None, 784])
W1 = tf.Variable(tf.random_normal([784, 600]))
b1 = tf.Variable(tf.random_normal([600]))

W2 = tf.Variable(tf.random_normal([600, 600]))
b2 = tf.Variable(tf.random_normal([600]))

W3 = tf.Variable(tf.random_normal([600, 600]))
b3 = tf.Variable(tf.random_normal([600]))

W0 = tf.Variable(tf.random_normal([600, 10]))
b0 = tf.Variable(tf.random_normal([10]))

y = tf.placeholder(tf.float32, [None, 10])

# 3 Hidden Layers
L1 = tf.nn.relu(tf.matmul(x, W1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
scores = tf.matmul(L3, W0) + b0
pred = tf.nn.softmax(scores)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=scores))
optimizer = tf.train.AdamOptimizer(0.001) # learning rate
train = optimizer.minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()


#epochs = 200
def trainModel():
    for i in range(100):
        sess.run(train, feed_dict={x: flatted_images, y: encoded})
        if i % 10 == 0:
          print(sess.run(loss, feed_dict={x: flatted_images, y: encoded}))
    saver.save(sess, save_file)
    print("Model saved in path: %s" % save_file)

def restoreModel():
    saver.restore(sess, save_file)

def predictTestImage():
    rand = np.random.randint(0, 1000)
    test_img = flatted_test_images[rand].reshape(1, 784)
    print(test_img.shape)
    plt.imshow(flatted_test_images[rand].reshape(28, 28), cmap='gray')
    pr = sess.run(pred, feed_dict={x: test_img})
    print(sess.run(tf.argmax(pr, 1)))
    print(pr)
    plt.show()

def predictExternalImage():
    img = cv2.resize(cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE), (28, 28))
    flat_img = np.array(img).reshape(1, -1)
    pr = sess.run(pred, feed_dict={x: flat_img})
    print(pr)
    print(sess.run(tf.argmax(pr, 1)))
    print(flat_img.shape)
    print(np.array(img).shape)
    plt.imshow(np.array(flat_img).reshape(28,28), cmap='gray')
    plt.show()


# trainModel()
restoreModel()
print("Accuracy : {0:.2f} %".format(sess.run(accuracy, {x: flatted_test_images, y: encoded_labels}) * 100))
# predictTestImage()
predictExternalImage()
