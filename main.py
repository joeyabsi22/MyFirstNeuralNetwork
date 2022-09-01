import basic_neural_network as bnn
import layer
import loss

import numpy as np
import cv2 as cv
import os

data = []

img_size = 128
data_pts = (img_size ** 2) * 3 # = number of input neurons

folder_dir = 'flowers'
list_dir = list_dir = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

for folder in list_dir:
    for file in os.listdir(os.path.join(folder_dir, folder)):
        if (file.endswith('jpg')):

            img = cv.imread(os.path.join(folder_dir, folder, file))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # just cause
            img = cv.resize(img, (img_size, img_size))

            entry = [0, 0, 0, 0, 0]
            entry[list_dir.index(folder)] = 1

            for x in range(img_size):
                for y in range(img_size):
                    entry.append(img[x, y, 0]) # red
                    entry.append(img[x, y, 1]) # green
                    entry.append(img[x, y, 2]) # blue

            data.append(entry)


data = np.array(data)
np.random.shuffle(data)

training_percentage = 0.1

num_images = len(data)
split_index = (int)(num_images * training_percentage)
num_train_images = split_index
num_test_images = num_images - num_train_images

X_train = np.reshape(data[:split_index, 5:], (num_train_images, data_pts, 1))
Y_train = np.reshape(data[:split_index, :5], (num_train_images, 5, 1))
X_test = np.reshape(data[split_index:, 5:], (num_test_images, data_pts, 1))
Y_test = np.reshape(data[split_index:, :5], (num_test_images, 5, 1))

layers = [layer.Dense(data_pts, 64), layer.Tanh(),
          layer.Dense(64, 5), layer.Softmax()]

nn = bnn.BasicNeuralNetwork(layers, loss = loss.CrossEntropy())

nn.train(X_train, Y_train, epochs = 100)

print(nn.predict(X_test[2]))
print(Y_test[2])