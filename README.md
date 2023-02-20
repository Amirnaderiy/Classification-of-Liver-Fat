# Classification of Liver based on fat level

![Alt text](https://github.com/Amirnaderiy/Classification-of-Liver-Fat/blob/main/cr%3Dt_0%2Cw_100.png)


This code is a convolutional neural network model for image classification using the EfficientNetB7 architecture and transfer learning. The model is trained on a dataset of images divided into four groups, and the goal is to classify new images into one of these four groups.
The code starts by importing the necessary libraries and modules, including NumPy, Matplotlib, PIL, OpenCV, Keras, TensorFlow, and EfficientNetB7. Then, it defines a function to load and resize images, and uses this function to load and resize the images in each group. It also defines the labels for each group.
Next, the code loads the EfficientNetB7 model and creates a new model on top of it, consisting of a GlobalAveragePooling2D layer, two Dense layers, and a Dropout layer. It freezes the convolutional base of the model and compiles the model with the Adam optimizer, categorical cross-entropy loss, and accuracy metric.
Then, the code defines the data generators for training, validation, and testing, using the ImageDataGenerator class from Keras. It also defines a learning rate scheduler using the step_decay function.
Finally, the code trains the model on the training data and evaluates its performance on the validation data for 250 epochs, using the fit method from Keras. It plots the training and validation accuracy and loss over time using Matplotlib.


