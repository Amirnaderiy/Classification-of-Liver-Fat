# Classification of Liver based on fat level

![Alt text](https://github.com/Amirnaderiy/Classification-of-Liver-Fat/blob/main/cr%3Dt_0%2Cw_100.png)


This code defines a neural network for image classification using the EfficientNetB7 model as the convolutional base. It first loads and resizes the images from four groups, each corresponding to a different label, and defines the labels for each group. Then, it loads the EfficientNetB7 model and freezes its convolutional base. Next, it defines the network architecture by adding layers to the base, including a global average pooling layer, two dense layers, and a dropout layer. It compiles the model using the Adam optimizer and categorical crossentropy loss function, and defines the data generators for training, validation, and testing. It also defines a learning rate scheduler using the step_decay function. Finally, it trains the model using the fit method, specifying the data generators for training and validation, the number of epochs, and the learning rate scheduler as callbacks. It then plots the training and validation accuracy and loss over the epochs using Matplotlib.



