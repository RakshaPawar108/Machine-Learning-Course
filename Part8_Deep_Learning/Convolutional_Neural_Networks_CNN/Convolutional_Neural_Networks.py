# Convolutional Neural Networks
# Training a CNN to recognize the image of a cat or a dog

# Data Preprocessing is already done manually as the dataset is split into training and test sets just some feature scaling will be required.

# Part 1 - Building the CNN!!!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution

# First parameter is the number of feature detectors AND the number of rows and columns in each feature detector

# Second parameter is the format in which the images should be expected that is the input shape and we will convert our images in that format during the Image Preprocessing part and here we are using the format for coloured images where the convolution will be made into 3D array. Since we don't want the code to run hours, we will use a smaller format(64X64). The first number is the number of channels(for Theano backend), for Tensorflow backend, the first two numbers represent size of channels and third is the number. Here we create 3 for RGB each as cats and dogs both do not have same colours.
        # classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Max Pooling
        # classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding another Convolution layer and applying max pooling to the o/p of the first layer

#         classifier.add(Convolution2D(
#     32, 3, 3, activation='relu'))

#         classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Making the Convolutional layers and Pooling using loop 

no_of_convolutions = 3
no_feature_detectors = 32
fd_rows = 3
fd_cols = 3

for convolutions in range(0, no_of_convolutions):
        if (convolutions == 0):
            print("Layer 1")
            classifier.add(Convolution2D(no_feature_detectors, fd_rows, fd_cols, input_shape = (64, 64, 3), activation = 'relu'))
            classifier.add(MaxPooling2D(pool_size = (2, 2)))

        elif (convolutions >= 2):
            print("Layer 3 onwards..")
            classifier.add(Convolution2D(no_feature_detectors + 32, fd_rows, fd_cols, activation='relu'))
            classifier.add(MaxPooling2D(pool_size=(2, 2)))

        else:
            print("Layer 2")
            classifier.add(Convolution2D(no_feature_detectors, fd_rows,fd_cols, activation='relu'))
            classifier.add(MaxPooling2D(pool_size=(2, 2)))


# Step 3 - Flattening 
classifier.add(Flatten())

# Step 4 - Full Connection

# Making a classic fully connected ANN to put the flattened input vectors for classification

# Making our hidden layers
classifier.add(Dense(output_dim = 128, activation = 'relu'))

# Making our output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting the CNN to the images

# Image Augmentation, basically scaling, resizing, shearing, flipping, etc. using the trick from Keras documentation where we get readymade generated code to apply to directory.

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')

#Parameters in Keras 2 have been changed and steps per epoch and validation steps need to be divided by the batch size.

classifier.fit_generator(
    training_set,
    steps_per_epoch=250,
    epochs = 25,
    validation_data = test_set,
    validation_steps = 63)




