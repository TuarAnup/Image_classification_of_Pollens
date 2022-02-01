#Convolutional_Neural_Network

#Part-I (Building the Convoultional neural network)

#Importingthe librarires
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing  import image

#Initializing the CNN

classifier = Sequential()

#Step1 -Convolution

classifier.add(Convolution2D(32,3,3,input_shape = (256,256,3),activation='relu'))

#Step2 - Pooling

classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding Other convolutional layer on pooled feature maps 
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step3 - Flattening

classifier.add(Flatten())

#Step 4 -Full Connection

classifier.add(Dense(units = 128,activation='relu'))
classifier.add(Dense(units = 5,activation='softmax'))

#Compiling the CNN

classifier.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

#Fitting the CNN to the images

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('datas/train',
                                                 target_size = (256, 256), 
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('datas/test',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'categorical')
classifier.fit(training_set,
                  steps_per_epoch = 250,
                  epochs = 25,
                  validation_data = test_set)
                #   validation_steps = 2000)


# Predict on a image.
file = image.load_img("20191207T005150504-152-28.png", target_size=(256, 256))
x = image.img_to_array(file)
x = np.expand_dims(x, axis=0)
image = np.vstack([x])

# Predict.
prediction = classifier.predict(image)
print(prediction)