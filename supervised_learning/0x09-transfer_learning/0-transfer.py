#!/usr/bin/env python3
"""python script that trains a convolutional
neural network to classify
the CIFAR 10 dataset"""

import tensorflow.keras as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import vgg16
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator


def get_bottleneck_features(model, input_imgs):
    features = model.predict(input_imgs, verbose=0)
    return features

def preprocess_data(X, Y):
    """pre-processes data

    @X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
    where m is the number of data points
    @Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    Returns: X_p, Y_p
    X_p: numpy.ndarray containing the preprocessed X
    Y_p: numpy.ndarray containing the preprocessed Y"""
    
    X_p = K.applications.vgg16.preprocess_input(
    X, data_format=None)
    Y_p = K.utils.to_categorical(
    y, num_classes=10, dtype='float32')

    return X_P, Y_P

if __name__ == '__main__':
    """Transfer learning VGG16, trains the model and save it
    in a file"""

    batch_size = 32
    epochs = 30
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    trainX = tf.keras.applications.vgg16.preprocess_input(trainX,
                                                          data_format=None)
    testX = tf.keras.applications.vgg16.preprocess_input(testX,
                                                         data_format=None)

    trainX = x_train.astype('float32')
    testX = x_test.astype('float32')
    trainX /= 255
    testX /= 255

    vgg = vgg16.VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=(32, 32, 3),
                      classes=trainY.shape[1])

    output = vgg.layers[-1].output
    output = K.layers.Flatten()(output)
    vgg_model = Model(vgg.input, output)

    vgg_model.trainable = False

    for layer in vgg_model.layers:
        layer.trainable = False

    le = LabelEncoder()
    le.fit(train_labels)
    train_labels_enc = le.transform(trainY)
    validation_labels_enc = le.transform(testY)

    train_features_vgg = get_bottleneck_features(vgg_model, trainX)
    validation_features_vgg = get_bottleneck_features(vgg_model, testX)
    print('Train Bottleneck Features:', train_features_vgg.shape, 
          '\tValidation Bottleneck Features:', validation_features_vgg.shape)

    input_shape = vgg_model.output_shape[1]
    model = Sequential()
    model.add(InputLayer(input_shape=(input_shape,)))
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['accuracy'])

    model.summary()

    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                   horizontal_flip=True, fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow(trainX, train_labels_enc, batch_size=30)
    val_generator = val_datagen.flow(testX, validation_labels_enc, batch_size=20)

    history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                                  validation_data=val_generator, validation_steps=50, 
                                  verbose=1) 
    model.save('cifar10.h5')
    print('Saved trained model in the current directory')

    # Score trained model.
    scores = model.evaluate(testX, testY verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
