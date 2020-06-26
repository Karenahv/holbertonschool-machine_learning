#!/usr/bin/env python3
"""Transfer learning"""

import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt


def preprocess_data(X, Y):
    """
    This method pre-processes the data
    for the model
    """
    X_p = K.applications.vgg19.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return (X_p, Y_p)


if __name__ == '__main__':

    """
    Transfer learning of the model VGG19
    and save it in a file cifar10.h5
    """
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    learn_rate = .001
    batch_size = 100
    epochs = 50

    x_train = K.applications.vgg19.preprocess_input(x_train)
    x_test = K.applications.vgg19.preprocess_input(x_test)

    y_train = K.utils.to_categorical(y_train, 10)
    y_test = K.utils.to_categorical(y_test, 10)

    train_datagen = K.preprocessing.image.ImageDataGenerator(rescale=1./255, 
                                                        zoom_range=0.3,
                                                        rotation_range=50,
                                                        width_shift_range=0.2,
                                                        height_shift_range=0.2,
                                                        shear_range=0.2,
                                                        horizontal_flip=True,
                                                        fill_mode='nearest')

    val_datagen = K.preprocessing.image.ImageDataGenerator(rescale=1./255)


    test_datagen = K.preprocessing.image.ImageDataGenerator(rotation_range=2,
                                                            horizontal_flip=True,
                                                            zoom_range=.1) 
    train_datagen.fit(x_train)
    test_datagen.fit(x_test)

    reduce_learning_rate = K.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                         factor=.01,
                                                         patience=3,
                                                         min_lr=1e-5)
    base_model_vgg19 = K.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(32, 32, 3),
        classes=y_train.shape[1])

    for layer in model_vgg19.layers[0:3]:
        layer.trainable = False

    sgradient_d = K.optimizers.SGD(lr=learn_rate,
                                   momentum=.9,
                                   nesterov=False)
    
    model_vgg19 = K.Sequential()
    model_vgg19.add(base_model_vgg19)
    model_vgg19.add(K.layers.Flatten())
    model_vgg19.add(K.layers.Dense(1024, activation=('relu'), input_dim=512))
    model_vgg19.add(K.layers.Dense(512, activation=('relu')))
    model_vgg19.add(K.layers.Dense(256, activation=('relu')))
    model_vgg19.add(K.layers.Dense(128, activation=('relu')))
    model_vgg19.add(K.layers.Dense(10, activation=('softmax')))

    model_vgg19.summary()

    model_vgg19.compile(optimizer=sgradient_d,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])


    history = model_vgg19.fit_generator(train_datagen.flow(x_train,
                                                           y_train,
                                                           batch_size=batch_size),
                                        epochs=epochs,
                                        steps_per_epoch=x_train.shape[0]//batch_size,
                                        validation_data=val_datagen.flow(x_test,
                                                                         y_test,
                                                                         batch_size=batch_size),
                                        validation_steps=250,
                                        callbacks=[reduce_learning_rate], verbose=1)
    
    plt.plot(history.history['accuracy'], linestyle='dashed')
    plt.plot(history.history['val_accuracy'])
    plt.show()
    model_vgg19.save('cifar10.h5')
