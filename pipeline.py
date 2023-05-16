# See: https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU

def get_data():
    seed_train_validation = 1
    shuffle_value = True
    validation_split = 0.3

    path = os.path.join('training','02')

    print(f'using this dataset: {path}')

    train_ds = tf.keras.utils.image_dataset_from_directory(
      directory=path,
      image_size=(224,224),
      validation_split=validation_split,
      subset='training',
      seed= seed_train_validation,
      shuffle=shuffle_value
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
      directory=path,
      image_size=(224,224),
      validation_split=validation_split,
      subset='validation',
      seed= seed_train_validation,
      shuffle=shuffle_value
    )

    val_batches = tf.data.experimental.cardinality(validation_ds)
    test_ds = validation_ds.take((2*val_batches) // 3)
    validation_ds = validation_ds.skip((2*val_batches) // 3)

    return train_ds, test_ds, validation_ds

def loss_fn(labels, predictions):
    return tf.math.confusion_matrix(
    labels, predictions, num_classes=2, weights=None, dtype=tf.dtypes.int32,
    name=None).numpy()[0,1]

def get_vgg19():
    print('using vgg19')
    model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    opt = tf.keras.optimizers.Adam(
      learning_rate= 0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False)

    tp = tf.keras.metrics.TruePositives(thresholds=None, name=None, dtype=None)
    tn = tf.keras.metrics.TrueNegatives(thresholds=None, name=None, dtype=None)
    fp = tf.keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None)
    fn = tf.keras.metrics.FalseNegatives(thresholds=None, name=None, dtype=None)

    classifier = Sequential()
    classifier.add(model)
    classifier.add(Flatten())
    classifier.add(Dense(1024, activation='relu'))
    classifier.add(Dense(1, activation = 'sigmoid'))
    # Add a dense output layer for classification
    classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy', tp,tn,fp,fn])

    return classifier

def get_efficientnetb0():
    print('using efficientnetb0')

    model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/efficientnet/b0/feature-vector/1", 
                   trainable=False)
    ])

    # Add a dense output layer for classification
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.add(Dense(1, activation = 'sigmoid'))

    opt = tf.keras.optimizers.Adam(
      learning_rate= 0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False)

    tp = tf.keras.metrics.TruePositives(thresholds=None, name=None, dtype=None)
    tn = tf.keras.metrics.TrueNegatives(thresholds=None, name=None, dtype=None)
    fp = tf.keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None)
    fn = tf.keras.metrics.FalseNegatives(thresholds=None, name=None, dtype=None)

    # Add a dense output layer for classification
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy', tp,tn,fp,fn])

    return model

def get_efficientnetb7():
    print('using efficientnetb7')
    
    model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/efficientnet/b7/feature-vector/1", 
                   trainable=False)
    ])

    # Add a dense output layer for classification
    model.add(Dense(10, activation='softmax'))
    model.add(Dense(1, activation = 'sigmoid'))

    opt = tf.keras.optimizers.Adam(
      learning_rate= 0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False)

    tp = tf.keras.metrics.TruePositives(thresholds=None, name=None, dtype=None)
    tn = tf.keras.metrics.TrueNegatives(thresholds=None, name=None, dtype=None)
    fp = tf.keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None)
    fn = tf.keras.metrics.FalseNegatives(thresholds=None, name=None, dtype=None)

    # Add a dense output layer for classification
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy', tp,tn,fp,fn])

    return model

def get_resnet():
    print('using resnet_v2_50')
    
    # Load the ResNet50 model from TensorFlow Hub
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5")
    ])

    # Add a dense output layer for classification
    model.add(Dense(10, activation='softmax'))
    model.add(Dense(1, activation = 'sigmoid'))

    opt = tf.keras.optimizers.Adam(
      learning_rate= 0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False)

    tp = tf.keras.metrics.TruePositives(thresholds=None, name=None, dtype=None)
    tn = tf.keras.metrics.TrueNegatives(thresholds=None, name=None, dtype=None)
    fp = tf.keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None)
    fn = tf.keras.metrics.FalseNegatives(thresholds=None, name=None, dtype=None)

    # Add a dense output layer for classification
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy', tp,tn,fp,fn])

    return model

def get_efficientnetb7_new_arch():
    print('using efficientnetb7 new arch')
    
    classifier = Sequential(hub.KerasLayer("https://tfhub.dev/google/efficientnet/b7/feature-vector/1", 
                   trainable=False))
    classifier.add(Flatten())
    classifier.add(Dense(1024, activation='relu'))
    classifier.add(Dense(1, activation = 'sigmoid'))

    opt = tf.keras.optimizers.Adam(
      learning_rate= 0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False)

    tp = tf.keras.metrics.TruePositives(thresholds=None, name=None, dtype=None)
    tn = tf.keras.metrics.TrueNegatives(thresholds=None, name=None, dtype=None)
    fp = tf.keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None)
    fn = tf.keras.metrics.FalseNegatives(thresholds=None, name=None, dtype=None)

    # Add a dense output layer for classification
    classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy', tp,tn,fp,fn])

    return classifier

def get_mesonet():
    print('using mesonet')
    x = Input(shape = (224, 224, 3))
    
    x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
    
    x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
    
    x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
    
    x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
    x4 = BatchNormalization()(x4)
    x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
    
    y = Flatten()(x4)
    y = Dropout(0.5)(y)
    y = Dense(16)(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Dropout(0.5)(y)
    y = Dense(1, activation = 'sigmoid')(y)

    model = KerasModel(x,y)

    opt = tf.keras.optimizers.Adam(
      learning_rate= 0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False)

    tp = tf.keras.metrics.TruePositives(thresholds=None, name=None, dtype=None)
    tn = tf.keras.metrics.TrueNegatives(thresholds=None, name=None, dtype=None)
    fp = tf.keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None)
    fn = tf.keras.metrics.FalseNegatives(thresholds=None, name=None, dtype=None)

    # Add a dense output layer for classification
    model.compile(optimizer = opt, loss = 'mean_squared_error', metrics = ['accuracy', tp,tn,fp,fn])

    return model

def get_new_arch():
    print('using chatgpt')
    # Input layer
    input_shape = (224, 224, 3)
    inputs = Input(shape=input_shape)

    # Convolutional layers
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Flatten and dense layers
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    # Create the model
    model = KerasModel(inputs=inputs, outputs=outputs)

    opt = tf.keras.optimizers.Adam(
      learning_rate= 0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-07,
      amsgrad=False)

    tp = tf.keras.metrics.TruePositives(thresholds=None, name=None, dtype=None)
    tn = tf.keras.metrics.TrueNegatives(thresholds=None, name=None, dtype=None)
    fp = tf.keras.metrics.FalsePositives(thresholds=None, name=None, dtype=None)
    fn = tf.keras.metrics.FalseNegatives(thresholds=None, name=None, dtype=None)

    # Add a dense output layer for classification
    model.compile(optimizer = opt, loss = 'mean_squared_error', metrics = ['accuracy', tp,tn,fp,fn])
    return model

def main():
    #classifier = get_vgg19()
    #classifier = get_efficientnetb0()
    #classifier = get_efficientnetb7()
    #classifier = get_resnet()
    #classifier = get_efficientnetb7_new_arch()
    #classifier = get_mesonet()
    classifier = get_new_arch()

    train_ds, test_ds, val_ds = get_data()

    my_callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=10),
      tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints/model.{epoch:02d}-{val_loss:.2f}.h5'),
      tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ]

    history = classifier.fit(train_ds, validation_data=test_ds, epochs = 100, shuffle=True, batch_size=32, callbacks=my_callbacks)

    # Evaluate the performance on the validation set
    val_result = classifier.evaluate(val_ds)
    print(f'Validation loss: {val_result[0]:.2f}, validation accuracy: {val_result[1]:.2%}')

    # Test the performance on the test set
    test_result = classifier.evaluate(test_ds)
    print(f'Test loss: {test_result[0]:.2f}, test accuracy: {test_result[1]:.2%}')

    df = pd.DataFrame()
    for idx, key in enumerate(history.history.keys()):
        row = pd.DataFrame(data={
          'item': key,
          'max': max(history.history[key]),
          'mean': np.mean(history.history[key]),
          'min': min(history.history[key]),
          'std': np.std(history.history[key]),
        }, index=[idx])
        df = pd.concat([df,row])

    df.sort_values(by=['item'],ascending=True)
    print(df)


if __name__ == '__main__':
    t = time.process_time()
    main()
    elapsed_time = time.process_time() - t
    print(f'All done. Runned in {elapsed_time} seconds')