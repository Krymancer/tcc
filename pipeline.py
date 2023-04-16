# See: https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import pandas as pd
import os


def get_data():
  seed_train_validation = 1
  shuffle_value = True
  validation_split = 0.3

  train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='data',
    image_size=(224,224),
    validation_split=validation_split,
    subset='training',
    seed= seed_train_validation,
    shuffle=shuffle_value
  )

  validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory='data',
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

def main():
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
  classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy', tp,tn,fp,fn])

  train_ds, test_ds, val_ds = get_data()
  
  history = classifier.fit(train_ds, validation_data=test_ds, epochs = 100, shuffle=True, batch_size=32)

# Evaluate the performance on the validation set
  val_loss, val_acc = model.evaluate(val_ds)
  print(f'Validation loss: {val_loss:.2f}, validation accuracy: {val_acc:.2%}')

  # Test the performance on the test set
  test_preds = model.predict(test_ds)
  test_loss, test_acc = model.evaluate(test_ds)
  print(f'Test loss: {test_loss:.2f}, test accuracy: {test_acc:.2%}')

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
  main()