# See: https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import load_img, img_to_array
from tqdm import tqdm
import os
import argparse

base_path = 'datasets' 
base_output_dir = 'data'
current_dataset = 'fake_baseline'

def get_datagen():
  return ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

def load_images_paths(images_paths):
  return [os.path.join(images_paths, filename) for filename in os.listdir(images_paths)]

def generate(input_path, output_path):
  datagen = get_datagen()
  images_paths = load_images_paths(input_path)
  output_dir = os.path.join(output_path)

  for img_path in tqdm((images_paths)):
    img = load_img(img_path, target_size=(224, 224))
    img_arr = img_to_array(img)
    img_arr = img_arr.reshape((1,) + img_arr.shape)
    img_norm = img_arr / 255.0
    
    # Generate augmented images and save to disk
    num_augmented_images = 5
    for i, _ in enumerate(datagen.flow(img_norm, batch_size=1, save_to_dir=output_dir, save_prefix='aug', save_format='jpeg')):
        if i >= num_augmented_images:
            break

def cli_setup():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', '-o', help="The desired output directory for the images", required=True)
  parser.add_argument('--input_dir', '-i', help="The desired output directory for the images", required=True)
  return parser

def main():
  parser = cli_setup()
  args = parser.parse_args()

  output_path = args.output_dir
  input_path = args.input_dir

  generate(output_path=output_path, input_path=input_path)

if __name__ == '__main__':
  main()