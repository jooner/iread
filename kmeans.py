import os, struct
from array import array

digit_images = os.path.join(".", 'digits/train-images')
digit_labels = os.path.join(".", 'digits/train-labels')

def load_mnist ():
  with open(digit_labels, 'rb') as f:
    labels = array("b", f.read())
  with open(digit_images, 'rb') as f:
    images = array("B", f.read())
  return images, labels

load_mnist()