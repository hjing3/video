import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import cv2
from PIL import Image
import numpy as np
import darknet as dn
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def main():
   #net = dn.load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
   net = dn.load_net("cfg/yolov3-tiny.cfg", "yolov3-tiny.weights", 0)
   meta = dn.load_meta("cfg/coco.data")

   while True:
      img = raw_input("Type full path to the image to process: ")
      r = dn.detect(net, meta, img)

      im = np.array(Image.open(img), dtype=np.uint8)
      # # Create figure and axes
      fig,ax = plt.subplots(1)

      # Display the image
      ax.imshow(im)

      # Create a Rectangle patch
      for res in r:
          box = res[2]
          rect = patches.Rectangle((box[0]-box[2]/2,box[1]-box[3]/2),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
          ax.add_patch(rect)

      # Add the patch to the Axes
      plt.show()


# Darknet
if __name__== "__main__":
   main()
