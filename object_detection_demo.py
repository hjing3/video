import sys, os
import cv2
from PIL import Image
import numpy as np
import ctypes as ct

sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im


def detect_np(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = array_to_image(image)
    num = ct.c_int(0)
    pnum = ct.pointer(num)
    dn.predict_image(net, im)
    dets = dn.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): dn.do_nms_obj(dets, num, meta.classes, nms)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    #free_image(im)
    dn.free_detections(dets, num)
    return res


def main():
   #net = dn.load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
   net = dn.load_net("cfg/yolov3-tiny.cfg", "yolov3-tiny.weights", 0)
   meta = dn.load_meta("cfg/coco.data")

   while True:
      img = raw_input("Type full path to the image to process: ")

      im = np.array(Image.open(img), dtype=np.uint8)
      #r = detect_np(net, meta, im)
      r = dn.detect(net, meta, img)

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
