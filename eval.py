import numpy as np
import cv2
from sklearn.metrics.cluster import rand_score #no. of matching pairs / total no of pairs
from sklearn

def pixelAccuracy(imPred, imLab):
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled

    return pixel_accuracy, pixel_correct, pixel_labeled

def iou(target,prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
a    return iou_score

pixelAcc = 0
Iou = 0
Rscore = 0
for i in range(0,400):
    labpath = 'saved_images/'+str(i)+'.png'
    imgpath = 'saved_images/pred_'+str(i)+'.png'
    img = cv2.imread(imgpath,0)
    lbl = cv2.imread(labpath,0)
    pAcc,_,_ = pixelAccuracy(img, lbl)
    iou_score = iou(lbl,img)
    Iou += iou_score
    r_score = rand_score(lbl.flatten(),img.flatten())
    Rscore += r_score
    pixelAcc += pAcc

print((400.0-pixelAcc)/400)

