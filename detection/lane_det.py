import time
import os
import cv2
import numpy as np
from numpy.linalg import inv

class Zebra_det(object):
    def __init__(self,thres):
        print('-------------------loading zebra crossing detector-------------------')
        ori_point=np.array([[439,587],[770,592],[874,713],[310,712]],dtype=np.float32)
        self.clock_wise=self.target_vertax_point(ori_point)
        self.matrix=cv2.getPerspectiveTransform(ori_point,self.clock_wise)
        self.thres=thres


    def predict(self,patches):
        """
    predict zebra crossing for every patches 1 is zc 0 is background
        """
        # print(len(patches))
        labels = np.zeros(len(patches))
        index = 0
        Amplitude, theta = patches
        mask = (Amplitude > 25).astype(np.float32)
        h, b = np.histogram(theta[mask.astype(np.bool)], bins=range(0, 80,5                  ))
        low, high = b[h.argmax()], b[h.argmax() + 1]
        newmask = ((Amplitude > 25) * (theta <= high) * (theta >= low)).astype(np.float32)
        value = ((Amplitude * newmask) > 0).sum()

        if value > self.thres:
            # print('@@@@@@@@@@@@@@@@@@@@@@@',value)
            labels[index] = 1

        # print(h)
        # print(low, high)
        # print(value)

        return labels


    def preprocessing(self,img):
        """
    Take the blue channel of the original image and filter it smoothly
        """
        kernel1 = np.ones((3, 3), np.uint8)
        kernel2 = np.ones((5, 5), np.uint8)
        gray = img[:, :, 0]
        gray = cv2.medianBlur(gray, 5)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel1, iterations=4)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel2, iterations=3)
        return gray


    def getGD(self,canny):
        """
    return gradient mod and direction
        """
        sobelx = cv2.Sobel(canny, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(canny, cv2.CV_32F, 0, 1, ksize=3)
        theta = np.arctan(np.abs(sobely / (sobelx + 1e-10))) * 180 / np.pi
        Amplitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        mask = (Amplitude > 30).astype(np.float32)
        Amplitude = Amplitude * mask
        return Amplitude, theta


    def getlocation(self,indices, labels, Ni, Nj):
        """
    return if there is a zebra cossing
    if true, Combine all the rectangular boxes as its position
    assume a picture has only one zebra crossing
        """
        zc = indices[labels == 1]
        if len(zc) == 0:
            return 0, None
        else:
            xmin = int(min(zc[:, 1]))
            ymin = int(min(zc[:, 0]))
            xmax = int(xmin + Nj)
            ymax = int(max(zc[:, 0]) + Ni)
            return 1, ((xmin, ymin), (xmax, ymax))

    def target_vertax_point(self,points):
        w1=np.linalg.norm(points[0]-points[1])
        w2=np.linalg.norm(points[2]-points[3])
        w=w1 if w1>w2 else w2
        h1 = np.linalg.norm(points[1] - points[2])
        h2 = np.linalg.norm(points[3] - points[0])
        h = h1 if h1 > h2 else h2
        w=int(round(w))
        h=int(round(h))
        tl=[0,0]
        tr=[w,0]
        br=[w,h]
        bl=[0,h]
        return np.array([tl,tr,br,bl],dtype=np.float32)
    def __call__(self, img):


        pers_img = cv2.warpPerspective(img,self.matrix,(self.clock_wise[2][0],self.clock_wise[2][1]))


    # frame=cv2.imread('/home/rvlab/Pictures/Screenshot 2020-06-09 16:55:59.png')
        gray = self.preprocessing(pers_img)
        canny = cv2.Canny(gray, 30, 90, apertureSize=3)
        Amplitude, theta = self.getGD(canny)

        # indices, patches = zip(
        #     *sliding_window(Amplitude, theta, patch_size=(Ni, Nj)))  # use sliding_window get indices and patches
        labels = self.predict((Amplitude,theta))  # predict zebra crossing for every patches 1 is zc 0 is background
        # indices = np.array(indices)
        # ret, location = getlocation(indices, labels, Ni, Nj)
        # draw
        # if DEBUG:
        #     for i, j in indices[labels == 1]:
        #         cv2.rectangle(img, (j, i), (j + Nj, i + Ni), (0, 0, 255), 3)
        if labels[0]==1 :
            return True
        else:
            return False