import os
from PIL import Image
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
import tensorflow as tf
import cv2
import sys
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

import glob
import ntpath

import skimage.io as io
import scipy.io as sio

from skimage.io import imsave, imread
from skimage import img_as_ubyte
from skimage.transform import rescale, resize
from math import *
sys.path.append("../")

sys.path.append("../")

def genEulerZXZMatrix(psi, theta, sigma):
    c1 = cos(psi)
    s1 = sin(psi)
    c2 = cos(theta)
    s2 = sin(theta)
    c3 = cos(sigma)
    s3 = sin(sigma)

    mat = np.zeros((3,3))

    mat[0,0] = (c1*c3) - (s1*c2*s3)
    mat[0,1] = (-c1 * s3) - (s1 * c2 * c3)
    mat[0,2] = (s1 * s2)
    mat[1,0] = (s1 * c3) + (c1 * c2 * s3)
    mat[1,1] = (-s1 * s3) + (c1 * c2 * c3)
    mat[1,2] = (-c1 * s2)
    mat[2,0] = (s2 * s3)
    mat[2,1] = (s2 * c3)
    mat[2,2] = c2

    return mat


def get_paths_and_transform(position):
    if position=='left':
        root_d = os.path.join('./odometry_dataset_gray/dataset/sequences/00/image_0/*.png')
    else : 
        root_d = os.path.join('./odometry_dataset_gray/dataset/sequences/00/image_1/*.png')
    image_paths=glob.glob(root_d)
    image_paths = sorted(image_paths)
    return image_paths


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8') # in the range [0,255]
    rgb_png =np.array(Image.fromarray(rgb_png).resize((1216,352), Image.NEAREST))
    img_file.close()
    return rgb_png

def read_ground_truth(direc,frame_id):
    scene_id = direc[-3]
    ground_truth_dir = direc[0]+'/'+direc[1]+'/poses/'+scene_id+'.txt'
    f = open(ground_truth_dir)
    for i,line in enumerate(f):
        if (i==frame_id):
            T = np.asarray(line.split(' '),dtype=float).reshape(3,4)
            T = np.vstack((T,np.asarray([0,0,0,1])))
            return T

class Data_load():
    def __init__(self,position):
        img_path = get_paths_and_transform(position)
        self.img_path=img_path
        self.num_sample=[i for i in range(len(self.img_path))]
        # np.random.shuffle(self.num_sample)
        self.index=0
        self.total_sample=len(self.img_path)
       
    def read_batch(self,batch_size=5,if_removal=False):
        i=0
        img_batch=[]

        while (i<(batch_size)):
            i=i+1
            img=rgb_read(self.img_path[self.num_sample[self.index]])        
            img_batch.append(img)
            self.index=self.index+1
                        
        if self.index+batch_size>self.total_sample:
            return [0],[1],[2]
        else:
            return  np.asarray(img_batch)


def compute_disparity_stereoOdomLoader(left_image,right_image,numDisparities):
    left_matcher = cv2.StereoBM_create(numDisparities,blockSize=15)
    disp = np.int16(left_matcher.compute(left_image,right_image))
    return disp

def find_keypoints_stereoOdomLoader(image,mask_img, feature_params, thresh, cam, disparity_threshold_low,disparity_threshold_high):
    # disparity_threshold_low = 5.0
    # disparity_threshold_high = 15.0
    rows,cols = image.shape
    disp_view1 = cv2.normalize(mask_img,None,beta=0,alpha=np.amax(mask_img)/16,norm_type=cv2.NORM_MINMAX)
    disp_view1 = np.uint8(disp_view1)
    mask_img1 = cv2.inRange(disp_view1,int(cam.f*cam.base/disparity_threshold_high),int(cam.f*cam.base/disparity_threshold_low))
    temp_th = disparity_threshold_low
    mask_img1 = cv2.inRange(disp_view1,int(cam.f*cam.base/disparity_threshold_high),int(cam.f*cam.base/disparity_threshold_low))


    while (np.sum(mask_img1)<20000000):
        temp_th -= 1
        mask_img1 = cv2.inRange(disp_view1,int(cam.f*cam.base/disparity_threshold_high),int(cam.f*cam.base/temp_th))
        if temp_th == 3:
            break

    mask_img1[-int(rows/4):,:] = 0
    mask_img1[:int(rows/10),:] = 0
    mask_img1[:,-int(3*cols/20):] = 0   
    plt.imshow(disp_view1)
    plt.show()

    # fast = cv2.FastFeatureDetector_create(thresh,False)
    # key_pts = fast.detect(image,mask_img1)
    # features = [[kp.pt[0],kp.pt[1]] for kp in key_pts]

    orb = cv2.ORB_create()
    kps = orb.detect(image,mask_img)
    features = []
    for kp in kps:
        features.append([kp.pt[0],kp.pt[1]]) 
    features = np.array(features,dtype=np.float32).reshape(-1,1,2)

    print("features shape ",features.shape)

    # features = cv2.goodFeaturesToTrack(image, mask=mask_img1, **feature_params)
    # print(features.shape)
    return features


def match_features_stereoOdomLoader(image,next_image,disp1,disp2,thresh,cam,disparity_threshold_low,disparity_threshold_high):
    rows,cols = image.shape

    disp_view1 = cv2.normalize(disp1,None,beta=0,alpha=np.amax(disp1)/16,norm_type=cv2.NORM_MINMAX)
    disp_view1 = np.uint8(disp_view1)
    mask_img1 = cv2.inRange(disp_view1,int(cam.f*cam.base/disparity_threshold_high),int(cam.f*cam.base/disparity_threshold_low))
    temp_th = disparity_threshold_low
    mask_img1 = cv2.inRange(disp_view1,int(cam.f*cam.base/disparity_threshold_high),int(cam.f*cam.base/disparity_threshold_low))
    while (np.sum(mask_img1)<20000000):
        temp_th -= 1
        mask_img1 = cv2.inRange(disp_view1,int(cam.f*cam.base/disparity_threshold_high),int(cam.f*cam.base/temp_th))
        if temp_th == 3:
            break

    mask_img1[-int(rows/4):,:] = 0
    mask_img1[:int(rows/10),:] = 0
    mask_img1[:,-int(3*cols/20):] = 0   
    plt.imshow(disp_view1)
    plt.show()

    disp_view2 = cv2.normalize(disp2,None,beta=0,alpha=np.amax(disp2)/16,norm_type=cv2.NORM_MINMAX)
    disp_view2 = np.uint8(disp_view2)
    mask_img2 = cv2.inRange(disp_view2,int(cam.f*cam.base/disparity_threshold_high),int(cam.f*cam.base/disparity_threshold_low))
    temp_th = disparity_threshold_low
    while (np.sum(mask_img2)<20000000):
        temp_th -= 1
        mask_img2 = cv2.inRange(disp_view2,int(cam.f*cam.base/disparity_threshold_high),int(cam.f*cam.base/temp_th))
        if temp_th == 3:
            break

    mask_img2[-int(rows/4):,:] = 0
    mask_img2[:int(rows/10),:] = 0
    mask_img2[:,-int(3*cols/20):] = 0

    plt.imshow(mask_img2)
    plt.show()
    # features = cv2.goodFeaturesToTrack(image, mask=mask_img, **feature_params)
    # fast = cv2.FastFeatureDetector_create(thresh,False)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image,mask_img1)
    kp2, des2 = orb.detectAndCompute(next_image,mask_img2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key=lambda x:x.distance)

    num_useful_matches = min(20,len(matches))
    print("num_useful_matches ",num_useful_matches)
    features, next_features = [], []
    for i in range(num_useful_matches):
        features.append([kp1[matches[i].queryIdx].pt[0],kp1[matches[i].queryIdx].pt[1]])
        next_features.append([kp2[matches[i].trainIdx].pt[0],kp2[matches[i].trainIdx].pt[1]])
    return np.asarray(features).reshape(-1,2),np.asarray(next_features).reshape(-1,2)

def track_keypoints_stereoOdomLoader(image,features,next_image,lk_params):
    gray_image = image
    next_gray_image = next_image
    next_features, status, error = cv2.calcOpticalFlowPyrLK(gray_image,next_gray_image, features,None,**lk_params)
    return next_features,status, error


class StereoCam():
    def __init__(self) :
        self.base = 0
        self.f = 0
        self.cx = 0
        self.cy = 0
        self.P = np.zeros((3,4))
        self.R_rect = np.zeros((3,3))

    def getCameraParams(self,sceneId):
        f_dir = os.path.join('./odometry_dataset_gray/dataset/sequences/'+sceneId+'/calib.txt')
        f = open(f_dir)
        for i,line in enumerate(f):
            if (i==1):
                P = np.asarray(line.split(' ')[1:],dtype=float).reshape(3,4)
                self.f = P[0,0]
                self.base = -P[0,3]/P[0,0]
                self.cx = P[0,2]
                self.cy = P[1,2]
                self.P_right = P
            if (i==0):
                self.P_left = np.asarray(line.split(' ')[1:],dtype=float).reshape(3,4)

    def compute_3D_points(self,image,disp,features,scale):
        pts_3d = [] 
        for i,pt in enumerate(features):
            x,y = int(pt[0]), int(pt[1])
            delta = disp[y,x]/scale
            pts_3d.append([(x-self.cx)*self.base/delta, (y-self.cy)*self.base/delta, self.base*self.f/delta])
        return np.array(pts_3d)

def func(dof, pts_3d_1, pts_2d_2, P):
    R = genEulerZXZMatrix(dof[0], dof[1], dof[2])
    t = np.array([[dof[3], dof[4], dof[5]]]).reshape(3,1)
    T = np.hstack((R,t))
    T = np.vstack((T,[[0,0,0,1]]))
    forward = np.matmul(P,np.linalg.inv(T))
    residual = np.zeros((len(pts_3d_1),3))
    pred_pts_2d_2 = []
    for i in range(len(pts_3d_1)):
        pred_pts_2d_2.append(np.matmul(forward,pts_3d_1[i]))
        pred_pts_2d_2[i] = pred_pts_2d_2[i]/pred_pts_2d_2[i][-1]
        # print(pred_pts_2d_2[i])
        # print(pts_2d_2[i])
        error = pts_2d_2[i] - pred_pts_2d_2[i]
        residual[i,:] = error.reshape(1,3)[0]
    # sq_error = np.asarray([i*i for i in residual])
    # print(sq_error)
    res = residual.flatten()
    # print(res.shape)
    return res