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

def compute_disparity_stereoOdomLoader(left_image,right_image,numDisparities):
    left_matcher = cv2.StereoBM_create(numDisparities,blockSize=15)
    disp = np.int16(left_matcher.compute(left_image,right_image))
    return disp

def find_keypoints(Image,Mask,Tile_H,Tile_W,nFeatures):
    featureEngine = cv2.FastFeatureDetector_create()
    H,W = Image.shape
    kp = []
    for y in range(0, H, Tile_H):
        for x in range(0, W, Tile_W):
            Patch_Img = Image[y:y+Tile_H, x:x+Tile_W]
            Patch_Mask = Mask[y:y+Tile_H, x:x+Tile_W]
            keypoints = featureEngine.detect(\
                Patch_Img,mask=Patch_Mask)
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

            if (len(keypoints) > nFeatures):
                keypoints = sorted(keypoints,\
                                   key=lambda x: -x.response)
                for kpt in keypoints[0:nFeatures]:
                    kp.append(kpt)
            else:
                for kpt in keypoints:
                    kp.append(kpt)
        
    trackPts = cv2.KeyPoint_convert(kp)
    trackPts = np.expand_dims(trackPts, axis=1)

    global debug
    if debug == 1:
        print ("# Points Tracked : " + str(len(trackPts)))
    return trackPts

def track_keypoints(Image_1,Image_2,Pts_1,nPts):
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),maxLevel = 3,\
                      criteria = (cv2.TERM_CRITERIA_EPS | \
                      cv2.TERM_CRITERIA_COUNT, 50, 0.03))
    Pts_2, st, err = cv2.calcOpticalFlowPyrLK(\
                     Image_1,Image_2, Pts_1, None,\
                     flags=cv2.MOTION_AFFINE, **lk_params)

    # separate points that were tracked successfully
    ptTrackable = np.where(st == 1, 1,0).astype(bool)
    TrkPts_1 = Pts_1[ptTrackable, ...]
    TrkPts_2 = Pts_2[ptTrackable, ...]
    TrkPts_2 = np.around(TrkPts_2)
    global debug
    if debug == 1:
        print ("Points successfully tracked: " + str(len(Pts_2)))

    error = 4
    errTrackablePts = err[ptTrackable, ...]
    errThreshPts = np.where(errTrackablePts < \
                            error, 1, 0).astype(bool)
    # Dynamically change threshold to get required points
    while np.count_nonzero(errThreshPts) > nPts:
        error = round(error - 0.1,1)
        errThreshPts = np.where(errTrackablePts < \
                                error, 1, 0).astype(bool)

    while np.count_nonzero(errThreshPts) < nPts :
        error = round(error + 0.1,1)
        errThreshPts = np.where(errTrackablePts < \
                                error, 1, 0).astype(bool)
        if error >= 8:
            if debug == 1:
                print ("Max Limit Reached... Exiting loop")
            break

    TrkPts_1 = TrkPts_1[errThreshPts, ...]
    TrkPts_2 = TrkPts_2[errThreshPts, ...]
    if debug == 1:
        print ("Points with error less than " \
               + str(error) + " : " + str(len(TrkPts_1)))

    return TrkPts_1,TrkPts_2


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


# Finding 3D coordinate of selected features using Disparity Map
def Calc_3DPts(DisparityA,PointsA,DisparityB,PointsB,\
               f,base,cx,cy,scale):
    Pts_3DA = []
    Pts_3DB = []
    Pts_2DA = []
    Pts_2DB = []
    for j in range(len(PointsA)):
        PtA = PointsA[j]
        PtB = PointsB[j]
    
        dA = DisparityA[int(PtA[1])][int(PtA[0])]/scale
        dB = DisparityB[int(PtB[1])][int(PtB[0])]/scale
        if dA > 0 and dB > 0:
            Pts_3DA.append([base*(PtA[0] - cx)/dA,\
                            -base*(PtA[1] - cy)/dA, -f*base/dA])
            Pts_3DB.append([base*(PtB[0] - cx)/dB,\
                            -base*(PtB[1] - cy)/dB, -f*base/dB])
            Pts_2DA.append(PtA)
            Pts_2DB.append(PtB)
    return np.asarray(Pts_2DA),np.asarray(Pts_2DB),\
           np.asarray(Pts_3DA),np.asarray(Pts_3DB)

def find_bestPts_ID(point_cloud1,point_cloud2,minReq) :
    dist_difference = 0.05
    max_node = -1
    max_count = 0
    point_cloud1 = np.asarray(point_cloud1)
    point_cloud2 = np.asarray(point_cloud2)
    
    num_points = point_cloud1.shape[0]
    W = np.zeros((num_points,num_points))
    count = 0
    point_clouds_relative_dist = np.zeros((num_points,num_points))
    while max_node == -1:
        for i in range(num_points) : 
            diff_nodes_t1 = point_cloud1 - point_cloud1[i,:]
            diff_nodes_t2 = point_cloud2 - point_cloud2[i,:]
            dist_nodes_t1 = np.linalg.norm(diff_nodes_t1,axis=1)
            dist_nodes_t2 = np.linalg.norm(diff_nodes_t2,axis=1)
            abs_dist = abs(dist_nodes_t1 - dist_nodes_t2)

            point_clouds_relative_dist[i] = \
                np.asarray(abs_dist).T 
            wIdx = np.where(abs_dist < dist_difference)
            W[i,wIdx[0]] = 1
            count = np.sum(W[i,:])
            if count > max_count: 
                max_count = count
                max_node = i
        if max_count < minReq and dist_difference < 0.5 :
            max_count = 0
            max_node = -1
        if max_node == -1:
            dist_difference += 0.01
    count = 0
    clique = [max_node]

    while True :
        max_count = 0
        max_node = 0
        potentialnodes = list()
        Wsub = W[clique,:]
    # print(Wsub)
        for i in range(num_points) : 
            sumclique = np.sum(Wsub[:,i])
            if sumclique == len(clique) : 
                isin = True
            else : 
                isin = False
            if isin == True and i not in clique : 
                potentialnodes.append(i)
        max_count = 0
        max_node = 0 
        for i in range(len(potentialnodes)) : 
            Wsub = W[potentialnodes[i],potentialnodes]
            sumclique = np.sum(Wsub)
            if sumclique > max_count : 
                max_count = sumclique
                max_node = potentialnodes[i]

        if max_count == 0 :
            if len(clique) >= minReq : 
                break
            else :
                dist_difference += 0.05
                for k in range(num_points) : 
                    diff_nodes_t1 = point_cloud1 \
                                    - point_cloud1[k,:]
                    diff_nodes_t2 = point_cloud2 \
                                    - point_cloud2[k,:]
                    dist_nodes_t1 = \
                        np.linalg.norm(diff_nodes_t1,axis=1)
                    dist_nodes_t2 = \
                        np.linalg.norm(diff_nodes_t2,axis=1)
                    abs_dist = abs(dist_nodes_t1 - dist_nodes_t2)
                    point_clouds_relative_dist[k] = \
                        np.asarray(abs_dist).T 
                    wIdx = np.where(abs_dist < dist_difference)
                    W[k,wIdx[0]] = 1

        if len(clique) >= minReq or dist_difference > 10  :
            break
        clique.append(max_node)
    return clique

# Finding Best Points using Outlier Rejection
def find_bestPts_OR(Pts_1,Pts_2,Pts3D_1,Pts3D_2,minReq):
    if len(Pts3D_1) < 6:
        print("ERROR : Less than 6 points")
        print(len(Pts3D_1))
        sys.exit()
    Compare3D = np.zeros((len(Pts3D_1),len(Pts3D_1)))
    for i in range(len(Pts3D_1)):
        for j in range(len(Pts3D_1)):
            Dis_1 = distance.euclidean(Pts3D_1[i],Pts3D_1[j])
            Dis_2 = distance.euclidean(Pts3D_2[i],Pts3D_2[j])
            Compare3D[i,j] = abs(Dis_1-Dis_2)

    Sum3D = np.sum(Compare3D,axis = 1)
    FinalIndex = np.argsort(Sum3D)
    while len(Sum3D) > minReq:
        Compare3D = np.delete(Compare3D,FinalIndex[len(Sum3D)-1],0)
        Compare3D = np.delete(Compare3D,FinalIndex[len(Sum3D)-1],1)
        Pts_1 = np.delete(Pts_1,FinalIndex[len(Sum3D)-1],0)
        Pts_2 = np.delete(Pts_2,FinalIndex[len(Sum3D)-1],0)
        Pts3D_1 = np.delete(Pts3D_1,FinalIndex[len(Sum3D)-1],0)
        Pts3D_2 = np.delete(Pts3D_2,FinalIndex[len(Sum3D)-1],0)
        Sum3D = np.sum(Compare3D,axis = 1)
        FinalIndex = np.argsort(Sum3D)

    return np.asarray(Pts_1),np.asarray(Pts_2),\
           np.asarray(Pts3D_1),np.asarray(Pts3D_2)

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
        error = pts_2d_2[i] - pred_pts_2d_2[i]
        residual[i,:] = error.reshape(1,3)[0]
    res = residual.flatten()
    return res

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
