#!/usr/bin/env python 

import cv2
import torch
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from math import *

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
        if (np.isfinite(pred_pts_2d_2[i][-1])):
	        pred_pts_2d_2[i] = pred_pts_2d_2[i]/pred_pts_2d_2[i][-1]
        else : 
        	print(pred_pts_2d_2[i])
        	continue
        error = pts_2d_2[i] - pred_pts_2d_2[i]
        residual[i,:] = error.reshape(1,3)[0]
    # sq_error = np.asarray([i*i for i in residual])
    # print(sq_error)
    res = residual.flatten()
    # print(res.shape)
    return res


# def find_SIFTKeypoints(image,lidar_image,maxNumFeatures):
# 	# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	gray_image = image
# 	mask_img = np.asarray(lidar_image,dtype=np.uint8)
# 	detector = cv2.xfeatures2d.SIFT_create(maxNumFeatures)
# 	features = detector.detect(gray_image,mask_img)

def find_keypoints(image,lidar_image, feature_params):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	mask_img = np.asarray(lidar_image,dtype=np.uint8) #cv2.inRange(lidar_image,lowerb=None,beta=0,aplha=np.amax(lidar_image),norm_type=cv2.NORM_MINMAX)
	# height, width = mask_img.shape
	# mask_img[(int)(4*height/5):,:] = 0
	# plt.imshow(mask_img),plt.show()
	# features = cv2.goodFeaturesToTrack(gray_image, mask=mask_img, **feature_params)
	# sift = cv2.xfeatures2d.SIFT_create()
	orb = cv2.ORB_create(45)
	kps = orb.detect(gray_image,mask_img)
	features = []
	for kp in kps:
		features.append([kp.pt[0],kp.pt[1]]) 
	return np.array(features,dtype = np.float32).reshape(-1,1,2)

def match_keypoints(image1, mask1, image2) : 
	PERCENT_GOOD_MATCHES = 1.0

	gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	mask1 = np.asarray(mask1, dtype=np.uint8)

	gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
	# mask2 = np.asarray(mask2, dtype=np.uint8)

	orb = cv2.ORB_create(45)
	kps1, des1 = orb.detectAndCompute(gray_image1, mask1)
	kps2, des2 = orb.detectAndCompute(gray_image2, None)

	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(des1,des2,None)

	matches.sort(key=lambda x: x.distance, reverse=False)
	numGoodMatches = (int)(len(matches) * PERCENT_GOOD_MATCHES)

	matches = matches[:numGoodMatches]

	features1 = np.zeros((len(matches),2), dtype = np.float32)
	features2 = np.zeros((len(matches),2), dtype = np.float32)
	for i,m in enumerate(matches) : 
		features1[i,:] = kps1[m.queryIdx].pt
		features2[i,:] = kps2[m.trainIdx].pt

	return features1, features2


def track_features(image,features,next_image,lk_params):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	next_gray_image = cv2.cvtColor(next_image,cv2.COLOR_BGR2GRAY)
	next_features, status, error = cv2.calcOpticalFlowPyrLK(gray_image,next_gray_image, features,None,**lk_params)
	return next_features,status, error

def compute_3D_points(lidar_image,features,K):
	fx = K[0,0]
	fy = K[1,1]
	cx = K[0,2]
	cy = K[1,2]
	pts_3d_1 = []
	for i in range(len(features)):
		x = int(features[i,0])
		y = int(features[i,1])
		depth = lidar_image[y,x,0]
		pts_3d_1.append([(x-cx)*depth/fx,(y-cy)*depth/fy,depth])
	return np.asarray(pts_3d_1)

class torchCloud : 
	def __init__(self,K,height,width):
		self.K = K
		self.height, self.width = height, width
		self.cu, self.cv = float(K[0,2]), float(K[1,2])
		# self.cu, self.cv = width/2, height/2
		self.fu, self.fv = float(K[0,0]), float(K[1,1])

		self.U = torch.arange(0,width).expand(height,width)
		self.V = torch.arange(0,height).expand(width,height).t()

		self.X_cam = (self.U - self.cu) / self.fu
		self.Y_cam = (self.V - self.cv) / self.fv

	def calcCamPoints(self,depth) : 
		X = torch.unsqueeze(self.X_cam * depth,-1)
		Y = torch.unsqueeze(self.Y_cam * depth,-1)
		depth = torch.unsqueeze(depth,-1)

		depth_np = depth.numpy()
		index = np.where(depth_np!=0)
		print(index)
		print(len(index))
		X = X[index]
		Y = Y[index]
		depth = depth[index]
		return torch.cat((X,Y,depth),-1)

class npCloud :
	def __init__(self,K,height,width) :
		self.K = K
		self.height, self.width = height, width

	def calcCamPoints(self,depth) :
		idx = np.where(depth!=0)
		pts = []
		for i in range(len(idx[0])):
		    u = idx[1][i]
		    v = idx[0][i]
		    z = depth[v,u]
		    pts.append([u*z,v*z,z])

		pts = np.asarray(pts).reshape(-1,3).T
		K_inv = np.linalg.inv(self.K)
		pts3 = np.matmul(K_inv,pts)
		return pts3