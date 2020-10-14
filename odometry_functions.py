#!/usr/bin/env python 

import cv2
import numpy as np
from scipy.optimize import least_squares
from math import *

def genEulerZXZMatrix(psi, theta, sigma):
	c1 = cos(psi)
	s1 = sin(psi)
	c2 = cos(theta)
	s2 = sin(theta)
	c3 = cos(sigma)
	s4 = sin(sigma)

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
	t = np.array([[dof[3], dof[4], dof[5]]])
	T = np.hstack((R,t))
	T = np.vstack((T,[[0,0,0,1]]))
	forward = np.matmul(P,np.linalg.inv(T))
	pred_pts_2d_2 = []
	residual = np.zeros((len(pts_3d_1),3))
	for i in range(pts_3d_1):
		pred_pts_2d_2.append(np.matmul(forward,pts_3d_1[i]))
		pred_pts_2d_2[i] = pred_pts_2d_2[i]/pred_pts_2d_2[i][-1]
		error = pts_2d_2[i] - pred_pts_2d_2[i]
		residual[i,:] = error.reshape(1,3)[0]
	return residual

def find_keypoints(image,lidar_image, feature_params):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	mask_img = np.asarray(lidar_image,dtype=np.uint8)
	features = cv2.goodFeaturesToTrack(gray_image, mask=mask_img, **feature_params)
	return features

def track_features(image,features,next_image,lk_params):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	next_gray_image = cv2.cvtColor(next_image,cv2.COLOR_BGR2GRAY)
	next_features, status, error = cv2.calcOpticalFlowPyrLK(gray_image,next_gray_image, features,None,**lk_params)
	return next_features,status, error

def compute_3D_points(lidar_image,features,f,cx,cy):
	pts_3d_1 = []
	pts_2d_2 = []
	for j in range(len(features)):
		x = int(features[i,0,1])
		y = int(features[i,0,0])
		depth = lidar_image[x,y,0]
		pts_3d_1.append([x*depth/f,y*depth/f,depth])
		pts_2d_2.append([x,y])
	return pts_3d_1, pts_2d_2


