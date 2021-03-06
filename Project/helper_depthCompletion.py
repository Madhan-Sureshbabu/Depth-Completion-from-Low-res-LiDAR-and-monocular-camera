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
    res = residual.flatten()
    return res


def find_keypoints(image,lidar_image, feature_params,n_features):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	mask_img = np.asarray(lidar_image,dtype=np.uint8) 
	orb = cv2.ORB_create(n_features)
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

class npCloud :
	def __init__(self,K,height,width) :
		self.K = K
		self.height, self.width = height, width

	def calc3DPoints(self,depth, features=[]) : # returns 3D world co-ordinates of the 2D image features
		pts = []
		if features == [] : 
			idx = np.where(depth!=0)
			for i in range(len(idx[0])):
			    u = idx[1][i]
			    v = idx[0][i]
			    z = depth[v,u]
			    pts.append([u*z,v*z,z])
		else : 
			for i in range(len(features)):
				u = int(features[i,0])
				v = int(features[i,1])
				z = depth[v,u]
				pts.append([u*z,v*z,z])

		pts = np.asarray(pts).reshape(-1,3).T # 3,N
		K_inv = np.linalg.inv(self.K)
		pts3 = np.matmul(K_inv,pts).T
		return pts3 # (N,3)
		
	def transform_points(self,pts,T) : 
		pts = np.hstack((pts,np.ones((pts.shape[0],1)))) # (N,4)
		return np.matmul(T,pts.T).T # (N,4)

	def checkLimits(self,u,v,z,condition):
		u = u[condition]
		v = v[condition]
		z = z[condition]
		return u,v,z

	def calc2DPoints(self,pts3d) :   # pts3d shape (N,3)
		pts2d = np.matmul(self.K,pts3d.T) 
		pts2d[:2,:] /= pts2d[2,:]
		u = pts2d[0,:].astype(int)
		v = pts2d[1,:].astype(int)
		z = pts3d[:,2]
		condition = np.where(u<self.width)
		u,v,z = self.checkLimits(u,v,z,condition)

		condition = np.where(v<self.height)
		u,v,z = self.checkLimits(u,v,z,condition)

		condition = np.where(u>=0)
		u,v,z = self.checkLimits(u,v,z,condition)

		condition = np.where(v>=0)
		u,v,z = self.checkLimits(u,v,z,condition)

		condition = np.where(z>0)
		u,v,z = self.checkLimits(u,v,z,condition)
		return u,v,z

	def calcRMSE(self,img1,img2):
		mask1 = np.where(img1!=0,1,0)
		mask2 = np.where(img2!=0,1,0)
		diff = mask1 * mask2 * (img1 - img2)
		diff_square = diff * diff
		num_diff = np.where(mask1 * mask2 == 1)[0].shape[0]
		rmse = np.sqrt(np.sum(diff_square)/num_diff)
		return rmse

