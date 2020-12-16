from skimage import color
import numpy as np
import cv2

class Cluster : 
	def __init__(self,l,a,b,row,col,masked_img,masked_lidar_img,num_lidar_points,mean_depth,index) :
		self.l = l
		self.a = a
		self.b = b
		self.mean_row = row
		self.mean_col = col
		self.masked_img = masked_img
		self.masked_lidar_img = masked_lidar_img
		self.num_lidar_points = num_lidar_points
		self.match_index_prev_img = -1
		self.match_index_next_img = -1
		self.mean_depth = mean_depth
		self.cluster_id = index

	def get_centre_xy(self):
		return np.asarray([[self.mean_col, self.mean_row]])

	def get_distance(self,cluster,ds,alpha) : 
		color_diff = np.asarray([self.l-cluster.l,self.a-cluster.a,self.b-cluster.b])
		dc = np.linalg.norm(color_diff)
		return alpha * dc + (1-alpha) * ds


	def show(self):
		print("l:{}, a:{}, b:{}, row:{}, col:{} ".format(self.l,self.a,self.b,self.mean_row,self.mean_col))

class SuperPixels :
	def __init__(self, img, lidar_img) : 
		self.clusters = {}
		self.imglab = color.rgb2lab(img)
		self.lidar_img = lidar_img

	def init_clusters(self, segments) :
		self.segments = segments 
		self.h, self.w = self.segments.shape
		num_clusters = np.unique(self.segments).shape[0]
		for i in range(num_clusters):
			mask = np.where(self.segments==i,1,0)
			rows, cols = np.where(self.segments==i)
			masked_lidar_img = mask * self.lidar_img
			num_lidar_points = np.where(masked_lidar_img!=0)[0].shape[0]
			if num_lidar_points!=0 : 
				mean_depth = np.sum(masked_lidar_img) / num_lidar_points
			else : 
				mean_depth = 0

			mask = np.expand_dims(mask,-1)
			mask = np.concatenate((mask,mask,mask),2)
			masked_img = self.imglab * mask
			l, a, b = np.sum(np.sum(masked_img,0),0)/np.where(segments==i)[0].shape[0]
			mean_row, mean_col = int(rows.mean()), int(cols.mean())
			ratio = mean_col/self.w
			# print("ratio : ",ratio)
			keys = self.clusters.keys()
			key = 0 if len(keys)==0 else max(keys)+1
			if (num_clusters >= 100 and (ratio >= 0.25 and ratio <= 0.75)) :
				c = Cluster(l,a,b,mean_row,mean_col,masked_img,masked_lidar_img,num_lidar_points,mean_depth,key ) 
				self.clusters[c.cluster_id] = c

	def update_KDtree(self,tree) : 
		self.tree = tree
