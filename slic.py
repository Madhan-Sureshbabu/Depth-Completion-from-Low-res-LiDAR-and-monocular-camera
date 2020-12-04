# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import mean_squared_error
# import matplotlib.pyplot as plt
from skimage import color
import numpy as np
import cv2

# from data_read import Data_load


class Cluster : 
	cluster_id = -1
	def __init__(self,l,a,b,row,col,masked_img,masked_lidar_img,num_lidar_points,index) :
		self.l = l
		self.a = a
		self.b = b
		self.mean_row = row
		self.mean_col = col
		# self.pixels = []
		self.masked_img = masked_img
		self.masked_lidar_img = masked_lidar_img
		self.num_lidar_points = num_lidar_points
		self.match_index_prev_img = -1
		self.match_index_next_img = -1
		Cluster.cluster_id += 1

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
		self.imglab = color.rgb2lab(img)#cv2.cvtColor(img,cv2.COLOR_RGB2Lab)
		self.lidar_img = lidar_img

	def init_clusters(self, segments) :
		self.segments = segments 
		self.h, self.w = self.segments.shape
		num_clusters = np.unique(self.segments).shape[0]
		print("num_clusters : ",num_clusters)
		for i in range(num_clusters):
			mask = np.where(self.segments==i,1,0)
			rows, cols = np.where(self.segments==i)
			# print("Cluster id {}, total pixels : {} ".format(i,rows.shape[0]))
			masked_lidar_img = mask * self.lidar_img
			num_lidar_points = np.where(masked_lidar_img!=0)[0].shape[0]

			mask = np.expand_dims(mask,-1)
			mask = np.concatenate((mask,mask,mask),2)
			masked_img = self.imglab * mask
			l, a, b = np.mean(np.mean(masked_img,0),0)
			mean_row, mean_col = int(rows.mean()), int(cols.mean())
			ratio = mean_col/self.w
			# print("ratio : ",ratio)
			if (num_clusters >= 100 and (ratio >= 0.25 and ratio <= 0.75)) :
				c = Cluster(l,a,b,mean_row,mean_col,masked_img,masked_lidar_img,num_lidar_points,i ) 
				self.clusters[c.cluster_id] = c

			elif (num_clusters < 100 and (ratio <0.25 or ratio > 0.75))  : 
				c = Cluster(l,a,b,mean_row,mean_col,masked_img,masked_lidar_img,num_lidar_points,i ) 
				self.clusters[c.cluster_id] = c



		print("added clusters : ",len(self.clusters))


# data_loader = Data_load(64,frames=5)
# _, _, img, _ = data_loader.read_frames(batch_size=1,index=79693)
# print(data_loader.index)
# img1, img2 = img[0], img[1]

# numSegments = 50
# compactness = 10.0
# segments1 = slic(img1,n_segments = numSegments, sigma = 5, compactness=compactness, max_iter=100, convert2lab=True)
# segments2 = slic(img2,n_segments = numSegments, sigma = 5, compactness=compactness, max_iter=100, convert2lab=True)

# fig = plt.figure("Superpixels %d segments, image : %d" % (numSegments,0))
# ax = fig.add_subplot(1,1,1)
# ax.imshow(mark_boundaries(img1,segments1))

# fig2 = plt.figure("Superpixels %d segments, image : %d" % (numSegments,1))
# ax2 = fig2.add_subplot(1,1,1)
# ax2.imshow(mark_boundaries(img2,segments2))

# numSegments = 300
# segments3 = slic(img1,n_segments = numSegments, sigma = 5, compactness=compactness, convert2lab=True)
# segments4 = slic(img2,n_segments = numSegments, sigma = 5, compactness=compactness, convert2lab=True)

# # mask = np.expand_dims(np.where(segments3==0,1,0),-1)
# # print(mask.shape)
# # mask = np.concatenate((mask,mask,mask),2)
# # print(mask.shape)
# # img1_masked = img1 * mask
# # lab = np.mean(np.mean(img1_masked,0),0)
# # print(lab)
# # print(lab.shape)

# # spx1 = SuperPixels(img1,segments3)
# # spx1.init_clusters()

# # spx2 = SuperPixels(img2,segments4)
# # spx2.init_clusters()

# # print(segments3)
# # x = np.where(segments3==0)[0]
# # y = np.where(segments3==0)[1]
# # print(x)
# # print(x.shape[0])
# # print(x.mean())
# # xx = int(x.mean())
# # yy = int(y.mean())
# # print(segments3[xx,yy])
# # print(np.unique(segments3))
# # print(np.unique(segments3).shape[0])







# fig5 = plt.figure("Superpixels %d segments, image : %d" % (numSegments,0))
# ax5 = fig5.add_subplot(1,1,1)
# ax5.imshow(mark_boundaries(img1,segments3))

# fig6 = plt.figure("Superpixels %d segments, image : %d" % (numSegments,1))
# ax6 = fig6.add_subplot(1,1,1)
# ax6.imshow(mark_boundaries(img2,segments4))

# fig7 = plt.figure("Original image 0")
# ax7 = fig7.add_subplot(1,1,1)
# ax7.imshow(img1)

# fig8 = plt.figure("Original image 1")
# ax8 = fig8.add_subplot(1,1,1)
# ax8.imshow(img2)

# # numSegments = 150
# # segments5 = slic(img1,n_segments = numSegments, sigma = 5, compactness=5.0, max_iter=100, convert2lab=True)
# # segments6 = slic(img2,n_segments = numSegments, sigma = 5, compactness=5.0, max_iter=100, convert2lab=True)

# # fig9 = plt.figure("Superpixels %d segments, image : %d" % (numSegments,0))
# # ax9 = fig9.add_subplot(1,1,1)
# # ax9.imshow(mark_boundaries(img1,segments5))

# # fig10 = plt.figure("Superpixels %d segments, image : %d" % (numSegments,1))
# # ax10 = fig10.add_subplot(1,1,1)
# # ax10.imshow(mark_boundaries(img2,segments6))

# # plt.show()
# # print(segments1)

# # ssim1 = ssim(img1,img2,data_range=img1.max()-img1.min(),multichannel=True)
# # mse1 = mean_squared_error(img1, img2)

# # print(ssim1)
# # print(mse1)
# # # fig, axes = plt.subplots(1,1)#,sharex=True,sharey=True)
# # # ax = axes.ravel()

# # fig = plt.figure("ssim")
# # ax = fig.add_subplot(1,1,1)

# # ax.imshow(img1,cmap=plt.cm.gray,vmin=0,vmax=1)
# # # ax.set_xlabel(label.format(mse1,ssim1))
# # # ax.set_title('Original image')

# plt.show()

# img1lab = cv2.cvtColor(img1, cv2.COLOR_RGB2Lab)
# img2lab = cv2.cvtColor(img2, cv2.COLOR_RGB2Lab)

# plt.imshow(img1lab)
# plt.show()

# plt.imshow(img2lab)
# plt.show()

# print(img1lab.shape)