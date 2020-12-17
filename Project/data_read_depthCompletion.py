import os
from PIL import Image
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
import cv2
import sys
import numpy as np
import os
import matplotlib.pyplot as plt

import glob
import ntpath

import skimage.io as io
import scipy.io as sio

from skimage.io import imsave, imread
from skimage import img_as_ubyte
from skimage.transform import rescale, resize
sys.path.append("../")

sys.path.append("../")

def get_paths_and_transform(num_line=64):
    if num_line==64:
        root_d = os.path.join('../depth_selection/KITTI/Sparse_Lidar')
    if num_line==32:
    	root_d = os.path.join('../depth_selection/KITTI/Sparse_Lidar_32')
    if num_line==16:
    	root_d = os.path.join('../depth_selection/KITTI/Sparse_Lidar_16')
    root_rgb = os.path.join('../depth_selection/KITTI/RGB')
    glob_sparse_lidar = "train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png"

    glob_sparse_lidar = os.path.join(root_d,glob_sparse_lidar)
    all_lidar_path_with_new=glob.glob(glob_sparse_lidar)
    all_lidar_path_without_new=[i for i in all_lidar_path_with_new if not (('left' in i) or('right' in i)) ]
    paths_sparse_lidar = sorted(all_lidar_path_without_new)
    def get_rgb_paths(p):
        ps = p.split('/')
        pnew = '/'.join([root_rgb]+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
        # print(pnew)
        return pnew
    
    glob_rgb = [get_rgb_paths(i) for i in paths_sparse_lidar]
    return paths_sparse_lidar,glob_rgb

def img_path_to_lidar(img_path):
    #img_path:'./Dataset/KITTI/RGB/train/2011_09_26_drive_0051_sync/image_02/data/0000000432.png'
    path_list=img_path.split('/')
    return path_list[0]+'/'+path_list[1]+'/'+path_list[2]+'/'+'Sparse_Lidar/'+path_list[4]+'/'+path_list[5]+'/proj_depth/velodyne_raw/'+path_list[6]+'/'+path_list[8]

def img_path_to_ground_truth(img_path):
    #img_path:'./Dataset/KITTI/RGB/train/2011_09_26_drive_0051_sync/image_02/data/0000000432.png'
    path_list=img_path.split('/')
    return path_list[0]+'/'+path_list[1]+'/'+path_list[2]+'/'+'ground_truth/'+path_list[4]+'/'+path_list[5]+'/proj_depth/groundtruth/'+path_list[6]+'/'+path_list[8]



    
    
def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8') # in the range [0,255]
    rgb_png =np.array(Image.fromarray(rgb_png).resize((1216,352), Image.NEAREST))
    img_file.close()
    return rgb_png

def depth_new_read(filename):
    depth=io.imread(filename)
    depth=depth/255.0*100
    return depth


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    depth = depth_png.astype(np.float32) / 256.
    # depth[depth_png == 0] = -1.

    depth  = np.array(Image.fromarray(depth).resize((1216,352), Image.NEAREST))

    depth = np.expand_dims(depth,-1)
    return depth


    




class Data_load():
    def __init__(self,input_line=64, frames=5):
        if input_line==64:
        	lidar_path,img_path=get_paths_and_transform(64)
        if input_line==32:
        	lidar_path,img_path=get_paths_and_transform(32)
        if input_line==16:
        	lidar_path,img_path=get_paths_and_transform(16)
        self.lidar_path=lidar_path
        self.img_path=img_path
        self.num_sample=[i for i in range(len(self.img_path))]
        self.index_array = self.num_sample
        # identifying the indices in self.img_path list where the next scene starts 
        scene_id = np.asarray([i.split('/')[-4] for i in self.img_path])
        unique_values, first_index_of_scenes = np.unique(scene_id,return_index=True)
        first_index_of_scenes = np.delete(first_index_of_scenes,0) 
        # removing the last (self.frames-1) indices 
        # before each first_index_of_scenes to prevent overlap
		
        for i in range(frames-1):
        	self.index_array.pop() # deleting last (frames-1) image indices

        for i in first_index_of_scenes :
        	j=1
        	while (j<frames):
	        	self.index_array.remove(i-j)
	        	j += 1

        self.index_array = np.asarray(self.index_array)
        # np.random.shuffle(self.num_sample)
        self.total_sample=len(self.img_path)
        self.frames = frames
        self.index = 0
       
    def read_frames(self, batch_size = 5, if_removal=False, index = None):
        img_frames=[]
        lidar_frames=[]
        gt_frames=[]
        self.index_list = [] # for identifying calibration params using scene number
        k=0
        while (k<batch_size):
            i=0
            if index == None : 
                sample = np.random.choice(self.index_array,size=1) 
                self.index = np.where(self.index_array==sample)[0][0] 
            else : 
                self.index = index
            
            self.index_array = np.delete(self.index_array, self.index)
            self.index_list.append(self.index)

            while (i<(self.frames)):
                img=rgb_read(self.img_path[self.num_sample[self.index+i]])
                depth=depth_read(self.lidar_path[self.num_sample[self.index+i]])
                
                if if_removal:
                    depth=outlier_removal(depth)

                gt_path=img_path_to_ground_truth(self.img_path[self.num_sample[self.index+i]])
                ground_truth=depth_read(gt_path)

                lidar_frames.append(depth)
                img_frames.append(img)
                gt_frames.append(ground_truth)
                i=i+1
            k=k+1
        return  np.asarray(lidar_frames),np.asarray(gt_frames),np.asarray(img_frames), self.index_list

    def get_camera_matrix(self):
        scene = self.img_path[self.index_list[0]].split('/')[5].split('_')[:3]
        scene_date = scene[0]
        camera_id = self.img_path[self.index_list[0]].split('/')[-3]
        for i in range(1,len(scene)) : 
            scene_date += '_' + scene[i]
        file_dir = '../depth_selection/KITTI/calib/'+scene_date+'/calib_cam_to_cam.txt'
        f_dir = os.path.join(file_dir)
        f = open(f_dir)
        lines = f.readlines()
        if camera_id=='image_02':
            K = np.array(lines[25].split(' ')[1:], dtype=np.float32).reshape(3,4)
            K = K[:3,:3]
            return K
        elif camera_id=='image_03':
            K = np.array(lines[33].split(' ')[1:], dtype=np.float32).reshape(3,4)
            K = K[:3,:3]
            return K

        
def read_one_val(index,line_number=64,with_semantic=True,if_removal=False):
    ground_truth_path='../depth_selection/val_selection_cropped/groundtruth_depth'
    if line_number==64:
        velodyne_raw_path='../depth_selection/val_selection_cropped/velodyne_raw'
    if line_number==32:
        velodyne_raw_path='../depth_selection/val_selection_cropped/velodyne_raw_32'
    if line_number==16:
        velodyne_raw_path='../depth_selection/val_selection_cropped/velodyne_raw_16'

    image_path='../depth_selection/val_selection_cropped/image'
    ground_truth=os.listdir('../depth_selection/val_selection_cropped/groundtruth_depth')
    image=os.listdir('../depth_selection/val_selection_cropped/image')
    velodyne_raw=os.listdir('../depth_selection/val_selection_cropped/velodyne_raw')
    intrinsics=os.listdir('../depth_selection/val_selection_cropped/intrinsics')
    k=0

    while (k<self.frames):
        i=image[index]
        img_one=[]
        lidar_one=[]
        ground_thuth_one=[]

        img_file = Image.open(image_path+'/'+i)
        # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
        rgb_png = np.array(img_file, dtype='uint8') # in the range [0,255]
        img_file.close()
        img=rgb_png

        img_file = Image.open(velodyne_raw_path+  '/'+i[:27]+'velodyne_raw'+i[32:])
        depth_png = np.array(img_file, dtype=int)
        img_file.close()
        depth = depth_png.astype(np.float32) / 256.
        # depth[depth_png == 0] = -1.
        depth = np.expand_dims(depth,-1)


        img_file = Image.open(ground_truth_path+  '/'+i[:27]+'groundtruth_depth'+i[32:])
        ground_truth = np.array(img_file, dtype=int)
        img_file.close()
        ground_truth = ground_truth.astype(np.float32) / 256.

        img_one.append(img)
        lidar_one.append(depth[:,:,0])
        ground_thuth_one.append(ground_truth)
        if with_semantic:
            semantic = Image.open('../depth_selection/val_selection_cropped/imagesegmented_images'+'/'+i)
            rgb_png = np.array(semantic, dtype='uint8') # in the range [0,255]
            img_file.close()
            img=rgb_png
        k = k + 1
        index = index + 1
    return  np.asarray(img_one),np.asarray(lidar_one),np.asarray(ground_thuth_one), img


def read_one_test(index):
    ground_truth_path='../depth_selection/test_depth_completion_anonymous/groundtruth_depth'
    velodyne_raw_path='../depth_selection/test_depth_completion_anonymous/velodyne_raw'
    image_path='../depth_selection/test_depth_completion_anonymous/image'
    image=os.listdir('../depth_selection/test_depth_completion_anonymous/image')
    velodyne_raw=os.listdir('../depth_selection/test_depth_completion_anonymous/velodyne_raw')
    intrinsics=os.listdir('../depth_selection/test_depth_completion_anonymous/intrinsics')
    k = 0

    while (k<self.frames):
	    i=image[index]
	    img_one=[]
	    lidar_one=[]
	    ground_thuth_one=[]
	    
	    img_file = Image.open(image_path+'/'+i)
	    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
	    rgb_png = np.array(img_file, dtype='uint8') # in the range [0,255]
	    img_file.close()
	    img=rgb_png
	    
	    img_file = Image.open(velodyne_raw_path+  '/'+i[:27])
	    depth_png = np.array(img_file, dtype=int)
	    img_file.close()
	    depth = depth_png.astype(np.float32) / 256.
	    # depth[depth_png == 0] = -1.
	    depth = np.expand_dims(depth,-1)
	    	    
	    img_one.append(img)
	    lidar_one.append(depth[:,:,0])

	    k += 1
	    index += 1

    return  np.asarray(img_one).astype(np.float32),np.asarray(lidar_one).astype(np.float32),image[index]




