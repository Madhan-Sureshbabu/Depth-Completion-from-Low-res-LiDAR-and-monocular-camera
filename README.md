# Depth Completion for Low-res Lidar

This project aims to perform depth completion using depth images from a 16-line / 32-line Lidar and rgb images from a monocular camera. <br />
The sparse depth images have been created by downsampling 64-line lidar depth images obtained from [KITTI Depth completion dataset](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion). 

## Project Pipeline

1. Find relative pose between consecutive frames using Perspective-n-Point algorithm  
<img src="media/PnP.png" width="400" height="300" />   
2. Use the above result to transform 3D points (calculated from the depth images) to the last co-ordinate frame and project it to the image frame to obtain the concatenated depth image (for benchmarking results) <br />
Sparse input Depth image <br />
<img src="media/sparse.png" width="400" height="150" />  
Concatenated Depth image from 5 consecutive frames:  
<img src="media/concDepth.png" width="400" height="150" />  
3. Form superpixels using Simple Linear Iterative Clustering in the RGB images and match them across consecutive frames using the distance from their cluster centres (weighted sum of distance in CIELAB color space and euclidean distance in pixels) <br />
Superpixels :  
<img src="media/slic.png" width="400" height="150" />  
Matched superpixels example :  
<img src="media/matched_segments.png" width="400" height="300" />   
4. Reject lidar depth points from superpixel matches whose ```mean depth``` difference is larger than a threshold, and concatenate remaining depth points. Refer Depth_completion.ipynb for quantitative results. <br />
Concatenated Depth image after Outlier rejection :  
<img src="media/concSLIC.png" width="400" height="150" />
5. The above concatenated depth image contains higher number of data points with better RMSE than the sparse input depth image. This could then be used to perform depth completion in a supervised learning framework to produce better results.

## Dependencies  
```
OpenCV
numpy
scipy
scikit-image
matplotlib 
math
PIL 
system 
date_time
```

### Code structure

1. [Depth Completion.ipynb](./Project/Depth_completion.ipynb) contains the main executable code
2. [data_read_depthCompletion.py](./Project/data_read_depthCompletion.py) contains the Data_load class for the downsampled dataset.
3. [helper_depthCompletion.py](./Project/helper_depthCompletion.py) contains the utility functions for the relative pose estimation and concatenation modules.
4. [slic.py](./Project/slic.py) contains the class definitions of SuperPixels & Cluster classes which are used for superpixel generation and matching.
