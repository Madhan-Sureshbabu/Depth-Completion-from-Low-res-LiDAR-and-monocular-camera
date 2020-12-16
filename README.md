# Depth Completion for Low-res Lidar

## Project Pipeline

Markup :* Find relative pose between different frames using Perspective-n-Point algorithm
<img src="media/PnP.png" width="700" height="400" />
	* Use the above result to transform 3D points to the last co-ordinate frame and project it back to obtain the concatenated depth image (for benchmarking results)
Concatenated Depth image : 
<img src="media/concDepth.png" width="700" height="400" />
Ground Truth Depth image : 
<img src="media/gt.png" width="700" height="400" />
	* Form superpixels using Simple Linear Iterative Clustering from pairs of RGB images and match them by comparing the mean depth value of the superpixels
Superpixels :
<img src="media/slic.png" width="700" height="400" />
Superpixel matche across 2 frames : 
<img src="media/matched_segments.png" width="700" height="400" />
	* Reject lidar depth points from superpixel matches whose depth difference is larger than a threshold parameter, and concatenate remaining depth points
Concatenated Depth image after Outlier rejection : 
<img src="media/concSLIC.png" width="700" height="400" />

