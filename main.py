#!/usr/bin/env python
import torch

from model import DepthCompletionNet
from data_read import *
import helper


def main(): 
	num_frames = 1
	batch_size = 8
	num_epochs = 2
	learning_rate = 1e-5
	output_directory = "./checkpoints/64_line_sparseToDense/"
	# load_file = 'checkpoint-0.pth.tar'
	data_loader = Data_load(input_line=64, frames = num_frames)
	

	model = DepthCompletionNet()
	criterion = helper.MaskedMSELoss()
	optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
	
	cuda = torch.cuda.is_available()
	if cuda : 
		import torch.backends.cudnn as cudnn
		cudnn.benchmark = True
		print("gpu")
		device = torch.cuda.device("cuda")
	else : 
		print("cpu")
		device = torch.cuda.device("cpu")
	print("Using '{}' for computation ".format(device))
	# To load model
	# checkpoint = torch.load(output_directory+load_file)
	# model.load_state_dict(checkpoint['model_state_dict'])
	# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	# epoch = checkpoint['epoch']
	# print("epoch : "+str(epoch))

	data = {}

	for epoch in range(num_epochs) :
		# Load data
		lidar_frames, gt_frames, img_frames, index_list = data_loader.read_frames(batch_size = batch_size, if_removal=False, index = None)

		# Reshaping to (N,C,H,W) from (N,H,W,C)
		lidar_frames = np.reshape(lidar_frames,(lidar_frames.shape[0],lidar_frames.shape[3],lidar_frames.shape[1],lidar_frames.shape[2]))
		img_frames = np.reshape(img_frames,(img_frames.shape[0],img_frames.shape[3],img_frames.shape[1],img_frames.shape[2]))
		gt_frames = np.reshape(gt_frames,(gt_frames.shape[0],gt_frames.shape[3],gt_frames.shape[1],gt_frames.shape[2]))
		print(lidar_frames.shape)	
		print(img_frames.shape)	
		print(gt_frames.shape)	

		# lidar_frames, gt_frames, img_frames = torch.from_numpy(lidar_frames).to(device=device), torch.from_numpy(gt_frames).to(device=device), torch.from_numpy(img_frames).to(device=device)
		lidar_frames, gt_frames, img_frames = torch.from_numpy(lidar_frames), torch.from_numpy(gt_frames), torch.from_numpy(img_frames)
		data['d'] = lidar_frames
		data['rgb'] = img_frames

		depth_pred = model(data)
		loss = criterion(depth_pred, gt_frames)

		print("loss : ",loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		helper.save_checkpoint({
			'epoch' : epoch,
			'model' : model,
			'optimizer' : optimizer,
			}, epoch, output_directory)


if __name__=='__main__' : 
	main()

