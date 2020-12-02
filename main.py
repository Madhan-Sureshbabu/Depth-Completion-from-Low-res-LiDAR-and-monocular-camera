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
	load_file = 'checkpoint-0.pth.tar'
	# data_loader = Data_load(input_line=64, frames = num_frames)
	

	model = DepthCompletionNet()
	criterion = helper.MaskedMSELoss()
	optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
	
	# To load model
	# checkpoint = torch.load(output_directory+load_file)
	# model.load_state_dict(checkpoint['model_state_dict'])
	# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	# epoch = checkpoint['epoch']
	# print("epoch : "+str(epoch))

	data = {}

	# for epoch in range(num_epochs) :
	# 	# Load data
	# 	lidar_frames, gt_frames, img_frames, index_list = data_loader.read_frames(batch_size = batch_size, if_removal=False, index = None)
	# 	print(lidar_frames.shape)	

	# 	lidar_frames, gt_frames, img_frames = torch.from_numpy(lidar_frames), torch.from_numpy(gt_frames), torch.from_numpy(img_frames)
	# 	data['d'] = lidar_frames
	# 	data['rgb'] = img_frames

	# 	depth_pred = model(data)
	# 	loss = criterion(depth_pred, gt_frames)

	# 	print("loss : ",loss.item())

	# 	optimizer.zero_grad()
	# 	loss.backward()
	# 	optimizer.step()

	# 	helper.save_checkpoint({
	# 		'epoch' : epoch,
	# 		'model' : model,
	# 		'optimizer' : optimizer,
	# 		}, epoch, output_directory)


if __name__=='__main__' : 
	main()

