import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

def init_weights(layer) : 
	if isinstance(layer,nn.Conv2d) or isinstance(layer,nn.Linear) or isinstance(layer,nn.ConvTranspose2d) : 
		layer.weight.data.normal_(0,1e-3)
		if layer.bias is not None :
			layer.bias.data.zero_()
	elif isinstance(layer,nn.BatchNorm2d) :
		layer.weight.data.fill_(1)
		layer.bias.data.zero_() 


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bn=True, relu=True):
	bias = not bn
	layers = []
	layers.append(nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias))

	if bn : 
		layers.append(nn.BatchNorm2d(out_channels))
	if relu : 
		layers.append(nn.LeakyReLU(0.2,inplace=True))

	layers = nn.Sequential(*layers)

	for m in layers.modules():
		init_weights(m)

	return layers

def convt_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bn=True, relu=True):
	bias = not bn
	layers = []
	layers.append(nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias))

	if bn : 
		layers.append(nn.BatchNorm2d(out_channels))
	if relu : 
		layers.append(nn.LeakyReLU(0.2,inplace=True))

	layers = nn.Sequential(*layers)

	for m in layers.modules():
		init_weights(m)

	return layers



class DepthCompletionNet(nn.Module) :
	def __init__(self) :
		super(DepthCompletionNet,self).__init__()
		# self.modality = 'rgbd'

		out_channels = int(64 // 4) # for d in r g b d 
		self.conv1_d = conv_bn_relu(1,out_channels,kernel_size=3,stride=1,padding=1)

		out_channels = int(64 * 3 // 4) # for r g b in r g b d 
		self.conv1_img = conv_bn_relu(3,out_channels,kernel_size=3,stride=1,padding=1)

		model_ = resnet.__dict__['resnet34'](pretrained = False)

		self.conv2 = model_._modules['layer1']
		self.conv3 = model_._modules['layer2']
		self.conv4 = model_._modules['layer3']
		self.conv5 = model_._modules['layer4']

		in_channels = 512 # 512 channels in resnet34
		
		self.conv6 = conv_bn_relu(in_channels, 512, kernel_size=3, stride = 2, padding=1)

		kernel_size = 3
		stride = 2

		self.convt5 = convt_bn_relu(512, 256, kernel_size, stride, padding=1)

		self.convt4 = convt_bn_relu(256+in_channels,128,kernel_size,stride,padding=1)

		self.convt3 = convt_bn_relu(128+256,64,kernel_size,stride,padding=1)

		self.convt2 = convt_bn_relu(64+128,64,kernel_size,stride,padding=1)

		self.convt1 = convt_bn_relu(64+64,64,kernel_size,stride,padding=1)

		self.convtf = convt_bn_relu(64+64,1,kernel_size,stride,padding=1,bn=False,relu=False)

	def forward(self,x) : 
		inp_d = self.conv1_d(x['d'])

		inp_rgb = self.conv1_img(x['rgb'])

		inp = torch.cat((inp_rgb,inp_d),1)

		conv2 = self.conv2(inp)
		conv3 = self.conv3(conv2)
		conv4 = self.conv4(conv3)
		conv5 = self.conv5(conv4)
		conv6 = self.conv6(conv5)

		convt5 = self.convt5(conv6)

		conc = torch.cat((convt5,conv5),1)
		convt4 = self.convt4(conc)

		conc = torch.cat((convt4,conv4),1)
		convt3 = self.convt3(conc)

		conc = torch.cat((convt3,conv3),1)
		convt2 = self.convt2(conc)

		conc = torch.cat((convt2,conv2),1)
		convt1 = self.convt1(conc)

		conc = torch.cat((convt1,inp),1)
		convf = self.convtf(conc)

		if self.training : 
			return 100 * convf

		else : 
			return F.relu(100*convf - self.min_distance) + self.min_distance






