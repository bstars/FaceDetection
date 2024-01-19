import torch
from torch import nn
import numpy as np
import abc

from torchvision.models import resnet34
from torchvision import transforms

from config import Params
from layers import YoloConvBlockNaive, YoloDetectionBlockNaive, FeatureFusionNaive, ConvNormRelu, UpSampling
from DetectionNetAbstract import DetectionNet


class DetectionNetMultiHead(DetectionNet):
	def __init__(self, backbone_pretrained, backbone_freeze, device='cpu'):
		super(DetectionNetMultiHead, self).__init__(backbone_pretrained, backbone_freeze, device)

		# low level feature
		self.low_level_small_detect = nn.Sequential(
			ConvNormRelu(64, 128, 3, 1, 1), # [n,128,112,112]
			ConvNormRelu(128, 128, 3, 1, 1),# [n,128,112,112]
			ConvNormRelu(128, 128, 2, 2, 0),# [n,128,56,56]
			ConvNormRelu(128, 128, 3, 1, 1),# [n,128,56,56]
			ConvNormRelu(128, 128, 3, 1, 1),# [n,128,56,56]
			YoloDetectionBlockNaive(128)# [n,5,56,56]
		)
		self.low_level_mid_detect = nn.Sequential(
			ConvNormRelu(64, 128, 3, 1, 1), # [n,128,112,112]
			ConvNormRelu(128, 128, 2, 2, 0),  # [n,128,56,56]
			ConvNormRelu(128, 128, 3, 1, 1),  # [n,128,56,56]
			ConvNormRelu(128, 128, 2, 2, 0),  # [n,128,28,28]
			ConvNormRelu(128, 128, 3, 1, 1),  # [n,128,28,28]
			YoloDetectionBlockNaive(128)  # [n,5,56,56]
		)
		self.low_level_large_detect = nn.Sequential(
			ConvNormRelu(64, 128, 3, 1, 1),  # [n,128,112,112]
			ConvNormRelu(128, 128, 4, 4, 0),  # [n,128,28,28]
			ConvNormRelu(128, 128, 3, 1, 1), # [n,128,28,28]
			ConvNormRelu(128, 128, 2, 2, 0), # [n,128,14,14]
			ConvNormRelu(128, 128, 3, 1, 1), # [n,128,14,14]
			YoloDetectionBlockNaive(128) # [n,5,14,14]
		)

		# mid level feature
		self.mid_level_small_detect = nn.Sequential(
			ConvNormRelu(128, 256, 3, 1, 1), # [n,256,56,56]
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,56,56]
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,56,56]
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,56,56]
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,56,56]
			YoloDetectionBlockNaive(256) # [n,5,56,56]
		)

		self.mid_level_mid_detect = nn.Sequential(
			ConvNormRelu(128, 256, 3, 1, 1),# [n,256,56,56]
			ConvNormRelu(256, 256, 3, 1, 1),# [n,256,56,56]
			ConvNormRelu(256, 256, 2, 2, 0),# [n,256,28,28]
			ConvNormRelu(256, 256, 3, 1, 1),# [n,256,28,28]
			ConvNormRelu(256, 256, 3, 1, 1),# [n,256,28,28]
			YoloDetectionBlockNaive(256)# [n,5,28,28]
		)

		self.mid_level_large_detect = nn.Sequential(
			ConvNormRelu(128, 256, 3, 1, 1),  # [n,256,56,56]
			ConvNormRelu(256, 256, 2, 2, 0),  # [n,256,28,28]
			ConvNormRelu(256, 256, 3, 1, 1),  # [n,256,28,28]
			ConvNormRelu(256, 256, 2, 2, 0),  # [n,256,14,14]
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,14,14]
			YoloDetectionBlockNaive(256) # [n,5,14,14]
		)

		# high level features
		self.high_level_small_detect = nn.Sequential(
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,28,28]
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,28,28]
			UpSampling(256, 256, 2, 2, 0), # [n,256,56,56]
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,56,56]
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,56,56]
			YoloDetectionBlockNaive(256) # [n,5,56,56]
		)

		self.high_level_mid_detect = nn.Sequential(
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,28,28]
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,28,28]
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,28,28]
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,28,28]
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,28,28]
			YoloDetectionBlockNaive(256)
		)

		self.high_level_large_detect = nn.Sequential(
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,28,28]
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,28,28]
			ConvNormRelu(256, 256, 2, 2, 0), # [n,256,14,14]
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,14,14]
			ConvNormRelu(256, 256, 3, 1, 1), # [n,256,14,14]
			YoloDetectionBlockNaive(256)
		)

		self.to(device)
		self.device = device

	def forward(self, x):
		x = self.transforms(x)
		r1, r2, r3 = self.backbone(x)
		# r1 = torch.randn(1, 64, 112, 112)
		# r2 = torch.randn(1, 128, 56, 56)
		# r3 = torch.randn(1, 256, 28, 28)

		# low level feature
		low_level_small_detect = self.low_level_small_detect(r1)
		low_level_mid_detect = self.low_level_mid_detect(r1)
		low_level_large_detect = self.low_level_large_detect(r1)

		# mid level feature
		mid_level_small_detect = self.mid_level_small_detect(r2)
		mid_level_mid_detect = self.mid_level_mid_detect(r2)
		mid_level_large_detect = self.mid_level_large_detect(r2)

		# high level feature
		high_level_small_detect = self.high_level_small_detect(r3)
		high_level_mid_detect = self.high_level_mid_detect(r3)
		high_level_large_detect = self.high_level_large_detect(r3)

		large_detect = low_level_large_detect + mid_level_large_detect + high_level_large_detect
		mid_detect = low_level_mid_detect + mid_level_mid_detect + high_level_mid_detect
		small_detect = low_level_small_detect + mid_level_small_detect + high_level_small_detect

		return large_detect, mid_detect, small_detect


if __name__ == '__main__':
	net = DetectionNetMultiHead(True, True, 'cpu')
	# x = torch.randn(1, 3, 448, 448)
	net(None)
