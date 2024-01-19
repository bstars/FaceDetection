import numpy as np
import torch
from torch import nn
import cv2


from config import Params
from layers import YoloConvBlockNaive, YoloDetectionBlockNaive, FeatureFusionNaive, ConvNormRelu, UpSampling
from DetectionNetAbstract import DetectionNet
from Detector import nonmax_suppression, plot_prediction


class InterpretModel(DetectionNet):
	def __init__(self, backbone_pretrained, backbone_freeze, device='cpu'):
		super(InterpretModel, self).__init__(backbone_pretrained, backbone_freeze, device)

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

		return (low_level_small_detect, mid_level_small_detect, high_level_small_detect), \
		       (low_level_mid_detect, mid_level_mid_detect, high_level_mid_detect), \
		       (low_level_large_detect, mid_level_large_detect, high_level_large_detect)

class Interpretor2():
	def __init__(self, ckpt_path, device='cpu'):
		self.ckpt_path = ckpt_path
		self.net = InterpretModel(True, True, device)
		self.device = device
		self.img_size = Params.IMG_SIZE
		ckpt = torch.load(self.ckpt_path, map_location=torch.device(device))
		self.net.load_state_dict(ckpt['model_state_dict'], strict=False)

	def filter_predictions(self, pred_large, pred_mid, pred_small):
		"""
		:param pred_large: np.array, of shape [5,14,14]
		:param pred_mid: np.array, of shape [5,28,28]
		:param pred_small: np.array, of shape [5,56,56]
		:return:
			np.array, of shape [n,5]
		"""
		preds = []
		for pred in [pred_large, pred_mid, pred_small]:
			confidence = pred[0, :, :]
			Is, Js = np.where(confidence >= Params.CONFIDENCE_THRESHOLD)
			for i in range(len(Is)):
				preds.append(
					pred[:, Is[i], Js[i]]
				)
		if len(preds) == 0:
			return []
		preds = np.array(preds)
		preds = nonmax_suppression(preds)
		return preds

	def interpret_one(self, fname):
		img = cv2.imread(fname)
		img = cv2.resize(img, (self.img_size, self.img_size))
		img = img[:, :, [2, 1, 0]] / 255
		img_tensor = img.transpose([2,0,1]).astype(np.float32)
		img_tensor = torch.from_numpy(img_tensor).float().to(self.device)[None, :,:,:]

		small_detect = torch.zeros(1, 5, Params.LARGE_SIZE, Params.LARGE_SIZE)
		mid_detect = torch.zeros(1, 5, Params.MEDIUM_SIZE, Params.MEDIUM_SIZE)
		large_detect = torch.zeros(1, 5, Params.SMALL_SIZE, Params.SMALL_SIZE)

		(low_level_small_detect, mid_level_small_detect, high_level_small_detect), \
		(low_level_mid_detect, mid_level_mid_detect, high_level_mid_detect), \
		(low_level_large_detect, mid_level_large_detect, high_level_large_detect) = self.net(img_tensor)

		print(low_level_small_detect.shape, mid_level_small_detect.shape, high_level_small_detect.shape)


		# bounding box only from high-level prediction
		small_detect[0,:,:,:] += high_level_small_detect[0,:,:,:].cpu().detach().numpy()
		mid_detect[0,:,:,:] += high_level_mid_detect[0,:,:,:].cpu().detach().numpy()
		large_detect[0,:, :, :] += high_level_large_detect[0, :, :, :].cpu().detach().numpy()

		pred_large, pred_mid, pred_small = self.net.interpret_outputs(large_detect, mid_detect, small_detect)
		preds = self.filter_predictions(pred_large.cpu().detach().numpy()[0],
		                                pred_mid.cpu().detach().numpy()[0],
		                                pred_small.cpu().detach().numpy()[0])
		plot_prediction(img, preds)

		# bounding box  large+mid level prediction
		small_detect[0, :, :, :] += mid_level_small_detect[0, :, :, :].cpu().detach().numpy()
		mid_detect[0, :, :, :] += mid_level_mid_detect[0, :, :, :].cpu().detach().numpy()
		large_detect[0, :, :, :] += mid_level_large_detect[0, :, :, :].cpu().detach().numpy()

		pred_large, pred_mid, pred_small = self.net.interpret_outputs(large_detect, mid_detect, small_detect)
		preds = self.filter_predictions(pred_large.cpu().detach().numpy()[0],
		                                pred_mid.cpu().detach().numpy()[0],
		                                pred_small.cpu().detach().numpy()[0])
		plot_prediction(img, preds)

		# bounding box only from large+mid+small level prediction
		small_detect[0, :, :, :] += low_level_small_detect[0, :, :, :].cpu().detach().numpy()
		mid_detect[0, :, :, :] += low_level_mid_detect[0, :, :, :].cpu().detach().numpy()
		large_detect[0, :, :, :] += low_level_large_detect[0, :, :, :].cpu().detach().numpy()

		pred_large, pred_mid, pred_small = self.net.interpret_outputs(large_detect, mid_detect, small_detect)
		preds = self.filter_predictions(pred_large.cpu().detach().numpy()[0],
		                                pred_mid.cpu().detach().numpy()[0],
		                                pred_small.cpu().detach().numpy()[0])
		plot_prediction(img, preds)

if __name__ == '__main__':
	interp = Interpretor2(Params.MULTIHEAD_CKPT_PATH)
	fname = './img_tests/elon.jpg'
	interp.interpret_one(fname)
