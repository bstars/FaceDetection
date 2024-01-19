import torch.optim
import numpy as np
from torch.utils.data import DataLoader
import os

from DetectionNetAbstract import DetectionNet
from DetectionNetMultiHead import DetectionNetMultiHead
from data_util import FDDB, plot_img_with_label, Unlabeled
from config import Params


def train_unsupervised(
		model:DetectionNet,
		labeled_loader: DataLoader,
		unlabeled_loader: DataLoader,
		testing_set:FDDB,
		learning_rate,
		epoches,
		device='cpu',
		checkpoint = None,
		restore_optimizer=False
):
	model.to(device)
	optimizer = torch.optim.Adam(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=learning_rate, weight_decay=1e-4
	)

	if checkpoint is not None:
		model.load_state_dict(checkpoint['model_state_dict'])
		if restore_optimizer:
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	for e in range(epoches):
		for i, (img, label_large, label_mid, label_small) in enumerate(labeled_loader):
			model.train()
			optimizer.zero_grad()

			# Training with labeled dataset
			img = img.to(device)
			label_large = label_large.to(device)
			label_mid = label_mid.to(device)
			label_small = label_small.to(device)

			optimizer.zero_grad()
			output_large, output_mid, output_small = model(img)
			loss_super = model.compute_loss(output_large, output_mid, output_small, label_large, label_mid, label_small)

			# Training with unlabeled dataset
			img_unlabeled = next(iter(unlabeled_loader))
			img_unlabeled = img_unlabeled.to(device)
			img_unlabeled_flipped = torch.flip(img_unlabeled, dims=[3])

			output_large, output_mid, output_small = model(img_unlabeled)
			output_large_flipped, output_mid_flipped, output_small_flipped = model(img_unlabeled_flipped)
			output_large_flipped[:,1,:,:] = 1. - output_large_flipped[:,1,:,:]
			output_mid_flipped[:,1,:,:] = 1. - output_mid_flipped[:,1,:,:]
			output_small_flipped[:,1,:,:] = 1. - output_small_flipped[:,1,:,:]

			output_large_flipped = torch.flip(output_large_flipped, dims=[3])
			output_mid_flipped = torch.flip(output_mid_flipped, dims=[3])
			output_small_flipped = torch.flip(output_small_flipped, dims=[3])


			mask_large = torch.clip(0.5 * output_large[:,0,:,:] + 0.5 * output_large_flipped[:,0,:,:], min=0., max=1)
			mask_large = mask_large.detach()
			mask_mid = torch.clip(0.5 * output_mid[:,0,:,:] + 0.5 * output_mid_flipped[:,0,:,:], min=0, max=1)
			mask_mid = mask_mid.detach()
			mask_small = torch.clip(0.5 * output_small[:,0,:,:] + 0.5 * output_small_flipped[:,0,:,:], min=0, max=1)
			mask_small = mask_small.detach()


			loss_unsuper_large = torch.sum(
					(output_large[:,1:,:,:] - output_large_flipped[:,1:,:,:])**2,
					dim=[1]
				) * mask_large
			loss_unsuper_large = torch.mean( torch.sum(loss_unsuper_large, dim=[1,2]) )

			loss_unsuper_mid = torch.sum(
				(output_mid[:,1:,:,:] - output_mid_flipped[:,1:,:,:]) ** 2,
				dim=[1]
			) * mask_mid
			loss_unsuper_mid = torch.mean( torch.sum(loss_unsuper_mid, dim=[1,2]) )

			loss_unsuper_small = torch.sum(
				(output_small[:,1:,:,:] - output_small_flipped[:,1:,:,:]) ** 2,
				dim=[1]
			) * mask_small
			loss_unsuper_small = torch.mean(torch.sum(loss_unsuper_small, dim=[1,2]) )

			loss_unsuper = loss_unsuper_large + loss_unsuper_mid + loss_unsuper_small
			loss = loss_super + Params.UNSUPER_SCALE * loss_unsuper

			loss.backward()
			optimizer.step()

			print("%d epoch, %d/%d, supervised loss=%.5f, unsupervised loss=%.5f" %
			      (e, i, len(labeled_loader), loss_super.item(), loss_unsuper.item())
			      )

			if i % 100 == 0:

				pred_large, pred_mid, pred_small = model.interpret_outputs(output_large, output_mid, output_small)
				idx = np.random.randint(0, pred_large.shape[0], 1)[0]
				plot_img_with_label(
					img_unlabeled[idx].cpu().detach().numpy().transpose([1, 2, 0]),
					pred_large[idx].cpu().detach().numpy(),
					pred_mid[idx].cpu().detach().numpy(),
					pred_small[idx].cpu().detach().numpy()
				)

				model.eval()
				idx = np.random.randint(0, len(testing_set), 1)[0]
				img_tensor, _, _, _ = testing_set[idx]
				img_tensor = img_tensor.to(device)
				output_large, output_mid, output_small = model(img_tensor[None,...])
				pred_large, pred_mid, pred_small = model.interpret_outputs(output_large, output_mid, output_small)


				plot_img_with_label(
					img_tensor.cpu().detach().numpy().transpose([1, 2, 0]),
					pred_large[0].cpu().detach().numpy(),
					pred_mid[0].cpu().detach().numpy(),
					pred_small[0].cpu().detach().numpy()
				)

		if e % 10 == 0:
			torch.save({
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()},
				os.path.join(Params.CHECKPOINTS_SAVING_PATH, "%d.pth" % (e)))



if __name__ == '__main__':
	labeled_loader = DataLoader(FDDB(test=False), Params.BATCH_SIZE, shuffle=False)
	testing_set = FDDB(test=True)
	unlabeled_loader = DataLoader(Unlabeled(), Params.BATCH_SIZE, shuffle=True)



	checkpoint = torch.load(Params.MULTIHEAD_CKPT_PATH, map_location=torch.device(Params.DEVICE))
	model = DetectionNetMultiHead(True, True, device=Params.DEVICE)
	train_unsupervised(
		model, labeled_loader, unlabeled_loader, testing_set, Params.SEMI_LEARNING_RATE,
		120, device=Params.DEVICE, checkpoint=checkpoint, restore_optimizer=False
	)

	# a = np.array([
	# 	[1,2,3],
	# 	[4,5,6]
	# ])
	# a = torch.from_numpy(a)
	# b = torch.flip(a,dims=[1])
	# print(b.detach().numpy())

