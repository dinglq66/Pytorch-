import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


if __name__ == '__main__':
	train_data_dir = './images/train'
	train_data_transforms = transforms.Compose([
		transforms.RandomSizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406],
			                 [0.229, 0.224, 0.225])
		])
	train_data = ImageFolder(train_data_dir, transform=train_data_transforms)
	print('the length of train_data.targets:', len(train_data.targets))
	train_data_loader = DataLoader(dataset=train_data,
		batch_size=10,
		shuffle=True,
		num_workers=1,)
	for step, (b_x, b_y) in enumerate(train_data_loader):
		if step > 0:
			break
		print('b_x.shape:{}'.format(b_x.shape))
		print('b_y.shape:{}'.format(b_y.shape))
		print('b_y values:{}'.format(b_y))

