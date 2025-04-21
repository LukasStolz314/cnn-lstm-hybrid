import torch
import numpy as np
from torch.utils.data import DataLoader
from lip_reading_dataset import LipReadingDataset

def compute_dataset_mean_std(train_paths, y_train, file):
        temp_dataset = LipReadingDataset(train_paths, y_train)
        loader = DataLoader(temp_dataset, batch_size=1, shuffle=False, num_workers=4)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        total = 0

        for i, (video_tensor, _) in enumerate(loader):
            video_tensor = video_tensor.squeeze(0)
            frames = video_tensor.view(3, -1)
            mean += frames.mean(dim=1)
            std += frames.std(dim=1)
            total += 1
            print(i)

        mean /= total
        std /= total
        
        mean = mean.numpy()
        std = std.numpy()
        np.save(file, {'mean': mean, 'std': std})

def verify_normalization(video_tensor):
        b, c, t, h, w = video_tensor.shape

        flattened = video_tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, -1)

        means = flattened.mean(dim=[0, 2])
        stds = flattened.std(dim=[0, 2])

        print("Channel-wise Mean:", means)
        print("Channel-wise Std:", stds)