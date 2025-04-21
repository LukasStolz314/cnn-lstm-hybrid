import cv2
import mediapipe as mp
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from settings import processed_img_dir

class LipReadingDataset(Dataset):
    def __init__(self, video_paths, labels, max_frames=30, mean=None, std=None):
        self.video_paths = video_paths
        self.labels = labels
        self.max_frames = max_frames
        self.normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx].split("\\")[-1]
        label = self.labels[idx]

        frames = np.load(f"{processed_img_dir}/{video_path}.npy") 

        video_tensor = torch.tensor(np.array(frames)).permute(3, 0, 1, 2).float() / 255.0

        # print(video_tensor.shape)
        for t in range(video_tensor.shape[1]):
            video_tensor[:, t] = self.normalize(video_tensor[:, t])

        label_tensor = torch.tensor(label).long()
        return video_tensor, label_tensor
