import numpy as np
import os
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lip_reading_dataset import LipReadingDataset
from normalization import *
from settings import *
from cnn import Conv3DModel

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import logging
logging.getLogger('mediapipe').setLevel(logging.FATAL)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')


if __name__ == "__main__":    
    video_paths = []
    labels = []

    # for label in os.listdir(base_dir):
    for label in corpus:
        label_path = os.path.join(base_dir, label)
        # print(label_path)
        if os.path.isdir(label_path):
            for video_file in os.listdir(os.path.join(label_path, 'train')):
                # print(video_file)
                if video_file.lower().endswith('.mp4'):
                    video_path = os.path.join(label_path, 'train', video_file)
                    video_paths.append(video_path)
                    labels.append(label)

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    train_paths, test_paths, y_train, y_test = train_test_split(video_paths, encoded_labels, test_size=0.2, random_state=42)

    model = Conv3DModel()
    optimizer = optim.Adam(model.parameters(), weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # compute_dataset_mean_std(train_paths, y_train, mean_std_file)

    # Load from checkpoint if it exists
    start_epoch = 0
    checkpoint_path = 'checkpoint_last_epoch.pth'

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    stats = np.load(mean_std_file, allow_pickle=True).item()
    mean = stats['mean']
    std = stats['std']
    
    batch_size = 16
    train_dataset = LipReadingDataset(train_paths, y_train, mean=mean, std=std)
    test_dataset = LipReadingDataset(test_paths, y_test, mean=mean, std=std)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=2)

    print(f'Train len: {len(train_dataset)}')    

    def train():
        epochs = 30
        epoch_logs = []
        for epoch in range(start_epoch, epochs):
            start = time.time()
            model.train()
            epoch_loss = 0
            print('Start')
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                start_batch = time.time()
                # verify_normalization(batch_x)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                epoch_loss += loss.item()

                end_batch = time.time()
                diff = end_batch - start_batch
                print(f'Batch {batch_idx} - {diff}')
            
            loss = epoch_loss/len(train_loader)
            end = time.time()
            elapsed = end - start 
            acc = evaluate()
            print(f"Epoch: {loss} - {acc} - {elapsed}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, 'checkpoint_last_epoch.pth')  

            epoch_logs.append({
                "epoch": epoch + 1,
                "loss": loss,
                "acc": acc,
                "time_seconds": round(elapsed, 2)
            })

            with open("training_log.json", "w") as f:
                json.dump(epoch_logs, f, indent=4)          

    def evaluate():
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1)

                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        print(f"Test Accuracy: {acc:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=encoder.classes_))

        return acc

    train()
    torch.save(model.state_dict(), 'model_weights.pth')
    acc = evaluate()
