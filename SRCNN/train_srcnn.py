import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from srcnn_model import SRCNN
from L1_minimization_recovering_code.main import random_destroy_to_image

class CIFAR10MaskedDataset(Dataset):
    def __init__(self, batch_dir, mask_percentage=0.3):
        self.images = []
        for i in range(1, 6):
            with open(os.path.join(batch_dir, f'data_batch_{i}'), 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                data = batch[b'data']
                imgs = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
                imgs_gray = np.mean(imgs, axis=1, keepdims=True)  # (N,1,32,32)
                self.images.append(imgs_gray)
        self.images = np.concatenate(self.images, axis=0)
        self.mask_percentage = mask_percentage

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].squeeze()  # (32,32)
        masked_img, _ = random_destroy_to_image(img * 255, missing_value=0, percentage=self.mask_percentage)
        masked_img = masked_img / 255.0
        return torch.tensor(masked_img).unsqueeze(0), torch.tensor(img).unsqueeze(0)

if __name__ == "__main__":
    dataset = CIFAR10MaskedDataset(batch_dir="cifar-10-batches-py", mask_percentage=0.3)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(30):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.6f}")

    torch.save(model.state_dict(), "srcnn_cifar_inpainting.pth")
    print("Model saved!")