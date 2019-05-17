import os

from torch.utils.data import Dataset, DataLoader
import matplotlib.image as mpimg
import pandas as pd
import cv2 as cv


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])

        image = mpimg.imread(image_name)

        # if image has an alpha color channel, get rid of it
        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        steering_throttle = self.data.iloc[idx, 1:].as_matrix()
        steering_throttle = steering_throttle.astype('float').reshape(-1, 2)
        sample = {'image': gray_image, 'steering_throttle': steering_throttle}

        if self.transform:
            sample = self.transform(sample)

        return sample
