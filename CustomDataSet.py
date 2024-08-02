import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None): #Run onve when instantiating the Dataset object
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self): #Returns number of samples in our dataset
        return len(self.img_labels)

    def __getitem__(self, idx): #Loads and returns a sample from the dataset at the given index idx
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) #Identifies the image's location on fisk
        image = read_image(img_path) #Converts to a tensor
        label = self.img_labels.iloc[idx, 1] #Retrieves corresponding label from the csv data in self.img_labels
        if self.transform: #Calls transform functions on them
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label #Returns tensor image and corresponding label in a tuple