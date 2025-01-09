import torch
import torch.nn as nn
import albumentations as aug
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as mat 
import numpy
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mapdataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self,index):
        image_path = self.list_files[index]
        image = os.path.join(self.root_dir,image_path)
        image = numpy.array(Image.open(image))

        input_image = image[:,:600,:]
        target_image = image[:,600:,:]
        
        augmented_images = both_transform(image = input_image, image0 = target_image)
        input_image = augmented_images["image"]
        target_image = augmented_images["image0"]

        input_image = transform_only_input(image = input_image)["image"]
        target_image = transform_only_target(image = target_image)["image"]

        return (input_image,target_image)
    
both_transform = aug.Compose(
    [aug.Resize(width = 256, height = 256), aug.HorizontalFlip(p = 0.5)], additional_targets = {"image0" : "image"}
)

transform_only_input = aug.Compose(
    [aug.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5], max_pixel_value = 255.0), ToTensorV2()],
)

transform_only_target = aug.Compose(
    [aug.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5], max_pixel_value = 255.0), ToTensorV2()],
)

def save_some_examples(gen,val_loader,epoch):
    image,target = next(iter(val_loader))
    image = image.to(device)
    gen.eval()
    with torch.no_grad():
        generated_image = gen(image)
        generated_image = generated_image * 0.5 + 0.5
        save_image(generated_image, f"generations/epoch {epoch}.png" )
    gen.train()