import torch
import torch.nn as nn
from torchvision.utils import save_image
from generator import Generator
import numpy
from PIL import Image
import albumentations as aug
from albumentations.pytorch import ToTensorV2 
import matplotlib.pyplot as mat

with torch.no_grad():    
    
    model = torch.load("checkpoints/model at epoch 495.pt")
    
    gen = Generator(input_channels = 3)

    gen.load_state_dict(model["generator"],strict = False)
    
    image_path = "maps/val/113.jpg"
    
    image = numpy.array(Image.open(image_path))
    
    image = image[:,:600,:]
    
    transform = aug.Compose(
        [aug.Resize(width = 256, height = 256),aug.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5], max_pixel_value = 255.0), ToTensorV2()]
    )
    
    image = transform(image = image)["image"]
    
    image = image.unsqueeze(0)
    
    image = gen(image)
    image = image * 0.5 + 0.5 

    image = image.squeeze(0)
    
    image = image.permute(1,2,0)
    
    image = image.numpy()

    mat.imshow(image)
    mat.savefig("image.png")
    mat.show()