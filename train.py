import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast,GradScaler
from generator import Generator
from discriminator import Discriminator
import dataset
import matplotlib.pyplot as mat

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

torch.backends.cudnn.benchmark = True 

gen = Generator(input_channels = 3).to(device)
disc = Discriminator(input_channels = 3).to(device)
gen_opt = torch.optim.Adam(gen.parameters(),lr = 2e-4, betas = (0.5,0.999))
disc_opt = torch.optim.Adam(disc.parameters(), lr = 2e-4, betas = (0.5,0.999))
BCE = nn.BCEWithLogitsLoss()
L1_loss = nn.L1Loss()
gen_scaler = GradScaler()
disc_scaler = GradScaler()

training_data = dataset.Mapdataset(root_dir = "maps/train")
training_dataloader = DataLoader(training_data, shuffle = False, batch_size = 32)

val_data = dataset.Mapdataset(root_dir = "maps/val")
val_dataloader = DataLoader(val_data, shuffle = False, batch_size = 1)

gen_losses = []
disc_losses = []

for epoch in range(500):
    for batch,data in enumerate(training_dataloader):
        
        #training generator
        image,mask = data

        image = image.to(device)
        mask = mask.to(device)
        
        with autocast():
             fake_image = gen(image)
             disc_score = disc(image,fake_image.detach())
             gen_bce_loss = BCE(disc_score,torch.ones_like(disc_score))
             gen_l1_loss = L1_loss(fake_image,mask) * 100
             gen_loss = gen_bce_loss + gen_l1_loss
             
        gen_opt.zero_grad()
        gen_scaler.scale(gen_loss).backward()
        gen_scaler.step(gen_opt)
        gen_scaler.update()
        
        #training discriminator
        with autocast():
             disc_real_score = disc(image,mask)
             disc_fake_score = disc(image,fake_image.detach())
             disc_real_loss = BCE(disc_real_score,torch.ones_like(disc_real_score))
             disc_fake_loss = BCE(disc_fake_score,torch.zeros_like(disc_real_score))
             disc_loss = (disc_real_loss + disc_fake_loss)/2
             
        disc_opt.zero_grad()
        disc_scaler.scale(disc_loss).backward()
        disc_scaler.step(disc_opt)   
        disc_scaler.update()  
    
    print(f"loss at epoch {epoch} is : \n generator loss : {gen_loss} \
         \n disriminator loss : {disc_loss}\n")
    
    dataset.save_some_examples(gen = gen, val_loader = val_dataloader, epoch = epoch)
    
    if epoch % 5 == 0 or epoch > 480:
        
        gen_losses.append(gen_loss.detach().item())
        disc_losses.append(disc_loss.detach().item())
 
        if epoch > 200:
            torch.save({
                "generator" : gen.state_dict(),
                "gen_optimizer" : gen_opt.state_dict()
                },f"checkpoints/model at epoch {epoch}.pt")
        
print(gen_losses)
print(disc_losses)
        
mat.plot(range(len(gen_losses)),gen_losses,label = "generator loss")
mat.plot(range(len(disc_losses)),disc_losses, label = "discriminator loss")
mat.xlabel("epochs")
mat.ylabel("loss")
mat.legend()
mat.savefig("loss.png")