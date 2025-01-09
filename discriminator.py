import torch
import torch.nn as nn 

class cnn_block(nn.Module):
    def __init__(self,input_channels, output_channels, kernel_size, stride ,padding):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding,
                padding_mode = "reflect"),
            nn.InstanceNorm2d(output_channels,affine = True),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self,x):
        x = self.conv(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,input_channels = 3, features = [64,128,256]):
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels*2, features[0], kernel_size = 4, stride = 2, padding = 1, \
                padding_mode = "reflect"),
            nn.LeakyReLU(0.2)
        )
        
        layers = []
        input_channels = features[0]
        
        for feature in features[1:]:
            layers.append(
                cnn_block(input_channels, feature, kernel_size = 4, stride = 1 if \
                    input_channels == features[-1] else 2, padding = 0 if input_channels == features[-1] else 0)
            )
            input_channels = feature            
            
        self.layers = nn.Sequential(*layers)
     
    def forward(self,x,y):
        x = torch.cat([x,y],dim = 1)
        x = self.initial(x)
        x = self.layers(x)
        return x
    
if __name__ == "__main__":
   x = torch.randn(1,3,256,256)
   y = torch.randn(1,3,256,256)
   discriminator = Discriminator()
   output = discriminator(x,y) 
   print(output.size())
   