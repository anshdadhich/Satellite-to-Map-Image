# Satellite image to Map Image using Pix2pix

This is a implementation of pix2pix model, this model can also be trained and used for other image transaltion task.

Here are some results:
<br>
![image](https://github.com/user-attachments/assets/2538b599-addd-4a69-b7e9-e5ab401cef50)
![image](https://github.com/user-attachments/assets/7848eecc-9fb8-4d51-bff7-7ffd57576ca3)
![image](https://github.com/user-attachments/assets/644a2f50-39da-4705-b820-acdb58176a74)

the satellite image dataset can be downloaded [here](https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset?select=maps)
<br><br>
pretrained weights can be downloaded [here](https://drive.google.com/drive/folders/1O0hohOW4nZEQNM8hOpSXRl-ajXiXbGJ7)
<br>

## To run the model : 
#### 1. Download the generator.py and inference.py file and the model's weight
#### 2. make sure they are in the same folder
#### 3. replace the image_path in the inference.py with the image you want to use and 
#### 4. make sure that either the image is of dimension 600x600 or 1200x600 (half input and half target like in the dataset)
