# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 00:57:47 2021

@author: Karthik
"""

from utils import save_checkpoint, save_predictions_as_imgs, check_accuracy,  load_checkpoint
from dataset import custom_dataset
from model import UNET
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_WORKERS = 4
IMAGE_HEIGHT = 101  
IMAGE_WIDTH = 101
PIN_MEMORY = True
LOAD_MODEL = False

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
def main():

    train_transform = transforms.Compose(
        [ transforms.ToTensor(),
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            #A.Rotate(limit=35, p=1.0),
            #A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.1),
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                #max_pixel_value=255.0,
            ),
            
        ],
    )

    val_transforms = transforms.Compose(
        [ transforms.ToTensor(),
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                #max_pixel_value=255.0,
            ),
        ],
    )    

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
 


    data = custom_dataset(img_dir ='E:/segmentation/tgs-salt-identification-challenge/train/images', 
                          mask_dir='E:/segmentation/tgs-salt-identification-challenge/train/masks', 
                          transform = train_transform )

    TEST_SIZE = 0.1
    BATCH_SIZE = 64
    SEED = 42

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, test_indices = train_test_split(
        range(len(data)),
        #data.targets,
        #stratify=data.targets,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    # generate subset based on indices
    train_split = Subset(data, train_indices)
    test_split = Subset(data, test_indices)

    # create batches
    train_loader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True,num_workers = NUM_WORKERS)
    val_loader = DataLoader(test_split, batch_size=BATCH_SIZE, shuffle=True,num_workers = NUM_WORKERS)    

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    #check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        '''
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
        '''


if __name__ == "__main__":
    main()
