import torch
from torchvision import datasets, transforms, models
from torch import nn
import time
import train
import json

checkpoint_name = 'checkpoint_2.pth'
#use it to make sure saved and loaded checkpoints are the same
#TRAINED MODEL IS SAVED IN 'checkpoint_PROPERLY_TRAINED.pth'


def load_checkpoint_and_rebuild_the_model(args):
    #print("Starting importing the checkpoint...")
    #checkpoint = torch.load(args[1]+checkpoint_name, map_location=lambda storage, loc: storage)
    model_loaded = train.create_model(args)
    #print("model_loaded printing", model_loaded)
    model_loaded.load_state_dict(torch.load(args[1]+checkpoint_name, map_location=lambda storage, loc: storage))
    #print("checkpoint loaded")
    
    #f = open('class_to_idx.json')
    #class_to_idx = json.load(f)
    #model_loaded.class_to_idx = class_to_idx
    
    
    #print("class_to_idx from checkpoint loading procedure")
    #print(class_to_idx)
    #model_loaded.class_to_idx(checkpoint['class_to_idx'])
    #model_loaded.load_state_dict(checkpoint['state_dict'])
    #print("Model was loaded successfully")
    return(model_loaded)


def save_the_checkpoint(model, args):
    print ("start saving checkpiont...")
    saving_start = time.time()
    torch.save(model.state_dict(), args[1]+checkpoint_name)
    #print("model.class_to_idx before saving = ")
    #print(model.class_to_idx)
    #checkpoint = {'class_to_idx': model.class_to_idx,
    #          'state_dict': model.state_dict()}
    #torch.save(checkpoint, args[1]+checkpoint_name)
    #print("class_to_idx from checkpoint = ", checkpoint[class_to_idx])
    #print("state_dict from checkpoint = ", checkpoint[state_dict])
    #with open('class_to_idx.json', 'w') as f:
    #    json.dump(model.class_to_idx, f)
    #class_to_idx.json
    saving_end = time.time()
    print(f"Saved to {args[1]}{checkpoint_name}, Duration: {(saving_end - saving_start):.3f}, sec")    
