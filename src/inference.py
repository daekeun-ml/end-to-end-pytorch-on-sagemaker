
from __future__ import absolute_import

import argparse
import json
import logging
import os
import sys
import time
import random
from os.path import join
import numpy as np
import io
import tarfile
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import copy
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
from torch import topk

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JSON_CONTENT_TYPE = 'application/json'

# Loads the model into memory from storage and return the model.
def model_fn(model_dir):
    logger.info("==> model_dir : {}".format(model_dir))
    model = models.resnet18(pretrained=True)
    last_hidden_units = model.fc.in_features
    model.fc = torch.nn.Linear(last_hidden_units, 186)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
    return model

# Deserialize the request body
def input_fn(request_body, request_content_type='application/x-image'):
    print('An input_fn that loads a image tensor')
    print(request_content_type)
    if request_content_type == 'application/x-image':             
        img = np.array(Image.open(io.BytesIO(request_body)))
    elif request_content_type == 'application/x-npy':    
        img = np.frombuffer(request_body, dtype='uint8').reshape(137, 236)   
    else:
        raise ValueError(
            'Requested unsupported ContentType in content_type : ' + request_content_type)

    img = 255 - img
    img = img[:,:,np.newaxis]
    img = np.repeat(img, 3, axis=2)    

    test_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    img_tensor = test_transforms(img)

    return img_tensor         
        

# Predicts on the deserialized object with the model from model_fn()
def predict_fn(input_data, model):
    logger.info('Entering the predict_fn function')
    start_time = time.time()
    input_data = input_data.unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    input_data = input_data.to(device)
                          
    result = {}
                                                 
    with torch.no_grad():
        logits = model(input_data)
        pred_probs = F.softmax(logits, dim=1).data.squeeze()   
        outputs = topk(pred_probs, 5)                  
        result['score'] = outputs[0].detach().cpu().numpy()
        result['class'] = outputs[1].detach().cpu().numpy()
    
    print("--- Elapsed time: %s secs ---" % (time.time() - start_time))    
    return result        

# Serialize the prediction result into the response content type
def output_fn(pred_output, accept=JSON_CONTENT_TYPE):
    return json.dumps({'score': pred_output['score'].tolist(), 
                       'class': pred_output['class'].tolist()}), accept
