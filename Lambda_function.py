#!/usr/bin/env python
# coding: utf-8

import json
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoConfig
import numpy as np
import pandas as pd
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import re
import io
import boto3

#model_name = 'monologg/koelectra-small-v3-discriminator'
#tokenizer = AutoTokenizer.from_pretrained(model_name)

max_len = 128          
batch_size = 1

'''
BUCKET_NAME = 'mbti-predict-s3'  
OBJECT_NAME = ['mbti_model_koe.pt','config.json','special_tokens_map.json','tf_model.h5','tokenizer_config.json','vocab.txt']
PATH_NAME = '/tmp/' 
PATH_t_NAME = '/torch/' 

s3 = boto3.client('s3')
for obj in OBJECT_NAME:
    s3.download_file(BUCKET_NAME, PATH_t_NAME+obj, PATH_NAME+obj)
'''

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_name = 'monologg/koelectra-small-v3-discriminator'
tokenizer = AutoTokenizer.from_pretrained('/tmp', local_files_only=True, cache_dir='/tmp')
model = AutoModelForSequenceClassification.from_pretrained('/tmp/pytorch.pt', num_labels=16)

model.to(device)
model.load_state_dict(torch.load("/tmp/mbti_model_koe.pt"))
model.eval()



def api_predict(sentence):
    
    
    cat_dict = {'0':'enfj','1':'enfp','2':'entj','3':'entp','4':'esfj','5':'esfp','6':'estj','7':'estp',
             '8':'infj','9':'infp','10':'intj','11':'intp','12':'isfj','13':'isfp','14':'istj','15':'istp'}
    result_dict = {}
    global mod
    sentence = re.sub("[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…]+", "", sentence)
    sentence = re.sub("\\n+", " ", sentence)
    sentence = re.sub("\\t+", " ", sentence)
    
    
    
    data_x = sentence_convert_data(sentence)   
    setattr(mod, 'chung', model.predict(data_x, batch_size=1))
    preds = str((chung).item())
    intChung = []
    
    for key, value in cat_dict.items():
        appendInt = int(np.round(chung[0,int(key)]*100))
        intChung.append(appendInt)
        result_dict.update({value:appendInt})
    
    x = np.arange(16)
    predi = cat_dict[preds]
    result_dict.update({'mbti':predi})

    return result_dict     





def handler(event, context):
  text = str(event.get('text'))
  result = api_predict(text)
  return {
    "statusCode": 200,
    "headers": {
      "Content-Type": "application/json"
      },
    "body": json.dumps(result)
    }
