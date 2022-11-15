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
batch_size = 8
num_epochs = 3
log_interval = 1600    # metrics 생성 시점

BUCKET_NAME = 'mbti-predict-s3'  
OBJECT_NAME = ['mbti_model_koe.pt']
PATH_NAME = '/tmp/' 
#PATH_t_NAME = '/torch/' 

s3 = boto3.client('s3')
for obj in OBJECT_NAME:
    s3.download_file(BUCKET_NAME, obj, PATH_NAME+obj)

device = torch.device('cpu')

model_name = 'monologg/koelectra-small-v3-discriminator'
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="tmp/")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=16, cache_dir="tmp/")
model.to(device)

model.load_state_dict(torch.load("/tmp/mbti_model_koe.pt",  map_location=device))

from torch.utils.data import Dataset, DataLoader

# 데이터셋 클래스
class ReviewDataset(Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels

  def __getitem__(self, idx):
    item = { key: torch.tensor(val[idx]) for key, val in self.encodings.items() }
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

training_args = TrainingArguments(
    output_dir='./tmp/electra',
    overwrite_output_dir='True',

    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
 
    logging_dir='./tmp/logs',
    logging_steps=log_interval,
    evaluation_strategy="steps",
    eval_steps=log_interval,

    # https://discuss.huggingface.co/t/save-only-best-model-in-trainer/8442/8
    save_total_limit=2,
    save_strategy='no',
    load_best_model_at_end=False,

    # 성능 그래프 시각화 유틸리티 (사용안함)
    # report_to="wandb",
    # run_name="transformer-kcelectra_1"
)
#정확도 측정을 위한 함수 정의
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # average: 'micro', 'macro', 'weighted' or 'samples' 
    # 참고 https://aimb.tistory.com/152
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'acc': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics
)

model.eval()




def mean_answer_label(*preds):
  preds_sum = np.zeros(preds[0].shape[0])
  for pred in preds:
    preds_sum += np.argmax(pred, axis=-1)
  return np.round(preds_sum/len(preds), 0).astype(int)


def api_predict(sentence):
    
    cat_dict = {'0':'enfj','1':'enfp','2':'entj','3':'entp','4':'esfj','5':'esfp','6':'estj','7':'estp',
             '8':'infj','9':'infp','10':'intj','11':'intp','12':'isfj','13':'isfp','14':'istj','15':'istp'}
    result_dict = {}
    global mod
    sentence = re.sub("[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…]+", "", sentence)
    sentence = re.sub("\\n+", " ", sentence)
    sentence = re.sub("\\t+", " ", sentence)
    
    test_df = pd.DataFrame({'id':[0,1,2,3,4,5,6,7],'data':[sentence,sentence,sentence,sentence,sentence,sentence,sentence,sentence],'label':[0,0,0,0,0,0,0,0]})

    encoded_test = tokenizer(
        test_df['data'].tolist(),
        return_tensors='pt',
        max_length=max_len,
        padding=True,
        truncation=True,
        add_special_tokens=True
    )

    test_dataset = ReviewDataset(encoded_test, test_df['label'].values)
    predictions = trainer.predict(test_dataset)
     
    predictions[0][0][0]

    pred = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range(0,8):
        for j in range(0,16):
            pred[j] += predictions[0][i][j]/8
    
    import math
    def normpdf(x, mean, sd):
        var = float(sd)**2
        denom = (2*math.pi*var)**.5
        num = math.exp(-(float(x)-float(mean))**2/(2*var))
        return num/denom
    
    sumno = 0
    for i in pred:
        sumno += normpdf(i,-3,1)
    sumno
    preds = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(0,16):
        preds[i] = round((normpdf(pred[i],-3,1)/sumno)*100)
   
    maxIndex = str(preds.index(max(preds)))
    
    

    
    for key, value in cat_dict.items():
        result_dict.update({value:preds[int(key)]})
    
    x = np.arange(16)
    predi = cat_dict[preds]
    result_dict.update({'mbti':cat_dict[maxIndex]})

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
