import logging
import time
import datetime
import os
import web
import json
import numpy as np
import random
import torch
import pandas as pd
import csv
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import BertTokenizerFast
from transformers import BertTokenizer, BertForSequenceClassification
from deep_translator import GoogleTranslator
from copy import deepcopy


SAVE_PATH_GEN = 'model/model_gen'   # path to generative model
TOKENIZER_PATH_GEN = 'model/tokenizer_gen'

SAVE_PATH_EMO = 'model/model_emo'   # path to emotion_classification model

greeting_input = ['你好', '嗨', '哈囉', 'hi', 'Hi', 'HI']
goodbye_input = ['再見', '掰掰', 'Bye', 'bye']
thank_input = ['謝謝', '感謝']

label_dict = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4:'like', 5:'sadness', 6:'surprise'}
zh_label = {'anger':'憤怒', 'disgust':'厭惡', 'fear':'恐懼', 'happiness':'幸福', 'like':'喜悅', 'sadness':'悲傷', 'surprise':'驚訝'}
negative = ['anger', 'disgust', 'fear', 'sadness']

random_seed = 87
# Set random states for reproducibility
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer_gen = AutoTokenizer.from_pretrained(TOKENIZER_PATH_GEN, config=AutoConfig.from_pretrained(SAVE_PATH_GEN))
model_gen = AutoModelForCausalLM.from_pretrained(SAVE_PATH_GEN)
model_gen = model_gen.to(device)
model_gen.eval()

tokenizer_emo = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
model_emo = BertForSequenceClassification.from_pretrained(SAVE_PATH_EMO, num_labels=7)
model_emo = model_emo.to(device)
model_emo.eval()

model_chitchat = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill')
tokenizer_chitchat = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
model_chitchat = model_chitchat.to(device)

input_context = []
total_length = 0

class web_server_template:  ##宣告一個class,在下文的web.application實例化時，會根據定義將對應的url連接到這個class
    def __init__(self): 
        print('initial in {}'.format(time.time()))
    
    def emotion_classification(self, input_text):
        # inputs_emo = input_context[-1]
        # if len(input_context)>2:
        #     inputs_emo = input_context[0]+'。'+input_context[2]
        inputs_emo = input_text
        print('input for emo classify: ',inputs_emo)
        inputs_emo = tokenizer_emo(inputs_emo, return_tensors="pt").to(device)
        outputs_emo = model_emo(**inputs_emo)
        emotion = torch.argmax(outputs_emo.logits).item()
        emotion = label_dict[emotion]
        print('emotion: ', emotion)
        return emotion

    def emp_respond(self):
        global total_length
        global input_context
        context = tokenizer_gen.eos_token.join(input_context)
        context += '<|endoftext|>'
        print('context: ', context)
        context_emb = context
        context_emb = tokenizer_gen.encode(context_emb)
        context_emb = torch.tensor(context_emb)
        context_emb = torch.unsqueeze(context_emb, 0)
        context_emb = context_emb.to(device)
        res_emb = model_gen.generate(context_emb , max_length=1000 #, max_length=len(context_emb[0])+30
        , pad_token_id=tokenizer_gen.eos_token_id  #, eos_token_id=tokenizer_gen.eos_token_id
        # , num_beams=5, min_length=2  #, no_repeat_ngram_size=2, early_stopping=True   #,length_penalty=0.1
        # ,do_sample=True , top_k=50 , top_p=0.8
         )
        res = tokenizer_gen.decode(res_emb[:, context_emb.shape[-1]:][0], skip_special_tokens=True)
        input_context.append(res)
        total_length += len(res)
        if not res.isdigit():
            res = GoogleTranslator(source='auto', target='zh-tw').translate(res)
        return res

    def chitchat_respond(self, eng_text):    #Only the current 1 sentence will be used to generate chitchat response. Input context will be cleared at the same time.
        global total_length
        global input_context

        # print('context: ', eng_text) 
        # total_length=0
        # input_context.clear()
        # context_emb = eng_text

        context_emb = tokenizer_chitchat.eos_token.join(input_context)
        print('context: ', context_emb) 

        inputs_chitchat = tokenizer_chitchat([context_emb], return_tensors='pt').to(device) 
        reply_ids = model_chitchat.generate(**inputs_chitchat)
        res = tokenizer_chitchat.decode(reply_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        input_context.append(res)
        total_length += len(res)
        
        if not res.isdigit():
            res = GoogleTranslator(source='auto', target='zh-tw').translate(res)
        return res

    def POST(self):  ##當server收到一個指向這個class URL的POST請求，會觸發class中命名為POST的函數，GET請求同理
        receive = json.loads(str(web.data(),encoding='utf-8'))  ##使用json.loads將json格式讀取為字典
        print('[Message] Post message receive:{}'.format(receive))
        result = True
        received_data = receive["msg"]
        user_id = str(receive["user_id"])

        global total_length
        global input_context

        has_keyword = False

        if received_data == '清除':
            total_length=0
            input_context.clear()
            passing_information = 'history cleared!'
            has_keyword = True

        if not has_keyword:
            for x in greeting_input:
                if x in received_data:
                    passing_information = '嗨，今天想和我聊些什麼呢?'
                    total_length=0
                    input_context.clear()
                    has_keyword = True
                    break

        if not has_keyword:
            for x in goodbye_input:
                if x in received_data:
                    passing_information = '再見，隨時歡迎你再來找我聊天'
                    total_length=0
                    input_context.clear()
                    has_keyword = True 
                    break

        if not has_keyword:
            for x in thank_input:
                if x in received_data:
                    passing_information = '不用客氣，很高興能幫上你的忙'
                    total_length=0
                    input_context.clear()
                    has_keyword = True
                    break
        
        if not has_keyword:
            text = received_data
            eng_text = text
            if not text.isdigit():
                eng_text = GoogleTranslator(source='auto', target='en').translate(text)

            # if not len(input_context)%2 == 0:
            #     total_length -= len(input_context[0])
            #     input_context.pop(0)
            # while len(input_context)>=4:
            #     for i in range(2):
            #         total_length -= len(input_context[0])
            #         input_context.pop(0)

            while len(input_context)>=3:
                total_length -= len(input_context[0])
                input_context.pop(0)

            input_context.append(eng_text)
            total_length += len(eng_text)
            # print('length of input_context : ',len(input_context))
            emotion = self.emotion_classification(text)

            if emotion in negative:
                res = self.emp_respond()
    
            else :
                res = self.chitchat_respond(eng_text)

            # res = self.emp_respond(input_context)

            print('response: ',res)
            
            passing_information = res

        return_json = {'results':result,'return_message':passing_information}
        return_data = json.dumps(return_json,sort_keys=True,separators=(',',':'),ensure_ascii=False) ##打包回傳信息為json
        return return_data  ##回傳

    def GET(self):
        return 'Hello World!'

URL_main = ("/","web_server_template")  ##宣告URL與class的連接
if __name__ == '__main__':
    logging.basicConfig()
    app = web.application(URL_main,globals(), autoreload = False)  ##初始化web application，默認地址為127.0.0.1:8080，locals()代表web.py會在當前文件內尋找url對應的class
    app.run()  ##運行web application

    #python server.py 9111
