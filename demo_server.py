import logging
import time
import web
import json
import numpy as np
import random
import torch
import pandas as pd
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from deep_translator import GoogleTranslator

#########################################################################################################
#   total gpu memory cost : ~3GB
#
#   偵測到關鍵字->模板回覆
#   模板回覆規則:
#       每個關鍵字對應到[empathy, encouragement, advice]三種回覆(有些只有1~2種)，每種回覆僅能被使用一次，輸入"清除"或打招呼關鍵字來重置使用次數。
#       第一次偵測到關鍵字時，會使用empathy回覆，下一句話中若有偵測到新的關鍵字，則使用新關鍵字的empahty回覆，
#       若下一句話中無新關鍵字，則依序使用encouragement、advice回覆
#   無關鍵字->Blender Bot
#########################################################################################################

greeting_input = ['你好', '嗨', '哈囉', 'hi', 'Hi', 'HI']
goodbye_input = ['再見', '掰掰', 'Bye', 'bye']
thank_input = ['謝謝', '感謝']

with open('template.json', newline='', encoding="utf-8") as f:
    template = json.load(f)

key_words = []
template_responses = []
prev_keyword_idx = -1
for key in template:
    key_words.append(key)
    temp_list = []
    if len(template[key]['empathy'])>0:
        temp_list.append(template[key]['empathy'][0])
    if len(template[key]['encouragement'])>0:
        temp_list.append(template[key]['encouragement'][0])
    if len(template[key]['advice'])>0:
        temp_list.append(template[key]['advice'][0])
    template_responses.append(temp_list)
used_response_num = [0]*len(key_words)

label_dict = {0:'positive', 1:'negative', 2:'ambiguous'}
random_seed = 87
# Set random states for reproducibility
random.seed(random_seed)
np.random.seed(random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_chitchat = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill')
tokenizer_chitchat = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
model_chitchat = model_chitchat.to(device)

input_context = []
total_length = 0
# translator = googletrans.Translator()

class web_server_template:  ##宣告一個class,在下文的web.application實例化時，會根據定義將對應的url連接到這個class
    def __init__(self):  ##初始化類別
        print('initial in {}'.format(time.time()))

    def POST(self):  ##當server收到一個指向這個class URL的POST請求，會觸發class中命名為POST的函數，GET請求同理
        recive = json.loads(str(web.data(),encoding='utf-8'))  ##使用json.loads將json格式讀取為字典
        print('[Message] Post message recive:{}'.format(recive))
        result = True
        received_data = recive["msg"]

        global total_length
        global input_context
        global prev_keyword_idx
        global used_response_num

        has_keyword = False

        if received_data == '清除':
            total_length=0
            input_context.clear()
            prev_keyword_idx = -1
            used_response_num = [0]*len(key_words)
            passing_information = 'history cleared!'
            has_keyword = True

        if not has_keyword:
            for x in greeting_input:
                if x in received_data:
                    passing_information = '嗨，今天想和我聊些什麼呢?'
                    total_length=0
                    input_context.clear()
                    prev_keyword_idx = -1
                    used_response_num = [0]*len(key_words)
                    has_keyword = True
                    break

        if not has_keyword:
            for x in goodbye_input:
                if x in received_data:
                    passing_information = '再見，隨時歡迎你再來找我聊天'
                    total_length=0
                    input_context.clear()
                    prev_keyword_idx = -1
                    used_response_num = [0]*len(key_words)
                    has_keyword = True 
                    break

        if not has_keyword:
            for x in thank_input:
                if x in received_data:
                    passing_information = '不用客氣，很高興能幫上你的忙'
                    total_length=0
                    input_context.clear()
                    prev_keyword_idx = -1
                    used_response_num = [0]*len(key_words)
                    has_keyword = True
                    break

        if not has_keyword:
            for i in range(len(key_words)):
                if key_words[i] in received_data and used_response_num[i] < len(template_responses[i]):   # keyword detected and response available
                    passing_information = template_responses[i][used_response_num[i]]
                    used_response_num[i] += 1
                    prev_keyword_idx = i
                    text = received_data
                    # text = translator.translate(text, dest='en').text
                    text = GoogleTranslator(source='auto', target='en').translate(text)
                    input_context.append(text)
                    total_length += len(text)
                    text = passing_information
                    # text = translator.translate(text, dest='en').text
                    text = GoogleTranslator(source='auto', target='en').translate(text)
                    total_length += len(text)
                    input_context.append(text)
                    has_keyword = True
                    break

        if not has_keyword:
            if prev_keyword_idx != -1 and used_response_num[prev_keyword_idx]<len(template_responses[prev_keyword_idx]):  # no new keywords detected after replying empathy response
                passing_information = template_responses[prev_keyword_idx][used_response_num[prev_keyword_idx]]
                used_response_num[prev_keyword_idx] += 1
                text = received_data
                # text = translator.translate(text, dest='en').text
                if not text.isdigit():
                    text = GoogleTranslator(source='auto', target='en').translate(text)
                input_context.append(text)
                total_length += len(text)
                text = passing_information
                # text = translator.translate(text, dest='en').text
                if not text.isdigit():
                    text = GoogleTranslator(source='auto', target='en').translate(text)
                total_length += len(text)
                input_context.append(text)
                has_keyword = True
        
        if not has_keyword:
            prev_keyword_idx = -1
            text = received_data
            # text = translator.translate(text, dest='en').text
            if not text.isdigit():
                text = GoogleTranslator(source='auto', target='en').translate(text)

            input_context.append(text)
            total_length += len(text)
            while len(input_context)>4:
                total_length -= len(input_context[0])
                input_context.pop(0)

            print('input context: ', text)      #use chitchat model to generate response
            total_length=0
            input_context.clear()
            context_emb = text
            inputs_chitchat = tokenizer_chitchat([context_emb], return_tensors='pt').to(device) 
            reply_ids = model_chitchat.generate(**inputs_chitchat)
            res = tokenizer_chitchat.decode(reply_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print(res)
            input_context.append(res)
            total_length += len(res)
            # res = translator.translate(res, dest='zh-tw').text
            if not res.isdigit():
                res = GoogleTranslator(source='auto', target='zh-tw').translate(res)
            print(res)
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

    #python demo_server.py 9111
