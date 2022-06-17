請執行以下指令:
pip install -r /path/to/requirements.txt
python demo_server.py xxxx      # xxxx為port數字

-----
其他說明:
key_words.txt為關鍵字列表，供參考

total gpu memory cost : ~3GB
偵測到關鍵字->模板回覆
模板回覆規則:
    每個關鍵字對應到[empathy, encouragement, advice]三種回覆(有些只有1~2種)，每種回覆僅能被使用一次，輸入"清除"或打招呼關鍵字來重置使用次數。
    第一次偵測到關鍵字時，會使用empathy回覆，下一句話中若有偵測到新的關鍵字，則使用新關鍵字的empahty回覆，
    若下一句話中無新關鍵字，則依序使用encouragement、advice回覆
無關鍵字->Blender Bot
