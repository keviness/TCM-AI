import os
import json
import pandas as pd

def json_to_excel(json_folder, excel_folder):
    if not os.path.exists(excel_folder):
        os.makedirs(excel_folder)
    
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            with open(os.path.join(json_folder, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data = []
                # 确保每个数据单元被正确处理
                for item in data:
                    all_data.append(item)
            
            df = pd.DataFrame(all_data)
            excel_filename = os.path.splitext(filename)[0] + '.xlsx'
            df.to_excel(os.path.join(excel_folder, excel_filename), index=False)

# 调用函数，将Data文件夹下的json文件转换为excel文件，并保存到Result文件夹
json_folder = 'C:/Users/gst-0123/Desktop/Projects/GSTAIProgect/TCM-ERNIE/Data'
excel_folder = 'C:/Users/gst-0123/Desktop/Projects/GSTAIProgect/TCM-ERNIE/Result/'
json_to_excel(json_folder, excel_folder)
