# -*- coding: utf-8 -*-
import requests
import json

def send_messag_example():
    resp = requests.post("http://sms-api.luosimao.com/v1/send.json",
    auth=("api", "key-eb3829cd732d929feb06556c063010d11"),
    data={
	"mobile": "150835844571",
	"message": "hello, world【luosimao】"
    }, 
    timeout=3, 
    verify=False
    )
    result = json.loads(resp.content)
    print(result)

if __name__ == "__main__":
    send_messag_example()