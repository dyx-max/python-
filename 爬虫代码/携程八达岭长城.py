import requests
import json
import csv
from concurrent.futures import ThreadPoolExecutor
from proxy import get_proxy

f = open('八达岭长城.csv',mode='a',newline='',encoding='utf-8')
writer = csv.writer(f)
def parsepage(i):
    url = 'https://m.ctrip.com/restapi/soa2/13444/json/getCommentCollapseList'
    headers = {
        'Uesr-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36'
    }
    data = {
    "arg":{
        "resourceId":230,
        "resourceType":11,
        "pageIndex":str(i+1),
        "pageSize":10,
        "sortType":3,
        "commentTagId":0,#0是好评，-12是差评
        "collapseType":1,
        "channelType":7,
        "videoImageSize":"700_392",
        "starType":0
    },
    "head":{
        "cid":"09031037316027920757",
        "ctok":"",
        "cver":"1.0",
        "lang":"01",
        "sid":"8888",
        "syscode":"09",
        "auth":"",
        "extension":[
            {
                "name":"protocal",
                "value":"https"
            }
        ]
    },
    "contentType":"json"
}
    proxies = get_proxy()
    response = requests.post(url=url,headers=headers,data=json.dumps(data),proxies=proxies)
    print(response)
    page_data = response.json()
    items = page_data['result']['items']
    for item in items:
        userId = item['userInfo']['userId']
        score = item['score']
        comment = item['content']
        writer.writerow([userId,score,comment])
    print('第%s页已爬取'% str(i + 1))
if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=5) as t:
        for i in range(0,92):
            t.submit(parsepage,i)
