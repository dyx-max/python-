import requests
import csv
import parsel
from concurrent.futures import ThreadPoolExecutor
from proxy import get_proxy

f = open('驴妈妈_八达岭长城.csv',mode='a',newline='',encoding='utf-8')
writer = csv.writer(f)
def parseone(i):
    url = 'http://ticket.lvmama.com/scenic_front/comment/newPaginationOfComments'
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36 Edg/100.0.1185.39'
    }
    data = {
    'type': 'all',
    'currentPage': i + 1,
    'totalCount': 8190,
    'placeId': 102987,
    'productId':None,
    'placeIdType': 'PLACE',
    'isPicture': None,
    'isBest':None,
    'isPOI': 'Y',
    'isELong': 'N'
    }
    proxies = get_proxy()
    response = requests.post(url=url,headers=headers,data=data,proxies=proxies)
    html_data = response.text
    divs = parsel.Selector(html_data).xpath('/html/body/div')[:10]
    for div in divs:
        userID = div.xpath('./div[@class="com-userinfo"]/p/a[1]/text()').get()
        score = div.xpath('./div[1]/p/span[2]/i/text()').get()
        if score != None:
            score = score.strip()
            score = score.replace('\n','')
            score = score.replace(' ','')
        content = div.xpath('./div[@class="ufeed-content"]/text()').getall()[1]
        if content != None:
            content = content.strip()
            content = content.replace('\n','')
            content = content.replace(' ','')
        writer.writerow([userID,score,content])
    print('第%s页已爬取'%str(i + 1))
if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=5) as t:
        for i in range(819):
            t.submit(parseone,i)
