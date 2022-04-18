import requests
import csv
from fake_useragent import UserAgent
from proxy import get_proxy
from concurrent.futures import ThreadPoolExecutor


f = open('八达岭长城.csv',mode='a',newline='',encoding='utf-8')
writer = csv.writer(f)
writer.writerow(['用户id','星数','评论内容'])
def getone(url):
    headers  = {
        'User-Agent' : UserAgent().random
    }
    proxies = get_proxy()
    response = requests.get(url=url,headers=headers,proxies=proxies)
    print(response)
    data = response.json()
    # print(data)
    comments = data['comments']
    for comment in comments:
        userid = comment['userId']
        star = comment['star']
        com = comment['comment']
        com = com.strip()
        com = com.replace('\n','')
        writer.writerow([userid,star,com])
        # print(userid,star,com)
    print(url + '完成')
if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=5) as t:
        for i in range(2019):
            t.submit(getone,f'https://www.meituan.com/ptapi/poi/getcomment?id=271720&offset={i * 10}&pageSize=10&mode=0&sortType=1')
