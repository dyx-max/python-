import time
import requests
import parsel
import csv
import random
from proxy import get_proxy
from fake_useragent import UserAgent
from concurrent.futures import ThreadPoolExecutor

# url = 'https://movie.douban.com/subject/26825482/comments?start=0&limit=20&status=P&sort=new_score'
f = open('comments.csv',mode='a',newline='',encoding='utf-8')
writer = csv.writer(f)
writer.writerow(['用户名','评论'])
def one_page(url):
    cookie = 'bid=B23pgM3bRjw; douban-fav-remind=1; __gads=ID=db3b84ba10f8a60f-226a683cdad00048:T=1646556723:RT=1646556723:S=ALNI_MZokOyQbIco8oeGYHHTuX9tYYCKdw; gr_user_id=50c303bd-bf91-48d0-b325-fa4cc5d43f93; ll="108288"; _vwo_uuid_v2=DEAF8A16F7900CBEDB4CAFD421D013626|25b068403322af1da8d257ffc2c2c70e; viewed="35106838_26378430_26677686"; __yadk_uid=kLsWns7wU9aO7yTntXCOoNdubm02XTox; dbcl2="255975996:5bct1RFbaW8"; push_noty_num=0; push_doumail_num=0; __utmv=30149280.25597; _ga_RXNMP372GL=GS1.1.1649840489.2.1.1649841653.57; ck=h8Qa; ap_v=0,6.0; __utma=30149280.306304810.1646556768.1649995631.1650032226.16; __utmc=30149280; __utmz=30149280.1650032226.16.13.utmcsr=cn.bing.com|utmccn=(referral)|utmcmd=referral|utmcct=/; __utmt=1; __utmb=30149280.2.10.1650032226; __utma=223695111.306304810.1646556768.1649995631.1650032230.10; __utmb=223695111.0.10.1650032230; __utmc=223695111; __utmz=223695111.1650032230.10.8.utmcsr=douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/; _pk_ref.100001.4cf6=["","",1650032230,"https://www.douban.com/"]; _pk_ses.100001.4cf6=*; _ga=GA1.2.306304810.1646556768; _gid=GA1.2.1813140820.1650032281; _pk_id.100001.4cf6=ae83c2c3ce18e7c2.1649164187.10.1650032289.1649997006.'
    headers = {
        'User-Agent': UserAgent().random,
        'cookie': cookie.encode('utf-8')
    }

    proxies = get_proxy()
    # print(proxies)
    resp = requests.get(url=url, headers=headers,proxies=proxies)
    print(resp)
    html_data = resp.text
    selector = parsel.Selector(html_data)
    divs = selector.xpath('//*[@id="comments"]/div')
    #print(divs)
    for div in divs:
        id_name = div.xpath('./div[2]/h3/span[2]/a/text()').get()
        comment = div.xpath('./div[2]/p/span/text()').get()
        if comment != None :
            comment = comment.replace('\n','')
        else:
            pass
        # print(id_name,comment)
        writer.writerow([id_name,comment])
    time.sleep(random.randint(0, 7))
    print(url+'爬取完成')


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=5) as t:
        for i in range(0,30):

           t.submit(one_page,f'https://movie.douban.com/subject/26825482/comments?start={i * 20}&limit=20&status=P&sort=new_score')
        for i in range(1,11):

            t.submit(one_page,f'https://movie.douban.com/subject/26825482/comments?start={i * 600}&limit=20&status=P&sort=new_score')