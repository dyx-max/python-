import time

import requests
from fake_useragent import UserAgent

def get_proxy():
    api = 'http://dps.kdlapi.com/api/getdps/?orderid=925017101556962&num=1&signature=phi8h5aflwas3pitih7us9y5jj3aw83p&pt=1&dedup=1&sep=1'
    headers = {
        'User-Agent': UserAgent().random
    }
    response = requests.get(url=api, headers=headers)
    # time.sleep(2)
    proxy = response.text.strip()
    proxies = {
        'http': 'http://' + proxy,
        'https': 'http://' + proxy
    }
    return proxies



