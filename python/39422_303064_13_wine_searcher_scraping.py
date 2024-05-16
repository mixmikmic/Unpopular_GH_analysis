import pandas as pd
import numpy as np

import requests
import json
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

from fake_useragent import UserAgent
import multiprocess as mp

from glob import glob

import dill
import re
import time

# A function to create the Selenium web driver

def make_driver(port):
    
    service_args = ['--proxy=127.0.0.1:{}'.format(port), '--proxy-type=socks5']
    
    dcap = dict(DesiredCapabilities.PHANTOMJS)
    ua = UserAgent()
    dcap.update({'phantomjs.page.settings.userAgent':ua.random})
    
    phantom_path = '/usr/bin/phantomjs'
    
    driver = webdriver.PhantomJS(phantom_path, 
                                   desired_capabilities=dcap,
                                   service_args=service_args)
    
    return driver

ncomputers = 16
nthreads = 16

port_nos = np.array([8081+x for x in range(ncomputers)])

# Start the ssh tunnels
get_ipython().system(' ./ssh_tunnels.sh')

def scrape_catalog(args):
    port, page_nums = args

    base_url = 'http://www.wine-searcher.com/biz/producers?s={}'
    
    driver = make_driver(port)

    table_list = list()
    for num in page_nums:
        print num
        
        full_url = base_url.format(num * 25 + 1)

        driver.get(full_url)
        time.sleep(10. + np.random.random()*5)
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        
        try:
            table = pd.read_html(html)[2]
            columns = table.iloc[0].values
            columns[2] = 'Wines'
            table = table.iloc[1:25]
            table.columns = columns

            url_list = [x.find('a').get('href') for x in soup.find_all(attrs={'class':'wlrwdt wlbdrl vtop'})]
            table['Url'] = url_list
            table['Page'] = num

            winery_urls = list()
            for url in url_list:
                try:
                    driver.get(url)
                    time.sleep(10. + np.random.random()*5)
                    html = driver.page_source
                    soup = BeautifulSoup(html, 'lxml')

                    winery_urls.append(soup.find(text=re.compile('Winery Profile')).find_parent().get('href'))
                except:
                    winery_urls.append('')

            table['Winery Url'] = winery_urls

            table.to_pickle('../pkl/13_wine_searcher_scraping_table_{}.pkl'.format(num))

            table_list.append(table)
        except:
            pass
        
    return table_list

# Load the completed data

num_list = np.arange(0,1742)

file_list = glob('../pkl/13_wine_searcher_scraping_table_*.pkl')
int_sorter = lambda x: int(re.search(r"""_([0-9]+)\.pkl""", x).group(1))
file_nums = sorted(np.array(map(int_sorter, file_list)))

num_list = num_list[np.invert(np.in1d(num_list, file_nums))]

num_list

len(num_list)

used_threads = np.min([nthreads, len(num_list)])

used_port_nos = port_nos
if used_threads < nthreads:
    used_port_nos = port_nos[:used_threads]
    
pool = mp.Pool(processes=used_threads)
table_list = pool.map(scrape_catalog, [x for x in zip(used_port_nos, 
                                                     np.array_split(num_list, used_threads))])
pool.close()

table_df = pd.concat(sum(table_list,[]), axis=0).reset_index(drop=True)

table_df.to_pickle('../pkl/13_wine_searcher_url_table.pkl')

# Get images from profile
driver.get(winery_url)
html = driver.page_source
soup = BeautifulSoup(html, 'lxml')

num_images = int(soup.find(attrs={'id':'img_high_t'}).text)

# iterate through each image, find "view larger" and download if it exists

pos = 1
for _ in range(num_images - 1):
    
    try:
        driver.find_element_by_id('showFullLabel1').click()
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        
        wine_info = soup.find(attrs={'id':'imgLabel'})
        wine_name = wine_info.get('alt')
        wine_url = wine_info.get('src')
        wine_height = wine_info.get('height')
        wine_width = wine_info.get('width')
        
        img = req.get(wine_url)#, proxies=req_proxy)
        time.sleep(1.2)

        filext = os.path.splitext(wine_url)[-1]
        path = 'tmp_' + str(pos) + '.' + filext

        if img.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in img:
                    f.write(chunk)
                    
        driver.find_element_by_id('okButtonModal').click()
    except:
        pass
                    
    pos += 1
    driver.find_element_by_id('nextImg').click()
        
        
    
    

