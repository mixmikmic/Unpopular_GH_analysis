from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
import time
import requests
import random
import pandas as pd

# This is a very simple example of how to parse stuff in the main Search page.
start_url = "http://www.indeed.com/m/jobs?q=backend+engineer"
page = requests.get(start_url)
start_soup = BeautifulSoup(page.text, "html.parser")
print(start_soup.title.text)

# This is a very simple example of how to parse stuff out of an individual job page.
test_url="https://www.indeed.com/m/viewjob?jk=10bca72276277b22"
test_job_link_page = requests.get(test_url)
test_job_link_soup = BeautifulSoup(test_job_link_page.text, "html.parser")
#print(test_job_link_soup.body.p.text)
print('job title:', test_job_link_soup.body.p.b.text.strip())
print('company name:', test_job_link_soup.body.p.b.next_sibling.next_sibling.string.strip())
print('location:', test_job_link_soup.body.p.span.text.strip())
#print('summary:', test_job_link_soup.find(name="div", attrs={"id":"desc"}).text)

# Given a soup object, parse out all the job urls.
def extract_job_links(soup): 
  job_links = []
  for h in soup.find_all(name="h2", attrs={"class":"jobTitle"}):
      for a in h.find_all(name="a", attrs={"rel":"nofollow"}):
        job_links.append(a["href"])
  return(job_links)

# Given a list of job urls (links), parse out relevant content for all the job pages and store in a dataframe
def extract_job_listings(job_links):
    job_link_base_url="https://www.indeed.com/m/{}"
    job_listings=[]
    for job_link in job_links:
        j = random.randint(1000,2200)/1000.0
        time.sleep(j) #waits for a random time so that the website don't consider you as a bot
        job_link_url = job_link_base_url.format(job_link)
        #print('job_link_url:', job_link_url)
        job_link_page = requests.get(job_link_url)
        job_link_soup = BeautifulSoup(job_link_page.text, "html.parser")
        #print('job_link_text:', job_link_soup.text)
        #job_listings_df.loc[count] = extract_job_listing(job_link_url, job_link_soup)
        job_listings.append(extract_job_listing(job_link_url, job_link_soup))
    
    
    columns = ["job_url", "job_title", "company_name", "location", "summary", "salary"]
    job_listings_df = pd.DataFrame(job_listings, columns=columns)
    return job_listings_df

# Given a single job listing url and the corresponding page, parse out the relevant content to create an entry 
def extract_job_listing(job_link_url, job_link_soup):
    job_listing = []
    job_listing.append(job_link_url)
    job_listing.append(job_link_soup.body.p.b.text.strip())
    job_listing.append(job_link_soup.body.p.b.next_sibling.next_sibling.string.strip())
    job_listing.append(job_link_soup.body.p.span.text.strip())
    job_listing.append(job_link_soup.find(name="div", attrs={"id":"desc"}).text)
    job_listing.append("Not_Found")
    return job_listing
    
    #print(job_listing)        

# Given a single page with many listings, go to the individual job pages and store all the content to CSV
def parse_job_listings_to_csv(soup, fileName):
    job_links = extract_job_links(soup)
    job_posts = extract_job_listings(job_links)
    job_posts.to_csv(fileName, encoding='utf-8', index=False)

# A simple example to show how to use Selenium to go through the next links.
next_page_url_pattern="https://www.indeed.com/m/{}"
driver = webdriver.Chrome('/Users/Raghu/Downloads/chromedriver')
start_url = "http://www.indeed.com/m/jobs?q=data+analyst"
driver.set_page_load_timeout(15)
driver.get(start_url)
start_soup = BeautifulSoup(driver.page_source, "html.parser")

print('first page jobs:')
print(extract_job_links(start_soup))
#print start_soup.find(name='a', text='Next')
next_link=driver.find_elements_by_xpath("//a[text()='Next']")[0]
#print next_link
next_link.click()

# Use Selenium to go to do pagination - Click the next links (for now limit to only 5 next links)
driver = webdriver.Chrome('/Users/Raghu/Downloads/chromedriver')
start_url = "http://www.indeed.com/m/jobs?q=frontend+engineer"
driver.set_page_load_timeout(15)
driver.get(start_url)
start_soup = BeautifulSoup(driver.page_source, "html.parser")

parse_job_listings_to_csv(start_soup, "job_postings_0.csv")

for i in range(1,5):
    print('loading {} page'.format(str(i)))
    j = random.randint(1000,3300)/1000.0
    time.sleep(j) #waits for a random time so that the website don't consider you as a bot
    next_page_url = driver.find_elements_by_xpath("//a[text()='Next']")[0]
    page_loaded = True
    try:
        next_page_url.click()
    except TimeoutException:
        get_info = False
        driver.close()
    if page_loaded:
        soup=BeautifulSoup(driver.page_source)
        parse_job_listings_to_csv(soup, "job_postings_{}.csv".format(str(i)))

driver.close()



