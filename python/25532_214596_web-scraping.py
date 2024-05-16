import requests
from bs4 import BeautifulSoup

URL = 'https://home-assistant.io/components/'

raw_html = requests.get(URL).text
data = BeautifulSoup(raw_html, 'html.parser')

print(data.select('a')[7])

print(data.select('.btn')[0])

print(data.select('a:nth-of-type(8)'))

print(data.select('a[href="#all"]'))

print(data.select('a[href="#all"]')[0].text)

print(data.select('a[href="#all"]')[0].text[5:8])

import homeassistant.remote as remote

HOST = '127.0.0.1'
PASSWORD = 'YOUR_PASSWORD'

api = remote.API(HOST, PASSWORD)

new_state = data.select('a[href="#all"]')[0].text[5:8]
attributes = {
  "friendly_name": "Home Assistant Implementations",
  "unit_of_measurement": "Count"
}
remote.set_state(api, 'sensor.ha_implement', new_state=new_state, attributes=attributes)

