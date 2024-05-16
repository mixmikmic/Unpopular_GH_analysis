import requests
import json
# from bs4 import BeautifulSoup
import time
import pickle

token = 'E9LQF4KUV7SE2MH0'

total_accounts = 9022265 # last update
# finding total number of hon international accounts using binary search
lo, hi = total_accounts, 9100000
while lo+1 < hi:
    mid = (lo+hi)//2
    r = requests.get('http://api.heroesofnewerth.com/player_statistics/ranked/accountid/{}/?token={}'.format(mid, token))
    if r.text == 'Unauthorized!':
        print(r.text)
        break
    try:
        player_dict = r.json()
        lo = mid
    except ValueError:
        hi = mid-1
    print(lo, hi)
print('total accounts:', lo)
total_accounts = lo

db_file = 'hon_matches.pkl'
try:
    with open(db_file, 'rb') as f:
        all_matches = pickle.load(f)
except FileNotFoundError:
    all_matches = dict()
print(len(all_matches))

def clean(accounts):
    new_accounts = []
    removed = 0
    for acc in accounts:
        secs = int(acc['secs'])
        if int(acc['actions']) <= 300: # if APM is too low, the player did not contribute to the team, (disconnection assumed)
            removed += 1
        else:
            new_acc = dict()
            new_acc['hero_id'] = int(acc['hero_id'])
            new_acc['win'] = int(acc['wins'])
            new_acc['concede'] = int(acc['concedes'])
            new_acc['match_id'] = int(acc['match_id'])
            new_acc['secs'] = secs
            new_acc['team'] = int(acc['team']) - 1 # legion 0, hellbourne 1
            new_accounts.append(new_acc)
    if removed:
        print('Removed Low APM:', removed)
    return new_accounts

# return a dict that can be accessed by [match_id]
def aggregate(accounts):
    matches = dict()
    for acc in accounts:
        matchid = acc['match_id'] 
        if matchid not in matches:
            matches[matchid] = {
                'concedes': 0,
                'secs': 0,
                'legion': list(),
                'hellbourne': list()
            }
        # use max because secs is the amount of seconds that player is in the game
        matches[matchid]['secs'] = max(matches[matchid]['secs'], acc['secs'])
        if acc['secs'] == matches[matchid]['secs']: # this guy plays until the end of the game
            matches[matchid]['winner'] = acc['team'] if acc['win'] else 1 - acc['team']
        if acc['concede']:
            matches[matchid]['concedes'] += 1
        if acc['team'] == 0:
            matches[matchid]['legion'].append(acc['hero_id'])
        else:
            matches[matchid]['hellbourne'].append(acc['hero_id'])
    return matches

matchid = 147862153 # latest match id found

import sys
dump_every = 10 # iterations
sleep_period = 0.5 # sleep to prevent API requests limit
for iter in range(2501):
    matchids = []
    for i in range(25):
        matchids.append(str(matchid))
        matchid -= 1
    url = 'http://api.heroesofnewerth.com/multi_match/statistics/matchids/{}/?token={}'.format('+'.join(matchids), token)
    print('-- Requesting...', end=' ')
    while True:
        try:
            r = requests.get(url)
            break
        except:
            print('ConnectionError, pause for {} secs'.format(sleep_period * 20), file=sys.stderr)
            time.sleep(sleep_period * 20)
    print('Sleeping...', end=' ')
    time.sleep(sleep_period)
    if r.text == 'No results.':
        print(r.text)
        continue
    try:
        print('Decoding...', end=' ')
        match_accounts = r.json()
    except ValueError:
        print('Error:', r.text, file=sys.stderr)
        time.sleep(sleep_period * 10)
        continue
    match_accounts = clean(match_accounts)
    matches = aggregate(match_accounts)
    all_matches.update(matches)
    print('Total matches in this/all batch:', len(matches), len(all_matches))
    # print('Next Unused MatchID:', matchid)
    if iter % dump_every == 0:
        print('==> Dumping at iter', iter)
        with open(db_file, 'wb') as f:
            pickle.dump(all_matches, f)

print('Next Unused MatchID:', matchid)

match_accounts

matches

with open(db_file, 'wb') as f:
    pickle.dump(all_matches, f)

r = requests.get('http://api.heroesofnewerth.com/heroes/all?token=E9LQF4KUV7SE2MH0')

heroes_dict = r.json()
heroes_dict

len(heroes_dict)

new_heroes_dict = dict()
for value in heroes_dict.values():
    hero_id, hero = list(value.items())[0]
    hero_id = int(hero_id)
    new_heroes_dict[hero_id] = hero
new_heroes_dict

with open('heroes_name.pkl', 'wb') as f:
    pickle.dump(new_heroes_dict, f)

