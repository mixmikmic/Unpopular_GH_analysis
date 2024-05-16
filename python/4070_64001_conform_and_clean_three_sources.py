import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('halverson')
get_ipython().magic('matplotlib inline')

fights = pd.read_csv('data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv', header=0, parse_dates=['Date'])
fighters_fm = pd.read_csv('data/fightmetric_fighters/fightmetric_fighters_CLEAN_3-6-2017.csv', header=0, parse_dates=['Dob'])
fighters_ufc = pd.read_csv('data/ufc_dot_com_fighter_data_CLEAN_28Feb2017.csv', header=0)
fighters_wiki = pd.read_csv('data/wikipedia_bdays_height_reach.csv', header=0, parse_dates=['Dob'])

fights.head()

fighters_fm['Age'] = (pd.to_datetime('today') - fighters_fm.Dob) / np.timedelta64(1, 'Y')
fighters_fm.head()

fighters_ufc.head()

fighters_wiki['Age'] = (pd.to_datetime('today') - fighters_wiki.Dob) / np.timedelta64(1, 'Y')
fighters_wiki.head()

win_lose = fights.Winner.append(fights.Loser, ignore_index=True)
win_lose = set(win_lose)

s = fights.Winner.append(fights.Loser, ignore_index=True).value_counts()
three_fights_fm = s[s >= 3].index

# should match names after convert to lowercase but will not do that here
set(three_fights_fm) - set(fighters_ufc.Name)

# note that several UFC fighters are not in the UFC database
# (e.g., Benji Radach, Scott Smith, Tito Ortiz)
idx = fighters_ufc[(fighters_ufc.Name == 'Tank Abbott') & (fighters_ufc.Nickname == 'Tank')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'David Abbott')
idx = fighters_ufc[(fighters_ufc.Name == 'Edwin Dewees') & (fighters_ufc.Nickname == 'Babyface')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Edwin DeWees')
idx = fighters_ufc[(fighters_ufc.Name == 'Ronaldo Souza') & (fighters_ufc.Nickname == 'Jacare')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Jacare Souza')
idx = fighters_ufc[(fighters_ufc.Name == 'Josh Sampo') & (fighters_ufc.Nickname == 'The Gremlin')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Joshua Sampo')
idx = fighters_ufc[(fighters_ufc.Name == 'Manny Gamburyan') & (fighters_ufc.Nickname == 'The Anvil')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Manvel Gamburyan')
idx = fighters_ufc[(fighters_ufc.Name == 'Marcio Alexandre') & (fighters_ufc.Nickname == 'Lyoto')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Marcio Alexandre Junior')
idx = fighters_ufc[(fighters_ufc.Name == 'Marcos Rogerio De Lima')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Marcos Rogerio de Lima')
idx = fighters_ufc[(fighters_ufc.Name == 'Miguel Angel Torres')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Miguel Torres')
idx = fighters_ufc[(fighters_ufc.Name == 'Mike De La Torre')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Mike de la Torre')
idx = fighters_ufc[(fighters_ufc.Name == 'Mike Van Arsdale')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Mike van Arsdale')
idx = fighters_ufc[(fighters_ufc.Name == 'Mostapha Al Turk')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Mostapha Al-Turk')
idx = fighters_ufc[(fighters_ufc.Name == 'Phil De Fries')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Philip De Fries')
idx = fighters_ufc[(fighters_ufc.Name == 'Marco Polo Reyes') & (fighters_ufc.Nickname == 'El Toro')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Polo Reyes')
idx = fighters_ufc[(fighters_ufc.Name == 'Rafael Cavalcante') & (fighters_ufc.Nickname == 'Feijao')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Rafael Feijao')
idx = fighters_ufc[(fighters_ufc.Name == 'Rameau Sokoudjou') & (fighters_ufc.Nickname == 'The African Assassin')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Rameau Thierry Sokoudjou')
idx = fighters_ufc[(fighters_ufc.Name == 'Rich Walsh') & (fighters_ufc.Nickname == 'Filthy')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Richard Walsh')
idx = fighters_ufc[(fighters_ufc.Name == 'Robbie Peralta') & (fighters_ufc.Nickname == 'Problems')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Robert Peralta')
idx = fighters_ufc[(fighters_ufc.Name == 'Antonio Rogerio Nogueira')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Rogerio Nogueira')
idx = fighters_ufc[(fighters_ufc.Name == 'Timothy Johnson')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Tim Johnson')
idx = fighters_ufc[(fighters_ufc.Name == 'Tony Frycklund') & (fighters_ufc.Nickname == 'The Freak')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Tony Fryklund')
idx = fighters_ufc[(fighters_ufc.Name == 'Tsuyoshi Kosaka') & (fighters_ufc.Nickname == 'TK')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'Tsuyoshi Kohsaka')
idx = fighters_ufc[(fighters_ufc.Name == 'William Macario') & (fighters_ufc.Nickname == 'Patolino')].index
fighters_ufc = fighters_ufc.set_value(idx, 'Name', 'William Patolino')

set(three_fights_fm) - set(fighters_ufc.Name)

from fuzzywuzzy import process

# create list of FightMetric fighters with 1 or 2 UFC fights
s = fights.Winner.append(fights.Loser, ignore_index=True).value_counts()
two_fights_fm = s[(s == 2) | (s == 1)].index

# fighters in the FightMetric database with 1 or 2 UFC fights not found in the UFC database
not_found = set(two_fights_fm) - set(fighters_ufc.Name)

# these names have no match
wrong_match = ['Nate Loughran', 'Julian Sanchez', 'Kit Cope', 'Edilberto de Oliveira' ,
               'Kevin Ferguson', 'Eddie Mendez', 'Danillo Villefort', 'Masutatsu Yano',
               'Joao Pierini', 'Saeed Hosseini']

for fighter in not_found:
     if (fighter not in wrong_match):
          best_match, score = process.extractOne(query=fighter, choices=fighters_ufc.Name)
          print fighter, '<--', best_match
          idx = fighters_ufc[fighters_ufc.Name == best_match].index
          fighters_ufc = fighters_ufc.set_value(idx, 'Name', fighter)

set(fighters_ufc[fighters_ufc.Active == 1].Name) - win_lose

len(set(fighters_wiki.Name))

len(win_lose)

len(win_lose - set(fighters_wiki.Name))

matches = ['Emil Meek', 'Joe Duffy', 'Rogerio Nogueira']
not_found = win_lose - set(fighters_wiki.Name)
for fighter in not_found:
     if (fighter in matches):
          best_match, score = process.extractOne(query=fighter, choices=fighters_wiki.Name)
          #if (score > 80): print fighter, '<--', best_match
          print fighter, '<--', best_match
          idx = fighters_wiki[fighters_wiki.Name == best_match].index
          fighters_wiki = fighters_wiki.set_value(idx, 'Name', fighter)
idx = fighters_wiki[fighters_wiki.Name == 'Dan Kelly'].index
fighters_wiki = fighters_wiki.set_value(idx, 'Name', 'Daniel Kelly')

len(win_lose - set(fighters_wiki.Name))

fighters_fm.shape[0]

s = ('_fm', '_ufc')
tmp = pd.merge(fighters_fm, fighters_ufc, on='Name', how='left', suffixes=s)
tmp.columns = [column if column != 'Dob' else 'Dob_fm' for column in tmp.columns]
tmp.head()

tmp.shape[0]

tmp = pd.merge(tmp, fighters_wiki, on='Name', how='left')
tmp.columns = tmp.columns.tolist()[:-4] + ['Dob_wiki', 'Height_wiki', 'Reach_wiki', 'Age_wiki']
tmp['ReachDiff'] = np.abs(tmp.Reach_fm - tmp.Reach_ufc)
tmp['HeightDiff'] = np.abs(tmp.Height_fm - tmp.Height_ufc)
tmp['AgeDiff'] = np.abs(tmp.Age_fm - tmp.Age_ufc)

tmp.ReachDiff.value_counts().sort_index()

tmp.HeightDiff.value_counts().sort_index()

tmp.shape[0]

tmp[['Name', 'Reach_fm', 'Reach_ufc', 'Reach_wiki', 'ReachDiff']].sort_values('ReachDiff', ascending=False).head(20)

tmp[['Name', 'Active', 'Height_fm', 'Height_ufc', 'Height_wiki', 'HeightDiff']].sort_values('HeightDiff', ascending=False).head(20)

tmp[['Name', 'Active', 'Age_fm', 'Age_ufc', 'Age_wiki', 'Dob_fm', 'Dob_wiki', 'AgeDiff']].sort_values('AgeDiff', ascending=False).head(40)

# slow but okay for small data
fighters = tmp.Name.copy()
for fighter in fighters:
     idx = tmp[tmp.Name == fighter].index
     # adjust reach
     if pd.isnull(tmp.loc[idx, 'Reach_fm'].values):
          tmp.set_value(idx, 'Reach_fm', tmp.loc[idx, 'Reach_wiki'].values)
     if pd.notnull(tmp.loc[idx, 'Reach_ufc'].values) and tmp.loc[idx, 'Active'].item():
          tmp.set_value(idx, 'Reach_fm', tmp.loc[idx, 'Reach_ufc'].values)
     # adjust height
     if pd.isnull(tmp.loc[idx, 'Height_fm'].values):
          tmp.set_value(idx, 'Height_fm', tmp.loc[idx, 'Height_wiki'].values)
     if pd.notnull(tmp.loc[idx, 'Height_ufc'].values) and tmp.loc[idx, 'Active'].item():
          tmp.set_value(idx, 'Height_fm', tmp.loc[idx, 'Height_ufc'].values)
     # date of birth
     if pd.isnull(tmp.loc[idx, 'Dob_fm'].values):
          tmp.set_value(idx, 'Dob_fm', tmp.loc[idx, 'Dob_wiki'].values)

tmp[['Name', 'Active', 'Reach_fm', 'Reach_ufc', 'Reach_wiki', 'ReachDiff']].head(20)

tmp[['Name', 'Active', 'Height_fm', 'Height_ufc', 'Height_wiki', 'HeightDiff']].head(20)

fnl = tmp.iloc[:, :11]
fnl['LegReach'] = tmp.LegReach
cols = ['Name', 'Nickname', 'Dob', 'Age', 'Weight', 'Height', 'Reach', 'Stance', 'Win', 'Loss', 'Draw', 'LegReach']
fnl.columns = cols
fnl.Age = fnl.Age.apply(lambda x: x if pd.isnull(x) else round(x, 1))
cols = ['Name', 'Nickname', 'Dob', 'Weight', 'Height', 'Reach', 'LegReach', 'Stance', 'Win', 'Loss', 'Draw']
fnl[cols].to_csv('data/fightmetric_fighters_with_corrections_from_UFC_Wikipedia_CLEAN.csv', index=False)

