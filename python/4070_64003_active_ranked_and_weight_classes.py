import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('halverson')
get_ipython().magic('matplotlib inline')

ufc = pd.read_csv('data/ufc_dot_com_fighter_data_CLEAN_28Feb2017.csv', header=0)
ufc.head(3)

ufc[ufc.Active == 1].shape[0]

iofile = 'data/fightmetric_fighters_with_corrections_from_UFC_Wikipedia_CLEAN.csv'
fighters = pd.read_csv(iofile, header=0, parse_dates=['Dob'])
fighters.head(3)

iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)

with open('data/ranked_ufc_fighters_1488838405.txt') as f:
     ranked = f.readlines()
ranked = [fighter.strip() for fighter in ranked]

set(ranked) - set(ufc[ufc.Active == 1].Name)

ufc[ufc.Name.str.contains('Souza')]

idx = ufc[ufc.Name == 'Ronaldo Souza'].index
ufc = ufc.set_value(idx, 'Name', 'Jacare Souza')
idx = ufc[ufc.Name == 'Timothy Johnson'].index
ufc = ufc.set_value(idx, 'Name', 'Tim Johnson')
idx = ufc[ufc.Name == 'Antonio Rogerio Nogueira'].index
ufc = ufc.set_value(idx, 'Name', 'Rogerio Nogueira')

set(ranked) - set(ufc[ufc.Active == 1].Name)

set(ranked) - set(fighters.Name)

set(ufc[ufc.Active == 1].Name) - set(fighters.Name)

from fuzzywuzzy import process

for fighter in set(ufc[ufc.Active == 1].Name) - set(fighters.Name):
     best_match, score = process.extractOne(query=fighter, choices=fighters.Name)
     if score >= 87:
          idx = ufc[ufc.Name == fighter].index
          ufc = ufc.set_value(idx, 'Name', best_match)
          print fighter, '-->', best_match, score

set(ufc[ufc.Active == 1].Name) - set(fighters.Name)

fights.WeightClass.value_counts()

f = 'Jessica Andrade'
wins = fights[fights.Winner == f][['Winner', 'WeightClass']]
loses = fights[fights.Loser == f][['Loser', 'WeightClass']]
loses.columns = ['Winner', 'WeightClass']
wins.append(loses).WeightClass.value_counts().sort_values(ascending=False).index[0]

win_lose = fights.Winner.append(fights.Loser).unique()
fighter_weightclass = []
for fighter in win_lose:
     wins = fights[fights.Winner == fighter][['Winner', 'WeightClass']]
     loses = fights[fights.Loser == fighter][['Loser', 'WeightClass']]
     loses.columns = ['Winner', 'WeightClass']
     weightclass = wins.append(loses).WeightClass.value_counts().sort_values(ascending=False).index[0]
     fighter_weightclass.append((fighter, weightclass))

majority = pd.DataFrame(fighter_weightclass)
majority.columns = ['Name', 'WeightClass']
majority.WeightClass.value_counts()

majority.shape[0]

majority = majority.merge(ufc[['Name', 'Active']], on='Name', how='left')

majority[majority.WeightClass == 'Open Weight']

idx = majority[majority.Name == 'Ken Shamrock'].index
majority = majority.set_value(idx, 'WeightClass', 'Light Heavyweight')

majority[majority.WeightClass == 'Catch Weight']

idx = majority[majority.Name == 'Augusto Mendes'].index
majority = majority.set_value(idx, 'WeightClass', 'Bantamweight')
idx = majority[majority.Name == 'Darrell Horcher'].index
majority = majority.set_value(idx, 'WeightClass', 'Lightweight')
idx = majority[majority.Name == 'Alexis Dufresne'].index
majority = majority.set_value(idx, 'WeightClass', "Women's Bantamweight")
idx = majority[majority.Name == 'Joe Jordan'].index
majority = majority.set_value(idx, 'WeightClass', 'Lightweight')

f = 'Ken Shamrock'
fights[(fights.Winner == f) | (fights.Loser == f)]

wc = ['Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight', 'Welterweight',
     'Middleweight', 'Light Heavyweight', 'Heavyweight', "Women's Strawweight", "Women's Bantamweight"]
ranked_weightclass = []
for i, w in enumerate(wc):
     for j in range(16):
          ranked_weightclass.append((ranked[i * 16 + j], w))

by_rank = pd.DataFrame(ranked_weightclass)
by_rank.columns = ['Name', 'WeightClass']
z = majority.merge(by_rank, on='Name', how='inner', suffixes=('_majority', '_by_rank'))
z[z.WeightClass_majority != z.WeightClass_by_rank]

idx = majority[majority.Name == 'Anthony Johnson'].index
majority = majority.set_value(idx, 'WeightClass', 'Light Heavyweight')

majority.columns = ['Name', 'WeightClassMajority', 'Active']
majority.to_csv('data/weight_class_majority.csv', index=False)

