import numpy as np
import pandas as pd

iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)

iofile = 'data/ufc_dot_com_fighter_data_CLEAN_28Feb2017.csv'
ufc = pd.read_csv(iofile, header=0)
ufc.head(3)

ufc['Education'] = pd.notnull(ufc.Degree) | pd.notnull(ufc.College)
ufc.Education = ufc.Education.astype(int)
ufc = ufc[ufc.Education == 1][['Name', 'Education']]

win_lose = fights.Winner.append(fights.Loser)
num_fights = win_lose.value_counts().to_frame()

set(ufc.Name) - set(win_lose)

idx = ufc[ufc.Name == 'Dan Downes'].index.item()
ufc = ufc.set_value(idx, 'Name', 'Danny Downes')

idx = ufc[ufc.Name == 'Josh Sampo'].index.item()
ufc = ufc.set_value(idx, 'Name', 'Joshua Sampo')

idx = ufc[ufc.Name == 'Miguel Angel Torres'].index.item()
ufc = ufc.set_value(idx, 'Name', 'Miguel Torres')

idx = ufc[ufc.Name == 'Rich Walsh'].index.item()
ufc = ufc.set_value(idx, 'Name', 'Richard Walsh')

idx = ufc[ufc.Name == 'Shane Del Rosario'].index.item()
ufc = ufc.set_value(idx, 'Name', 'Shane del Rosario')

idx = ufc[ufc.Name == 'Wang Sai'].index.item()
ufc = ufc.set_value(idx, 'Name', 'Sai Wang')

set(ufc.Name) - set(win_lose)

ufc.to_csv('data/ufc_name_education.csv', index=False)

