import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)

fights = fights[(fights.Date > pd.to_datetime('2005-01-01')) & (fights.Outcome != 'no contest')]
fights.shape[0]

win_lose = fights.Winner.append(fights.Loser).unique()
win_lose.size

fighter = 'Anderson Silva'
msk = (fights.Winner == fighter) | (fights.Loser == fighter)
first_fight = fights[msk].sort_values('Date').head(1)[['Winner', 'Outcome', 'Loser', 'Date']]
first_fight

total_fights = 0
wins = 0
losses = 0
draws = 0
for fighter in win_lose:
     msk = (fights.Winner == fighter) | (fights.Loser == fighter)
     first_fight = fights[msk].sort_values('Date').head(1)[['Winner', 'Outcome', 'Loser', 'Date']]
     assert first_fight.shape[0] == 1, "DF length: " + fighter
     if (first_fight.Winner.item() == fighter and first_fight.Outcome.item() == 'def.'): wins += 1
     if (first_fight.Loser.item() == fighter and first_fight.Outcome.item() == 'def.'): losses += 1
     if (first_fight.Outcome.item() == 'draw'): draws += 1
     total_fights += 1
wins, losses, draws, total_fights

598/1379.

773/1379.

8/1379.

598+ 773+ 8

