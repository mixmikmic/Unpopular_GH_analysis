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

win_after_loss = 0
loss_after_loss = 0
draw_after_loss = 0

win_after_win = 0
loss_after_win = 0
draw_after_win = 0

total_after_loss = 0
total_after_win = 0

win_lose = fights.Winner.append(fights.Loser).unique()
for fighter in win_lose:
     msk = (fights.Winner == fighter) | (fights.Loser == fighter)
     all_fights = fights[msk].sort_values('Date').reset_index()
     for i in range(0, all_fights.shape[0] - 1):
          cond1 = all_fights.loc[i + 1, 'Winner'] == fighter
          cond2 = all_fights.loc[i + 1, 'Outcome'] == 'def.'
          cond3 = all_fights.loc[i, 'Loser'] == fighter
          cond4 = all_fights.loc[i, 'Outcome'] == 'def.'
          if all([cond1, cond2, cond3, cond4]):
               win_after_loss += 1
               total_after_loss += 1
          cond1 = all_fights.loc[i + 1, 'Loser'] == fighter
          cond2 = all_fights.loc[i + 1, 'Outcome'] == 'def.'
          cond3 = all_fights.loc[i, 'Loser'] == fighter
          cond4 = all_fights.loc[i, 'Outcome'] == 'def.'
          if all([cond1, cond2, cond3, cond4]):
               loss_after_loss += 1
               total_after_loss += 1
          cond1 = all_fights.loc[i + 1, 'Winner'] == fighter
          cond2 = all_fights.loc[i + 1, 'Outcome'] == 'def.'
          cond3 = all_fights.loc[i, 'Winner'] == fighter
          cond4 = all_fights.loc[i, 'Outcome'] == 'def.'
          if all([cond1, cond2, cond3, cond4]):
               win_after_win += 1
               total_after_win += 1
          cond1 = all_fights.loc[i + 1, 'Loser'] == fighter
          cond2 = all_fights.loc[i + 1, 'Outcome'] == 'def.'
          cond3 = all_fights.loc[i, 'Winner'] == fighter
          cond4 = all_fights.loc[i, 'Outcome'] == 'def.'
          if all([cond1, cond2, cond3, cond4]):
               loss_after_win += 1
               total_after_win += 1

          # draw after loss
          cond2 = all_fights.loc[i + 1, 'Outcome'] == 'draw'
          cond3 = all_fights.loc[i, 'Loser'] == fighter
          cond4 = all_fights.loc[i, 'Outcome'] == 'def.'
          if all([cond2, cond3, cond4]):
               draw_after_loss += 1
               total_after_loss += 1     
          # draw after win
          cond2 = all_fights.loc[i + 1, 'Outcome'] == 'draw'
          cond3 = all_fights.loc[i, 'Winner'] == fighter
          cond4 = all_fights.loc[i, 'Outcome'] == 'def.'
          if all([cond2, cond3, cond4]):
               draw_after_win += 1
               total_after_win += 1
               
print win_after_loss/float(total_after_loss), loss_after_loss/float(total_after_loss)
print win_after_win/float(total_after_win), loss_after_win/float(total_after_win)
print win_after_loss, loss_after_loss, total_after_loss, draw_after_loss
print win_after_win, loss_after_win, total_after_win, draw_after_win

from scipy.stats import binom

2 * binom.cdf(p=0.5, k=1226, n=2475)

2 * binom.cdf(p=0.5, k=1226, n=2475+12)

2 * binom.cdf(p=0.5, k=1545, n=3244+20)

p_draw = 20 / 3581.0
p_win = 0.5 - 0.5 * p_draw
x2 = (1249 - 2487 * p_win)**2 / (2487 * p_win) + (1226 - 2487 * p_win)**2 / (2487 * p_win) + (12 - 2487 * p_draw)**2 / (2487 * p_draw)
x2

from scipy.stats import chisquare
chi2_stat, p_value = chisquare(f_obs=[1249, 1226, 12], f_exp=[2487 * p_win, 2487 * p_win, 2487 * p_draw])
chi2_stat, p_value

chi2_stat, p_value = chisquare([1699, 1545, 20], [3264 * p_win, 3264 * p_win, 3264 * p_draw])
chi2_stat, p_value

fights[fights.Outcome == 'draw'].shape[0]

p_draw*2487

