import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('halverson')
get_ipython().magic('matplotlib inline')

iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
d = {'Women\'s Featherweight':'W--FW', 'Middleweight':'MW', 'Lightweight':'LW', 'Bantamweight':'BW'}
d['Women\'s Bantamweight'] = 'W--BW'
d['Women\'s Strawweight'] = 'W--SW'
d['Light Heavyweight'] = 'LHW'
d['Flyweight'] = 'FLW'
d['Featherweight'] = 'FTW'
d['Welterweight'] = 'WW'
d['Heavyweight'] = 'HW'
d['Women\'s Flyweight'] = 'W--FLW'
d['Catch Weight'] = 'CTH'
fights.WeightClass = fights.WeightClass.replace(d)
d = {'no contest':'NC'}
fights.Outcome = fights.Outcome.replace(d)
fights.Event = fights.Event.str.replace('The Ultimate Fighter', 'TUF')
fights.Event = fights.Event.str.replace('Fight Night', 'F. Night')
fights.MethodNotes = fights.MethodNotes.str.replace('Rear Naked Choke', 'RNC')
fights.MethodNotes = fights.MethodNotes.str.replace('Spinning Back', 'Spn. Bck.')
fights.columns = ['Winner', 'Out', 'Loser', 'WC', 'Method', 'Notes', 'Rd', 'Time', 'Event', 'Date', 'Location']
cols = ['Winner', 'Out', 'Loser', 'WC', 'Method', 'Notes', 'Rd', 'Time', 'Event', 'Date']
fights.Event = fights.Event.apply(lambda x: x[:x.index(':')] if ':' in x else x)
fights[cols].head(476).to_latex('fights_table.tex', index=False, na_rep='', longtable=True)

iofile = 'data/fightmetric_fighters_with_corrections_from_UFC_Wikipedia_CLEAN.csv'
fighters = pd.read_csv(iofile, header=0, parse_dates=['Dob'])
fighters['Age'] = (pd.to_datetime('today') - fighters.Dob) / np.timedelta64(1, 'Y')
fighters.Age = fighters.Age.apply(lambda x: x if pd.isnull(x) else round(x, 1))
fighters.head(480).to_latex('fm_table.tex', index=False, na_rep='', longtable=True)

