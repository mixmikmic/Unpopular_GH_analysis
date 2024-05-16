import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter
get_ipython().magic('matplotlib notebook')
np.set_printoptions(suppress=True, precision=3)

with open('hon_matches.pkl', 'rb') as db:
    matches = pickle.load(db)
print('Total Matches:', len(matches))

# try removing matches that do not have total 10 players
# or have duplicate heroes
remove_incomplete = False
removes = []
dup_removes = []
for k, v in matches.items():
    ll = len(v['legion'])
    hl = len(v['hellbourne'])
    if remove_incomplete and ll + hl < 10:
        removes.append(k)
    # checking for duplicate heroes match
    lls = len(set(v['legion']))
    hls = len(set(v['hellbourne']))
    if lls < ll or hls < hl:
        dup_removes.append(k)
removes = set(removes)
for key in removes:
    del matches[key]
print('Incomplete Matches Removed:', len(removes))
dup_removes = set(dup_removes) & set(matches.keys())
for key in dup_removes:
    del matches[key]
print('Duplicate-Heroes Matches Removed:', len(dup_removes))
print('Remaining Matches:', len(matches))

X = []
for i, item in enumerate(matches.items()):
    if i == 5:
        break
    print(item)
    X.append(item[1])

AVAIL_HEROES = 260 # actually 134 but extra for future
def vectorize_matches(matches, include_Y=True):
    legion_vec = np.zeros([len(matches), AVAIL_HEROES])
    hellbourne_vec = np.zeros([len(matches), AVAIL_HEROES])
    if include_Y:
        winner = np.zeros([len(matches), 1])
        concede = np.zeros([len(matches), 1])
        secs = np.zeros([len(matches), 1])
    for m, match in enumerate(matches):
        for hero_id in match['legion']:
            legion_vec[m, hero_id] = 1.
        for hero_id in match['hellbourne']:
            hellbourne_vec[m, hero_id] = 1.
        if include_Y:
            if match['winner']:
                winner[m, 0] = 1.
            if match['concedes']:
                concede[m, 0] = 1.
            secs[m, 0] = match['secs']
    x = np.concatenate([legion_vec, hellbourne_vec], axis=1)
    if include_Y:
        y = np.concatenate([winner, concede, secs], axis=1)
    return (x, y) if include_Y else x

X, Y = vectorize_matches(matches.values())
X.shape, Y.shape

with open('heroes_name.pkl', 'rb') as f:
    heroes_dict = pickle.load(f)
heroes_dict[125]

def hero_id_to_name(hero_id):
    return heroes_dict[hero_id]['disp_name']
hero_id_to_name(125)

def hero_name_to_id(name):
    if not name:
        return None
    name = name.lower()
    for id, hero in heroes_dict.items():
        if name in hero['disp_name'].lower():
            return id, hero['disp_name']
    return None
hero_name_to_id('BUB')

from operator import itemgetter
# returns a hero that maximize win probability in a given team
# if 'optimal' is false, it will return a hero that minimize win probability
def optimal_hero_choice(model, match, hellbourne_side=False, as_list=True, as_name=True, optimal=True):
    legion = match['legion']
    hellbourne = match['hellbourne']
    team_ids = hellbourne if hellbourne_side else legion
    hypothesis = []
    for id in set(heroes_dict.keys()) - set(legion + hellbourne): # all choosable hero ids
        team_ids.append(id)
        x = vectorize_matches([match], include_Y=False)
        team_ids.pop()
        p = model.predict(x, verbose=0)[0]
        hero = id
        if as_name:
            hero = hero_id_to_name(hero)
        hypothesis.append((hero, p[0, 1 if hellbourne_side else 0]))
    extrema = max if optimal else min
    return sorted(hypothesis, key=itemgetter(1), reverse=optimal) if as_list else extrema(hypothesis, key=itemgetter(1))

def humanize(xrow):
    legion, hellbourne = [], []
    for i, el in enumerate(xrow):
        if el:
            if i < AVAIL_HEROES:
                name = hero_id_to_name(i)
                legion.append(name)
            else:
                name = hero_id_to_name(i - AVAIL_HEROES)
                hellbourne.append(name)
    return {'legion': legion, 'hellbourne': hellbourne}

# which team won more? Hellbourne!
Counter(Y[:,0]), Y.mean(axis=0)

played = [] # played heroes
for i, item in enumerate(matches.items()):
    v = item[1]
    if 0 in v['legion']:
        print(item)
    if 0 in v['hellbourne']:
        print(item)
    played.extend(v['legion'])
    played.extend(v['hellbourne'])
len(played)

Counter(played).most_common()

[(hero_id_to_name(id), freq) for (id, freq) in Counter(played).most_common() if id != 0]

# how many players are there in a game?
players = Counter(X.sum(axis=1))
players

plt.plot(list(players.keys()), list(players.values()))

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.91)
X_train.shape, Y_test.shape

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, ELU, Reshape, Convolution2D, Flatten, Permute, BatchNormalization, Input
from keras.regularizers import l2, activity_l2
hero_input = Input(shape=[2 * AVAIL_HEROES], name='hero_input')
h = Reshape([1, 2, AVAIL_HEROES], input_shape=[AVAIL_HEROES*2,])(hero_input)
h = Permute([2, 3, 1])(h)
h = Convolution2D(135, 1, AVAIL_HEROES, border_mode='valid')(h) # learn to represent 135 heroes from 260-d vector
# h = ELU()(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)
h = Convolution2D(64, 1, 1, border_mode='valid')(h)
# h = ELU()(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)
h = Flatten()(h)
h = Dense(128)(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)
h = Dropout(0.5)(h)
h = Dense(128)(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)
h = Dropout(0.5)(h)
h = Dense(128)(h)
h = BatchNormalization()(h)
h = Activation('relu')(h)
logit = Dropout(0.5)(h)
out_winner = Dense(output_dim=2, activation='softmax', name='out_winner')(logit) # 0 for legion and 1 for hellbourne team
out_concede = Dense(output_dim=1, activation='sigmoid', name='out_concede')(logit) # would the loser concede ?
out_secs = Dense(output_dim=1, name='out_secs')(logit) # how many seconds would the game last ?
model = Model([hero_input], [out_winner, out_concede, out_secs])
loss = ['sparse_categorical_crossentropy', 'binary_crossentropy', 'mean_squared_error']
metrics = {
    'out_winner': 'accuracy',
    'out_concede': 'accuracy',
    'out_secs': 'mean_absolute_error'
}
loss_weights = {
    'out_winner': 2.,
    'out_concede': 1.,
    'out_secs': 1./600000 # mean squared loss is so high we have to penalize it, otherwise, it would steal all computations
}
model.compile(loss=loss, optimizer='adam', metrics=metrics, loss_weights=loss_weights)
hist = model.fit(X_train, np.split(Y_train, 3, axis=1), batch_size=32, nb_epoch=10, verbose=1, validation_split=0.075)

model.summary()
model.save('honnet_brain.h5')

loss_and_metrics = model.evaluate(X_test, np.split(Y_test, 3, axis=1), batch_size=32, verbose=1)
loss_and_metrics

from collections import Counter
Counter(Y[:, 0]), Counter(Y[:, 1]) # win team and concede counts

plt.hist(Y[:, -1], bins=50) # lasting time in secs

pred = model.predict(X_test[:10, :])
# prediction is [legion_win_prob, hellbourne_win_prob, loser_concede_prob, estimated_game_time_in_secs]
np.concatenate(pred, 1), Y_train[:10, :]

humanize(X_test[0])

def inputName():
    name = input('Hero Name: ')
    hero_id, hero_name = hero_name_to_id(name)
    return hero_id, hero_name

legion = []
hellbourne = []
hellbourne_bool = 0
match = [{'legion': legion, 'hellbourne': hellbourne}]

hellbourne_bool = not hellbourne_bool
'Hellbourne' if hellbourne_bool else 'Legion'

hero_id, hero_name = inputName()
if hero_id is not None:
    print('Hero:', hero_name)
    if hellbourne_bool:
        hellbourne.append(hero_id)
    else:
        legion.append(hero_id)
    x = vectorize_matches(match, include_Y=False)
    print('Team:', humanize(x[0]))
    proba = model.predict(x, verbose=0)
    print('Proba:', np.concatenate(proba, axis=1))

# selecting an optimal hero for the current team
choice = optimal_hero_choice(model, match[0], hellbourne_side=hellbourne_bool, as_list=False, as_name=False, optimal=True)
print(choice, hero_id_to_name(choice[0]))
team_ids = hellbourne if hellbourne_bool else legion
team_ids.append(choice[0])
match = [{'legion': legion, 'hellbourne': hellbourne}]
x = vectorize_matches(match, include_Y=False)
print('Team:', humanize(x[0]))
proba = model.predict(x, verbose=0)
print('Proba:', np.concatenate(proba, axis=1))

