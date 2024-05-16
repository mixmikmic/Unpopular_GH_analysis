import random
random.seed(1234)

trials = 10000
wins = 0
die = range(1, 17)
for _ in xrange(trials):
    for _ in range(16):
        if (random.choice(die) == 1):
            wins += 1
            break
print wins / float(trials)

p = 1 / 16.0
q = 15 / 16.0
sum([p * q**(k - 1) for k in range(1, 17)])

from scipy.stats import geom
geom.cdf(p=1/16.0, k=16)

1.0 - (15 / 16.0)**16

1.0 - (15 / 16.0)**11

trials = 100000
wins1 = 0
wins1_attempt = 0
wins2 = 0
wins2_attempt = 0
last_four = ['null', 'null', 'null', 'null']

for i in xrange(trials):
    outcome = random.choice(['heads', 'tails'])
    
    # gambler 1
    if (i % 5 == 0):
        if (outcome == 'heads'):
            wins1 += 1
        wins1_attempt += 1
        
    # gambler 2
    if (last_four == ['tails', 'tails', 'tails', 'tails']):
        if (outcome == 'heads'):
            wins2 += 1
        wins2_attempt += 1
        
    last_four.insert(0, outcome)
    _ = last_four.pop()
        
print wins1 / float(wins1_attempt)
print wins2 / float(wins2_attempt)

