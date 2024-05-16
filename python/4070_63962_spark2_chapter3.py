lines = sc.textFile('text_file.md')
print lines.count()
print lines.first()

print lines.take(5)

print lines.collect()

lines.sample(withReplacement=False, fraction=0.1).collect()

plines = lines.filter(lambda x: 'Python' in x or 'Spark' in x)
print plines.count()

chars = lines.map(lambda x: len(x))
print chars.take(5)

small = sc.parallelize(['dog', 'fish', 'cat', 'mouse'])
small_and_keys = small.union(plines)
print small_and_keys.collect()

small_and_ints = small.union(chars)
print small_and_ints.take(20)

print chars.count(), chars.distinct().count()

# find the maximum
print chars.reduce(lambda x, y: x if x > y else y), chars.max()

print chars.collect()

pairs = lines.flatMap(lambda x: x.split()).map(lambda x: (x, 1))
print pairs.take(5)

# note that we change types here from int to string
trans = chars.map(lambda x: 'dog' if x > 10 else 'cat')
print trans.take(5)

print chars.countByValue()

print chars.top(5)

# note that persist does not force evaluation
chars.persist(StorageLevel.DISK_ONLY_2)

from random import random as rng
import Player

players = []
for i in range(100):
    players.append(Player.Player(rng(), i))
players[0].talk()
print players[0].name

#sc.addPyFile('Player.py')
rdd = sc.parallelize(players)
print rdd.count()

best = rdd.filter(lambda p: p.x > 0.9)
print best.count()

