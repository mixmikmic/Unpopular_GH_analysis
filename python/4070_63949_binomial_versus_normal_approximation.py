from scipy.stats import binom
1.0 - binom.cdf(k=59, n=100, p=0.5)

sum([binom.pmf(k, n=100, p=0.5) for k in range(60, 101)])

mu = 50.0
s = (100 * 0.5 * (1 - 0.5))**0.5

from scipy.stats import norm
1.0 - norm.cdf(59.5, loc=mu, scale=s)

from random import choice

trials = 10000
success = 0
data = []
heads_cutoff = 60
n = 100
for _ in xrange(trials):
    heads = 0
    for _ in xrange(n):
        if (choice(["heads", "tails"]) == "heads"):
            heads += 1
    if (heads >= heads_cutoff):
        success += 1
    data.append(heads)
print(float(success) / trials)

# binomial
x1 = range(0, 101)
y1 = binom.pmf(x1, n=100, p=0.5)
plt.bar(x1, y1, align='center', label='Binomial')

# normal
mu = 50.0
s = (100 * 0.5 * (1 - 0.5))**0.5
x = np.linspace(0, 100, num=101)
y = norm.pdf(x, loc=mu, scale=s)
lines = plt.plot(x, y, 'r-', lw=2, label='Normal')

plt.xlim(20, 80)
plt.xlabel('Number of Heads')
plt.ylabel('Probability')
plt.legend()

