"""

alpha = a parameter used to make the algorithm more or less greedy (higher is more greedy)
N = number of candidate campaigns
L = list of scored, candidate campaigns
O = output, re-ordered, scored, candidate campaigns

scaledL = for each campaign, raise the eCPM to eCPM ^ alpha

normL = normalize the list of scaledL (where we devide each eCPM ^ alpha by the sum of them all)

cumNormL = perform a cumulative sum from the start to the end of the 
            normalized list of scored, candidate campaigns 
            (the first campaigns cumEcpm will be 0, while the last will be 1 - it's normed ecpm)

while (N > 0) do
    compute normL from current elements of L
    compute cumNorm from normL
    i = rand(0,1)
    for j in range 0 to length of cumNormL:
        if i > cumNormL(j):
            append cumNormL(j) to the tail of O
            remove the jth element from L
            N -= 1
            break
        else:
            continue
"""

from collections import namedtuple
from random import random as rand

Campaign = namedtuple('Campaign', ['id', 'ecpm'])

class Sampler:
    
    def __init__(self, campaigns, alpha = 1.0):
        self.remaining_campaigns = {}
        self.total_sum = 0
        for c in campaigns:
            scaled_ecpm = c.ecpm ** alpha
            self.ramaining_campaigns[c.id] = (scaled_ecpm, c)
            self.total_sum += scaled_ecpm
    
    def size(self):
        return len(self.ramaining_campaigns)
    
    def rand(self):
        return rand() * self.total_sum
    
    def get_next_sample_id(self):
        lastCumSum = 0.0
        r = this.rand()
        for cid, t in self.ramaining_campaigns.items():
            next_cum_sum = lastCumSum + t[0]
            if next_cum_sum >= r:
                return cid
            else:
                lastCumSum = next_cum_sum
        
    def pop(self, cid):
        self.totalSum -= self.remaining_campaigns[cid][0]
        return self.remaining_campaigns.pop(cid)
    
    def get_next_sample(self):
        if self.size() > 0:
            next_cid = get_next_sample_id()
            return pop(next_cid)
        else:
            return None



