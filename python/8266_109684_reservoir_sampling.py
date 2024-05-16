from random import randint

class ReservoirSampler:
    
    def __init__(self, sample_size):
        self.samples = []
        self.sample_size = sample_size
        self.seen_values = 0
    
    def is_full(self):
        return len(self.samples) == self.sample_size
    
    def update_sample(self, new_value):
        self.seen_values += 1
        if not self.is_full():
            # collect the first k
            self.samples.append(new_value)
        else:
            # select value, j, between (0,n-1)
            # it has a k/n probability of being in range (0,k-1)
            # if it is, replace samples[j] with the new value
            j = randint(0,self.seen_values)
            if j < self.sample_size:
                self.samples[j] = new_value
        

r_sampler = ReservoirSampler(25)
for i in range(1000):
    r_sampler.update_sample(i)

print(r_sampler.samples)



