import numpy as np
from tqdm import tqdm
from collections import Counter
L = 4
l = 3
num_simulations = 10
coverage = 0.95




seen = Counter()
for i in tqdm(range(num_simulations)):
    trial = np.random.choice(L, size=l,replace=False)
    #print(trial)
    seen.update(trial)

print((sum(i for i in seen.values())/len(seen))/num_simulations)
'''
total_draws = 0
for i in tqdm(range(num_simulations)):
    seen = set()
    sim_draws=0
    while len(seen)<coverage*L:
        trial = np.random.choice(L, size=l,replace=False)
        #print(trial)
        seen.update(trial)
        sim_draws+=1
    total_draws+=sim_draws
print(total_draws/num_simulations)'''
    