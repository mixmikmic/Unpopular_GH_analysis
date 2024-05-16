import random as random
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(font_scale=1.5)

winrate_stay = []
winrate_change = []
wins_stay = 0
wins_change = 0
random.seed(0)
for i in range(10000):
    #Initialize cars/goats setup
    doors = ['Car', 'Goat_1', 'Goat_2']
    random.shuffle(doors)
    
    #Player makes a guess
    first_guess = random.choice(doors)
    
    #Monty opens a door
    if first_guess == 'Goat_2':
        monty_opens = 'Goat_1'
    elif first_guess == 'Goat_1':
        monty_opens = 'Goat_2'
    else:
        monty_opens = random.choice(['Goat_1', 'Goat_2'])                                   
        #Adds one wins if Player stays with the first choice
        wins_stay += 1
    
    #Switch doors
    second_guess = doors
    second_guess.remove(monty_opens)
    second_guess.remove(first_guess)
    
    #Adds one win if player stays with the second choice                             
    if second_guess == ['Car']:
        wins_change += 1
        
    winrate_stay.append(wins_stay*100/(i+1))
    winrate_change.append(wins_change*100/(i+1))
    
print('Win rate (don\'t change): {} %'.format(wins_stay/100))
print('Win rate (changed doors): {} %'.format(wins_change/100))

fig = plt.figure(figsize=(10,5))
ax = plt.plot(winrate_stay, label='Stayed with first choice')
ax = plt.plot(winrate_change, label='Changed doors')

plt.xlim(0,200)
plt.xlabel('Number of Simulations')
plt.ylabel('Win rate (%)')
plt.legend()
plt.show()



