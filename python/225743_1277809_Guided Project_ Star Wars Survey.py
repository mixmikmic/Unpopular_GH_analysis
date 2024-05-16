import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
star_wars = pd.read_csv('star_wars.csv', encoding='ISO-8859-1')
star_wars.head(3)

print(star_wars.shape)
star_wars.columns

star_wars = star_wars[star_wars['RespondentID'].notnull()]


yes_no = {"Yes": True, "No": False, np.nan:False}

for col in [
    "Have you seen any of the 6 films in the Star Wars franchise?",
    "Do you consider yourself to be a fan of the Star Wars film franchise?"
    ]:
    star_wars[col] = star_wars[col].map(yes_no)

star_wars.head()

true_false = {
    "Star Wars: Episode I  The Phantom Menace": True,
    "Star Wars: Episode II  Attack of the Clones": True,
    "Star Wars: Episode III  Revenge of the Sith": True,
    "Star Wars: Episode IV  A New Hope": True,
    "Star Wars: Episode V The Empire Strikes Back": True,
    "Star Wars: Episode VI Return of the Jedi": True,
    np.nan: False,
}
for col in star_wars.columns[3:9]:
    star_wars[col] = star_wars[col].map(true_false)
    
star_wars.head()

#Change the column names with the .rename() method
star_wars = star_wars.rename(columns={
    'Which of the following Star Wars films have you seen? Please select all that apply.': "seen_1",
    "Unnamed: 4": "seen_2",
    "Unnamed: 5": "seen_3",
    "Unnamed: 6": "seen_4",
    "Unnamed: 7": "seen_5",
    "Unnamed: 8": "seen_6",
    })

star_wars.columns

star_wars.dtypes

star_wars[star_wars.columns[9:15]] = star_wars[star_wars.columns[9:15]].astype(float)

star_wars = star_wars.rename(columns={
    'Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.': "ranking_1",
    "Unnamed: 10": "ranking_2",
    "Unnamed: 11": "ranking_3",
    "Unnamed: 12": "ranking_4",
    "Unnamed: 13": "ranking_5",
    "Unnamed: 14": "ranking_6",
    })

star_wars.columns

means = star_wars[star_wars.columns[9:15]].mean()
get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(range(1,7), means)
plt.xlabel("Movie #")
plt.ylabel('Average Ranking')

sums = star_wars[star_wars.columns[3:9]].sum()
plt.bar(range(1,7), sums)
plt.xlabel("Movie #")
plt.ylabel('Total Respondants')

star_wars_males = males = star_wars[star_wars["Gender"] == "Male"]
star_wars_females = females = star_wars[star_wars["Gender"] == "Female"]

means_males = star_wars_males[star_wars_males.columns[9:15]].mean()
plt.bar(range(1,7), means_males)
plt.xlabel("Movie #")
plt.ylabel('Average Ranking')
plt.show()


means_females = star_wars_females[star_wars_females.columns[9:15]].mean()
plt.bar(range(1,7), means_females)
plt.xlabel("Movie #")
plt.ylabel('Average Ranking')
plt.show()

sums_males = star_wars_males[star_wars_males.columns[3:9]].sum()
plt.bar(range(1, 7), sums_males)
plt.xlabel("Movie #")
plt.ylabel('Total Respondants')
plt.show()

sums_females = star_wars_females[star_wars_females.columns[3:9]].sum()
plt.bar(range(1, 7), sums_females)
plt.xlabel("Movie #")
plt.ylabel('Total Respondants')
plt.show()



