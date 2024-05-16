# Read in the list of 250 movies, making sure to remove commas from their names
# (actually, if it has commas, it will be read in as different fields)
import csv
movies = []
with open('movies.csv','r') as csvfile:
    myreader = csv.reader(csvfile)
    for index, row in enumerate(myreader):
        movies.append( ' '.join(row) ) # the join() call merges all fields
# We might like to split this into two tasks, one for movies pre-1980 and one for post-1980, 
import re  # used for "regular-expressions", a method of searching strings
cutoffYear = 1980
oldMovies = []
newMovies = []
for mv in movies:
    sp = re.split(r'[()]',mv)
    #print sp  # output looks like: ['Kill Bill: Vol. 2 ', '2004', '']
    year = int(sp[1])
    if year < cutoffYear:
        oldMovies.append( mv )
    else:
        newMovies.append( mv )
print("Found", len(newMovies), "new movies (after 1980) and", len(oldMovies), "old movies")
# and for simplicity, let's just rename "newMovies" to "movies"
movies = newMovies

# Make a dictionary that will help us convert movie titles to numbers
Movie2index = {}
for ind, mv in enumerate(movies):
    Movie2index[mv] = ind
# sample usage:
print('The movie  ', movies[3],'  has index', Movie2index[movies[3]])

# Read in the list of 60 questions
AllQuestions = []
with open('questions60.csv', 'r') as csvfile:
    myreader = csv.reader(csvfile)
    for row in myreader:
        # the rstrip() removes blanks
        AllQuestions.append( row[0].rstrip() )
print('Found', len(AllQuestions), 'questions')
questions = list(set(AllQuestions))
print('Found', len(questions), 'unique questions')

# As we did for movies, make a dictionary to convert questions to numbers
Question2index = {}
for index,quest in enumerate( questions ):
    Question2index[quest] = index
# sample usage:
print('The question  ', questions[40],'  has index', Question2index[questions[40]])

YesNoDict = { "Yes": 1, "No": -1, "Unsure": 0, "": 0 }
# load from csv files
X = []
y = []
with open('MechanicalTurkResults_149movies_X.csv','r') as csvfile:
    myreader = csv.reader(csvfile)
    for row in myreader:
        X.append( list(map(int,row)) )
with open('MechanicalTurkResults_149movies_y.csv','r') as csvfile:
    myreader = csv.reader(csvfile)
    for row in myreader:
        y = list(map(int,row))

from sklearn import tree
# the rest is up to you

# up to you

