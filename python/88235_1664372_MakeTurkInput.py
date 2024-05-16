import csv

# string info: http://www.openbookproject.net/books/bpp4awd/ch03.html
# [lists] are mutable, (tuples) and 'strings' are not

# Read in the list of 100 questions, putting it into 25 groups of 4
questions = []
rowQuestions= []
with open('questions.csv', 'rb') as csvfile:
    myreader = csv.reader(csvfile)
    for index,row in enumerate(myreader):
        rowQuestions.append( row[0].rstrip() )
        if index%4 is 3:
            #print index, ' '.join(row)
            #print index, rowQuestions
            questions.append( rowQuestions )
            rowQuestions = []
len(questions)

# Read in the list of 250 movies, making sure to remove commas from their names
# (actually, if it has commas, it will be read in as different fields)
movies = []
with open('movies.csv','rb') as csvfile:
    myreader = csv.reader(csvfile)
    for index, row in enumerate(myreader):
        movies.append( ' '.join(row) ) # the join() call merges all fields

N = len(movies)
with open('input.csv', 'wb') as csvfile:
    mywriter = csv.writer(csvfile)
    mywriter.writerow( ['MOVIE','QUESTION1','QUESTION2','QUESTION3','QUESTION4'])
    for i in range(5):
        for q in questions:
            mywriter.writerow( [movies[i], q[0], q[1], q[2], q[3] ])
            #mywriter.writerow( [movies[i]+','+','.join(q)] ) # has extra " "

with open('Batch_2832525_batch_results.csv', 'rb') as csvfile:
    myreader = csv.DictReader(csvfile)
    #myreader = csv.reader(csvfile)
    # see dir(myreader) to list available methods
    for row in myreader:
        #print row
        print row['Input.MOVIE'] +": " + row['Input.QUESTION1'] , row['Answer.MovieAnswer1']
        print '               ' + row['Input.QUESTION2'] , row['Answer.MovieAnswer2']
        print '               ' + row['Input.QUESTION3'] , row['Answer.MovieAnswer3']
        print '               ' + row['Input.QUESTION4'] , row['Answer.MovieAnswer4']

#import os
cwd = os.getcwd()
print cwd
#dir(myreader)
myreader.line_num
#row
#row['Input.QUESTION1']



