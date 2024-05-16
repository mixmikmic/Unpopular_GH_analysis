from bs4 import BeautifulSoup as bs
import pandas as pd
import re
import urllib
import time

p1 = 'https://www.goodreads.com/list/show/11.Best_Crime_Mystery_Books?page='
page_id = [str(n) for n in range(1,51)]

books_all = []  ## Store names of all the books from all the pages
meta_ratings = [] ## Store ratings for all the books
meta_votes = []  ## Store users' votes for all the books

for i in page_id:
    link = p1+i
    print ('page'+str(i))
    
    page = urllib.request.urlopen(url = link)
    soup = bs(page, 'lxml')
    time.sleep(2)

    ## Get book title:
    book_titles = []
    title = soup.find_all(name = 'a', class_ = 'bookTitle')
    for t in title:
        book_titles.append(t.span.text)
  
    ## Get metadata-1:
    metadata_ratings = []
    metadata = soup.find_all(name = 'span', class_ = 'minirating')
    for meta in metadata:
        metadata_ratings.append(meta.text.strip())

    ## Get metadata-2:
    metadata_score = []
    metadata_votes = []
    metadata_2 = soup.find_all(name = 'span', class_ = 'smallText uitext')
    for meta in metadata_2:
        temp = meta.text.strip('\n')
        score = temp.split('\n')[0]
        metadata_score.append(score)

        num_votes = temp.split('\n')[2]
        metadata_votes.append(num_votes)        
    
    books_all.extend(book_titles)
    meta_ratings.extend(metadata_ratings)
    meta_votes.extend(metadata_votes)

print ('Total books getched:', len(books_all))
print ('Total ratings fetched:', len(meta_ratings))
print ('Total votes fetched:', len(meta_votes))

## For ratings:
avg_ratings = []
total_ratings = []

for rat in meta_ratings:
    avg_ratings.append(re.findall('([0-9]+[.]+[0-9]+)', rat))
    total_ratings.append(rat.split(' ')[-2])

## For votes:
total_votes = []

for vote in meta_votes:
    total_votes.append(vote.split(' ')[0])

books_df = pd.DataFrame({'Books': books_all, 'Avg_Rating': avg_ratings, 
                          'Total_Num_Ratings': total_ratings, 'Total_Num_Votes': total_votes})

books_df.head()

books_df.shape

books_df.to_csv('books_df.csv', index=False)

books_df = pd.read_csv('../data/books_df.csv', encoding='latin1')
books_df.head()



