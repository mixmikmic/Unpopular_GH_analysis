get_ipython().magic('load_ext autotime')

get_ipython().magic('unload_ext autotime')

## Import required modules:
import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

col_names = ['productID', 'title', 'price', 'userID', 'profileName', 
             'helpfulness', 'score', 'time', 'summary', 'text']
books_amazon = pd.read_csv('E://MRP//0102/Books.csv', nrows = 10, encoding = 'utf-8', names=col_names)

books_amazon

## Get all the book-names from Amaozn-data (later this list will be used to find crime-mystery books):
book_names = []
i = 0
with open ('E://MRP//0102/Books.csv') as books:
    for line in books:
        book_names.append(line.split(',')[1])

## Total unique books:
len(set(book_names))

unique_books = list(set(book_names))
unique_books = [book.strip().lower() for book in unique_books]
unique_books.sort()

books_goodreads = pd.read_csv('E:/MRP/0331/Book-Recommender-System/data/books_df.csv', encoding = 'latin1')

books_goodreads['Books'] = [book.strip().lower() for book in books_goodreads['Books']]
books_goodreads['Total_Num_Ratings'] = [int(rating.replace(',','')) for rating in books_goodreads['Total_Num_Ratings']]
books_goodreads['Total_Num_Votes'] = [int(vote.replace(',','')) for vote in books_goodreads['Total_Num_Votes']]
books_goodreads['Avg_Rating'] = [float(rating.strip("['']")) for rating in books_goodreads['Avg_Rating']]

books_goodreads.head()

common_books = list(set(unique_books) & set(books_goodreads['Books']))
print (len(common_books))
common_books[0:10]

temp_book = ['Dr. Seuss', 'Its Only Art If Its Well Hung!']
# temp_books = set(common_books)

list_df = []
with open ('E://MRP//0102//Books.csv') as books:
    for i in range(5):
        line = books.readline()
        book = line.split(',')[1].strip()
        
        if book in temp_book:
            list_df.append(line)

## Load only required columns to avoid memory-issues:
books_amazon_whole = pd.read_csv('E://MRP//0102/Books.csv', encoding = 'utf-8', header=None, usecols=[0,1,3,6])
new_cols = ['BookID', 'BookTitle', 'UserID', 'Score']
books_amazon_whole.columns = new_cols
books_amazon_whole['BookTitle'] = [title.strip().lower() for title in books_amazon_whole['BookTitle']]
books_amazon_whole.head()

## Subset the previous data-frame to keep records for only crime-mystery books:
books_amazon_whole = books_amazon_whole[books_amazon_whole['BookTitle'].isin(common_books)]
books_amazon_whole.head()

## Unique users
len(set(books_amazon_whole['UserID']))

amazon_rating = pd.DataFrame(books_amazon_whole.groupby(by = 'BookTitle').mean()).reset_index(drop = False)

fig, (ax1, ax2) = plt.subplots(figsize= (14,7), ncols = 2, sharey = True)
ax1.hist(amazon_rating['Score'], bins = [1,2,3,4,5], normed = True)
ax1.set_xlabel('Average Rating', fontsize = 15)
ax1.set_ylabel('Frequency (Normalized)', fontsize = 15)
ax1.set_title('Distribution of Average Rating \n (Amazon)', fontsize = 20)

ax2.hist(books_goodreads['Avg_Rating'], bins = [1,2,3,4,5], normed = True)
ax2.set_xlabel('Average Rating', fontsize = 15)
# ax2.set_ylabel('Frequency (Normalized)', fontsize = 15)
ax2.set_title('Distribution of Average Rating \n (Goodreads)', fontsize = 20)

plt.show()

## Consider users who rated >1 books. Two reasons: 1) Expect to get better results; 2) Speeds-up calculation and easy to manage.
filter_users = pd.DataFrame(books_amazon_whole.groupby(by = 'UserID').size(), columns = ['count'])
filter_users = filter_users.loc[filter_users['count'] > 1]
filter_users.reset_index(drop = False, inplace = True)

filter_users = filter_users[filter_users['UserID'] != ' unknown']  ## Remove unknown user-ID which has given 949 ratings
filter_users = books_amazon_whole.loc[books_amazon_whole['UserID'].isin(filter_users['UserID'])]

## Unique users who rated more than 1 books
len(set(filter_users['UserID']))

filter_users.head()

# books_amazon_whole.drop_duplicates(subset = ['BookTitle', 'UserID'], inplace = True)
user_item_df = pd.pivot_table(data = filter_users, index = 'UserID', columns = 'BookTitle', values = 'Score')
user_item_df.head()

## Fill NAs with 0 and get the new list of book-titles:
user_item_df.fillna(0, inplace=True)
new_book_names = user_item_df.columns
user_item_df.head()

user_item_df.shape

true_user_id = list(enumerate(user_item_df.index))
true_user_id[0:10]

## Test mapping of user-ids:
user_0 = pd.DataFrame(user_item_df.loc[' A0134066213WYQXLTVGYT'])
user_0[user_0[' A0134066213WYQXLTVGYT'] > 0]

user_data = [np.count_nonzero(user_item_df.iloc[i,:]) for i in range(user_item_df.shape[0])]
book_data = [np.count_nonzero(user_item_df.iloc[:,i]) for i in range(user_item_df.shape[1])]

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14,6))
ax1.hist(user_data, bins = range(0,15,1))
ax2.hist(book_data, bins = range(0, 2000,50))

ax1.set_title('Distribution of User-Ratings', fontsize = 15)
ax2.set_title('Distribution of Book-Ratings', fontsize = 15)

ax1.set_xlabel('#Books rated by user', fontsize = 15)
ax2.set_xlabel('#Ratings each book received', fontsize = 15)

ax1.set_ylabel('#Users', fontsize = 15)
ax2.set_ylabel('#Books', fontsize = 15)

plt.show()

user_item_mat = csr_matrix(user_item_df)
user_item_mat

# def get_dense_users(row):
#     if np.count_nonzero(user_item_mat[row,:].toarray()) > 1:
#         return row        
    
# subset_matrix_ind = [get_dense_users(row) for row in range(user_item_mat.shape[0])]

book_similarity_mat = cosine_similarity(user_item_mat.transpose(), dense_output = False)
book_similarity_mat = book_similarity_mat.toarray()
book_similarity_mat.shape

def most_similar_books(book_ind, book_names):
    
    most_sim_books = np.argsort(book_similarity_mat[book_ind,:])[::-1][0:20]
    recommended_books = [book_names[i] for i in most_sim_books]
    
    recommendation_df = pd.DataFrame({'BookIndex': most_sim_books, 'BookTitle': recommended_books})
    
    print ('Target book: ', book_names[book_ind])
    
    return recommendation_df

most_similar_books(483, new_book_names)  ## Try: 837, 483

most_similar_books(book_ind = 483, book_names = new_book_names)

import random
from sklearn.decomposition import TruncatedSVD

random.seed(11)

user_ids_all = list(user_item_df.index)

train_ind = random.sample(k = 50000, population = range(58991))
train_user_ids = [user_ids_all[i] for i in train_ind]
train_data = csr_matrix(user_item_df.iloc[train_ind, :])

test_user_ids = [user_ids_all[j] for j in range(len(user_ids_all)) if j not in train_ind]
test_data = csr_matrix(user_item_df.loc[test_user_ids, :])

print ('Shape of training data: ', train_data.shape)
print ('Shape of testing data: ', test_data.shape)

def fit_svd(train, n_compo = 10, random_state = 11):
    
    tsvd = TruncatedSVD(n_components = n_compo, random_state = random_state)
    tsvd.fit(train)
    
    return tsvd

def predict_train(tsvd_obj, train):
    
    train_predictions = np.dot(tsvd_obj.transform(train), tsvd_obj.components_)
    
    return train_predictions

def predict_test(tsvd_obj, test):
    
    test_predictions = np.dot(tsvd_obj.transform(test), tsvd_obj.components_)
        
    return test_predictions

tsvd = fit_svd(train = train_data, n_compo = 25)
predict_ratings_train = predict_train(tsvd_obj = tsvd, train = train_data)
predict_ratings_train.shape

predict_ratings_test = predict_test(tsvd_obj = tsvd, test = test_data)
predict_ratings_test.shape

def plot_rmse(rmse_list, n_users = 500):
    
    fig, ax = plt.subplots(figsize = (14,8))
    ax.plot(rmse_list[0:n_users])
    ax.axhline(y = np.mean(rmse_list), label = 'Avg. RMSE: {}'.format(round(np.mean(rmse_list), 3)), 
               color = 'r', linestyle = 'dashed')
    ax.set_ylabel('RMSE', fontsize = 15)
    ax.set_xlabel('UserId', fontsize = 15)
    ax.set_title('RMSE for each user', fontsize = 20)
    ax.legend()
    plt.show()    
    
    return None

rmse_train = np.sqrt(np.mean((train_data.toarray() - predict_ratings_train)**2, axis = 1))

plot_rmse(rmse_train)

def get_recommended_books(user_id, books_list, latent_ratings, ui_mat, top_n = 15):
    
    ## Get recommendations for a given user:
    ind_top_rated_books = np.argsort(latent_ratings.iloc[user_id])[::-1][0:top_n]
    recommended_books = [books_list[ind] for ind in ind_top_rated_books]    
    recommendation_df = pd.DataFrame({'UserID': user_id, 'BookID': ind_top_rated_books, 
                                     'Recommended_Books': recommended_books})
    
    ## Get actual books that the user rated:
    user_rated_books = ui_mat[user_id,:].toarray()
    rated_books_ind = np.argwhere(user_rated_books != 0)[:,1]
    rated_books = [books_list[ind] for ind in rated_books_ind]
    user_rated_books_df = pd.DataFrame({'BookID': rated_books_ind, 'RatedBooks': rated_books, 'UserID': user_id})
    
    return user_rated_books_df, recommendation_df

## Try: 211
user_rated_books, recommended_books = get_recommended_books(user_id = 211, books_list = new_book_names, 
                                      latent_ratings = predict_ratings, ui_mat = user_item_mat)

rated_books, recommended_books = get_recommended_books(user_id = 100, books_list = new_book_names, 
                                      latent_ratings = pd.DataFrame(predict_ratings_train), ui_mat = train_data)

rated_books

recommended_books

rmse_test = np.sqrt(np.mean((test_data.toarray() - predict_ratings_test)**2, axis = 1)) 
plot_rmse(rmse_test)



