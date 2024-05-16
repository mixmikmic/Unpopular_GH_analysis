get_ipython().magic('run helper_functions.py')

lst = unpickle_object("2nd_degree_connections_LDA_complete.pkl")

lst[8] #an example of a bad handle dictionary

handle_names = []
for dictionary in lst:
    name = list(dictionary.keys())
    handle_names.append(name)

handle_names = sum(handle_names, [])

#an example of me finding which user's in my LDA results tweet about "machine" --> alluding to "machine learning"
cnt = -1

for handle in handle_names:
    cnt +=1
    try:
        topics = lst[cnt][handle]['LDA']
        
        if "machine" in topics:
            print(handle)
    except:
        pass

# handles to be removed as they do not have valid LDA analysis
handle_to_remove = []
cnt = -1

for handle in handle_names:
    cnt += 1
    sub_dict = lst[cnt][handle]
    
    if "LDA" not in sub_dict:
        handle_to_remove.append(handle)

indicies = []

for handle in handle_to_remove:
    index = handle_names.index(handle)
    indicies.append(index)

#extracting the valid LDA handle
verified_handles_lda = [v for i,v in enumerate(handle_names) if i not in frozenset(indicies)]

handle_to_remove[:5] #a peek at the 'bad handles'

pickle_object(verified_handles_lda, "verified_handles_lda")

pickle_object(handle_to_remove, "LDA_identified_bad_handles")

#extracting the appropriate dictionaries to be used in TF-IDF analysis
final_database_lda_verified = [v for i,v in enumerate(lst) if i not in frozenset(indicies)]

pickle_object(final_database_lda_verified, "final_database_lda_verified")



