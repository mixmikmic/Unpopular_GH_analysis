get_ipython().magic('load_ext autotime')

import gzip
import csv

col_names = ['productID', 'title', 'price', 'userID', 'profileName', 
             'helpfulness', 'score', 'time', 'summary', 'text']

def process_data(n):
    
    """
    The function reads the txt.gz file, which contains all the reviews, line-by-line, and when it 
    gathers all the data for a sinngle review, it writes (or adds) a single row for that review in a 
    csv file, and does this for all ~13M reviews.
    
    Estimated completion time for ~13M reviews: 20 min
    Estimated size of file for ~13M reviews: 10 GB 
    
    Args:
        n (int): #rows to read, provide n as a multiple of 11.
                    If n = 1100, then fetches data for 1100/11 = 100 users.
                    
    Returns:
        None
    
    """
    
    cnt = 0
    temp = []
    
    with gzip.open('E:\\MRP\\0102\\Books.txt.gz') as rf:
        with open('E:\\MRP\\0102\\Books.csv', 'w', newline = '') as cw:
            csv_writer = csv.writer(cw, delimiter = ',')
            
            while cnt < n:
                l1 = rf.readline().decode('utf-8')
                
                if len(l1) > 1:                
                    value = l1.split(':')[1].strip('\n')                  
                    temp.append(value)
                else:
                    csv_writer.writerow(temp)
                    temp = []

                cnt+=1          
            
    return None

all_records = process_data(141751368)

sample = pd.read_csv('E://MRP//0102//Books.csv', nrows = 500, names=col_names)

sample.head()

pd.pivot_table(data = sample, index = 'userID', columns = 'title', values = 'score')



