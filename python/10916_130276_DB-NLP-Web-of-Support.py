import pandas as pd
pd.set_option('display.max_rows', 6)

candidates = pd.DataFrame.from_csv('public.raw_candidate_filings.csv').sort_values(by='last_name')  # id_nmbr
candidates[[col for col in candidates.columns if 'name' in col or 'id' in col]]

pacs = pd.DataFrame.from_csv('public.raw_committees.csv').sort_values(by='committee_name')  # no ID that I can find
pacs[[col for col in pacs.columns if 'name' in col or 'id' in col or 'type' in col]]

pacs_scraped = pd.DataFrame.from_csv('public.raw_committees_scraped.csv', index_col='id').sort_values(by='name')  # id
pacs_scraped

candidates

# Thanks
pd.set_option("display.max_rows",101)

get_ipython().system('pip install pystemmer')
import Stemmer
english_stemmer = Stemmer.Stemmer('en')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: english_stemmer.stemWords(analyzer(doc))
     
tfidf = StemmedTfidfVectorizer(min_df=1, stop_words='english', analyzer='word', ngram_range=(1, 2))

