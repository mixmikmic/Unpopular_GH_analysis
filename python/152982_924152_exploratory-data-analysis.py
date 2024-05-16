import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from wordcloud import WordCloud

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = (5,3)

data = pd.read_pickle('data/data.pickle')
data.columns = ['drug_name','symptoms','label','severity_label']

data.describe()

data.severity_label.value_counts()

data = data[data['severity_label'].isin(['doctor','emergency','noneed'])]

def plot(df):

    plot_df = df.value_counts(normalize = True).apply(lambda x: x*100)
    
    recs = plot_df.plot(kind='barh')
    

plot(data['severity_label'])

symptoms = data['symptoms'].str.cat(sep=', ')

wordcloud = WordCloud(max_font_size=50).generate(symptoms)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title('A wordcloud of symptoms')
plt.show()

symp_list = ['stomach','pain','abdominal','tiredness','weakness']

data['temp1'] = data['symptoms'].apply(lambda x: 1 if any(symp in x for symp in symp_list) else 0)

data.temp1.value_counts(normalize = True)





