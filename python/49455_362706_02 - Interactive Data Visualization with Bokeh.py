# Science Stack 
import numpy as np
import pandas as pd

# Bokeh Essentials 
from bokeh.io import output_notebook
from bokeh.plotting import figure, show, ColumnDataSource

# Layouts 
from bokeh.layouts import layout
from bokeh.layouts import widgetbox

# Figure interaction layer
from bokeh.io import show
from bokeh.io import output_notebook 

# Data models for visualization 
from bokeh.models import Text
from bokeh.models import Plot
from bokeh.models import Slider
from bokeh.models import Circle
from bokeh.models import Range1d
from bokeh.models import CustomJS
from bokeh.models import HoverTool
from bokeh.models import LinearAxis
from bokeh.models import ColumnDataSource
from bokeh.models import SingleIntervalTicker

# Palettes and colors
from bokeh.palettes import brewer
from bokeh.palettes import Spectral6

# Load Bokeh for visualization
output_notebook()

import bokeh.sampledata
bokeh.sampledata.download()

def process_data():
    
    # Import the gap minder data sets
    from bokeh.sampledata.gapminder import fertility, life_expectancy, population, regions
    
    # The columns are currently string values for each year, 
    # make them ints for data processing and visualization.
    columns = list(fertility.columns)
    years = list(range(int(columns[0]), int(columns[-1])))
    rename_dict = dict(zip(columns, years))
    
    # Apply the integer year columna names to the data sets. 
    fertility = fertility.rename(columns=rename_dict)
    life_expectancy = life_expectancy.rename(columns=rename_dict)
    population = population.rename(columns=rename_dict)
    regions = regions.rename(columns=rename_dict)

    # Turn population into bubble sizes. Use min_size and factor to tweak.
    scale_factor = 200
    population_size = np.sqrt(population / np.pi) / scale_factor
    min_size = 3
    population_size = population_size.where(population_size >= min_size).fillna(min_size)

    # Use pandas categories and categorize & color the regions
    regions.Group = regions.Group.astype('category')
    regions_list = list(regions.Group.cat.categories)

    def get_color(r):
        return Spectral6[regions_list.index(r.Group)]
    regions['region_color'] = regions.apply(get_color, axis=1)

    return fertility, life_expectancy, population_size, regions, years, regions_list

# Process the data and fetch the data frames and lists 
fertility_df, life_expectancy_df, population_df_size, regions_df, years, regions = process_data()

# Create a data source dictionary whose keys are prefixed years
# and whose values are ColumnDataSource objects that merge the 
# various per-year values from each data frame. 
sources = {}

# Quick helper variables 
region_color = regions_df['region_color']
region_color.name = 'region_color'

# Create a source for each year. 
for year in years:
    # Extract the fertility for each country for this year.
    fertility = fertility_df[year]
    fertility.name = 'fertility'
    
    # Extract life expectancy for each country for this year. 
    life = life_expectancy_df[year]
    life.name = 'life' 
    
    # Extract the normalized population size for each country for this year. 
    population = population_df_size[year]
    population.name = 'population' 
    
    # Create a dataframe from our extraction and add to our sources 
    new_df = pd.concat([fertility, life, population, region_color], axis=1)
    sources['_' + str(year)] = ColumnDataSource(new_df)

sources

dictionary_of_sources = dict(zip([x for x in years], ['_%s' % x for x in years]))

js_source_array = str(dictionary_of_sources).replace("'", "")
js_source_array

xdr = Range1d(1, 9)
ydr = Range1d(20, 100)

plot = Plot(
    x_range=xdr,
    y_range=ydr,
    plot_width=800,
    plot_height=400,
    outline_line_color=None,
    toolbar_location=None, 
    min_border=20,
)

# show(plot)

# Create a dictionary of our common setting. 
AXIS_FORMATS = dict(
    minor_tick_in=None,
    minor_tick_out=None,
    major_tick_in=None,
    major_label_text_font_size="10pt",
    major_label_text_font_style="normal",
    axis_label_text_font_size="10pt",

    axis_line_color='#AAAAAA',
    major_tick_line_color='#AAAAAA',
    major_label_text_color='#666666',

    major_tick_line_cap="round",
    axis_line_cap="round",
    axis_line_width=1,
    major_tick_line_width=1,
)


# Create two axis models for the x and y axes. 
xaxis = LinearAxis(
    ticker=SingleIntervalTicker(interval=1), 
    axis_label="Children per woman (total fertility)", 
    **AXIS_FORMATS
)

yaxis = LinearAxis(
    ticker=SingleIntervalTicker(interval=20), 
    axis_label="Life expectancy at birth (years)", 
    **AXIS_FORMATS
)   

# Add the axes to the plot in the specified positions.
plot.add_layout(xaxis, 'below')
plot.add_layout(yaxis, 'left')

# show(plot)

# Create a data source for each of our years to display. 
text_source = ColumnDataSource({'year': ['%s' % years[0]]})

# Create a text object model and add to the figure. 
text = Text(x=2, y=35, text='year', text_font_size='150pt', text_color='#EEEEEE')
plot.add_glyph(text_source, text)

# show(plot)

# Select the source for the first year we have. 
renderer_source = sources['_%s' % years[0]]

# Create a circle glyph to generate points for the scatter plot. 
circle_glyph = Circle(
    x='fertility', y='life', size='population',
    fill_color='region_color', fill_alpha=0.8, 
    line_color='#7c7e71', line_width=0.5, line_alpha=0.5
)

# Connect the glyph generator to the data source and add to the plot
circle_renderer = plot.add_glyph(renderer_source, circle_glyph)

# Add the hover (only against the circle and not other plot elements)
tooltips = "@index"
plot.add_tools(HoverTool(tooltips=tooltips, renderers=[circle_renderer]))

# show(plot)

# Position of the legend 
text_x = 7
text_y = 95

# For each region, add a circle with the color and text. 
for i, region in enumerate(regions):
    plot.add_glyph(Text(x=text_x, y=text_y, text=[region], text_font_size='10pt', text_color='#666666'))
    plot.add_glyph(
        Circle(x=text_x - 0.1, y=text_y + 2, fill_color=Spectral6[i], size=10, line_color=None, fill_alpha=0.8)
    )
    
    # Move the y coordinate down a bit.
    text_y = text_y - 5

# show(plot)

# Add the slider
code = """
    var year = slider.get('value'),
        sources = %s,
        new_source_data = sources[year].get('data');
    renderer_source.set('data', new_source_data);
    text_source.set('data', {'year': [String(year)]});
""" % js_source_array

callback = CustomJS(args=sources, code=code)
slider = Slider(start=years[0], end=years[-1], value=1, step=1, title="Year", callback=callback)
callback.args["renderer_source"] = renderer_source
callback.args["slider"] = slider
callback.args["text_source"] = text_source

# show(widgetbox(slider))

show(layout([[plot], [slider]], sizing_mode='scale_width'))

import nltk 
import string
import pickle
import gensim
import random 

from operator import itemgetter
from collections import defaultdict 
from nltk.corpus import wordnet as wn
from gensim.matutils import sparse2full
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

CORPUS_PATH = "data/baleen_sample"
PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
CAT_PATTERN = r'([a-z_\s]+)/.*'

class PickledCorpus(CategorizedCorpusReader, CorpusReader):
    
    def __init__(self, root, fileids=PKL_PATTERN, cat_pattern=CAT_PATTERN):
        CategorizedCorpusReader.__init__(self, {"cat_pattern": cat_pattern})
        CorpusReader.__init__(self, root, fileids)
        
        self.punct = set(string.punctuation) | {'“', '—', '’', '”', '…'}
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.wordnet = nltk.WordNetLemmatizer() 
    
    def _resolve(self, fileids, categories):
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories=categories)
        return fileids
    
    def lemmatize(self, token, tag):
        token = token.lower()
        
        if token not in self.stopwords:
            if not all(c in self.punct for c in token):
                tag =  {
                    'N': wn.NOUN,
                    'V': wn.VERB,
                    'R': wn.ADV,
                    'J': wn.ADJ
                }.get(tag[0], wn.NOUN)
                return self.wordnet.lemmatize(token, tag)
    
    def tokenize(self, doc):
        # Expects a preprocessed document, removes stopwords and punctuation
        # makes all tokens lowercase and lemmatizes them. 
        return list(filter(None, [
            self.lemmatize(token, tag)
            for paragraph in doc 
            for sentence in paragraph 
            for token, tag in sentence 
        ]))
    
    def docs(self, fileids=None, categories=None):
        # Resolve the fileids and the categories
        fileids = self._resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield self.tokenize(pickle.load(f))

# Create the Corpus Reader
corpus = PickledCorpus(CORPUS_PATH)

# Count the total number of documents
total_docs = 0

# Count the number of documents per category. 
for category in corpus.categories():
    num_docs = sum(1 for doc in corpus.fileids(categories=[category]))
    total_docs += num_docs 
    
    print("{}: {:,} documents".format(category, num_docs))
    
print("\n{:,} documents in the corpus".format(total_docs))

fid = random.choice(corpus.fileids())
doc = next(corpus.docs(fileids=[fid]))
print(" ".join(doc))

# Create the lexicon from the corpus 
lexicon = gensim.corpora.Dictionary(corpus.docs())

# Create the document vectors 
docvecs = [lexicon.doc2bow(doc) for doc in corpus.docs()]

# Train the TF-IDF model and convert vectors to TF-IDF
tfidf = gensim.models.TfidfModel(docvecs, id2word=lexicon, normalize=True)
tfidfvecs = [tfidf[doc] for doc in docvecs]

# Save the lexicon and TF-IDF model to disk.
lexicon.save('data/topics/lexicon.dat')
tfidf.save('data/topics/tfidf_model.pkl')

# Covert random document from above into TF-IDF vector 
dv = tfidf[lexicon.doc2bow(doc)]

# Print the document terms and their weights. 
print(" ".join([
    "{} ({:0.2f})".format(lexicon[tid], score)
    for tid, score in sorted(dv, key=itemgetter(1), reverse=True)
]))

# Select the number of topics to train the model on.
NUM_TOPICS = 10 

# Create the LDA model from the docvecs corpus and save to disk.
model = gensim.models.LdaModel(docvecs, id2word=lexicon, alpha='auto', num_topics=NUM_TOPICS)
model.save('data/topics/lda_model.pkl')

model[lexicon.doc2bow(doc)]

topics = [
    max(model[doc], key=itemgetter(1))[0]
    for doc in docvecs
]

for tid, topic in model.print_topics():
    print("Topic {}:\n{}\n".format(tid, topic))

# Create a sum dictionary that adds up the total probability 
# of each document in the corpus to each topic. 
tsize = defaultdict(float)
for doc in docvecs:
    for tid, prob in model[doc]:
        tsize[tid] += prob

# Create a numpy array of topic vectors where each vector 
# is the topic probability of all terms in the lexicon. 
tvecs = np.array([
    sparse2full(model.get_topic_terms(tid, len(lexicon)), len(lexicon)) 
    for tid in range(NUM_TOPICS)
])

# Import the model family 
from sklearn.decomposition import TruncatedSVD 

# Instantiate the model form, fit and transform 
topic_svd = TruncatedSVD(n_components=2)
svd_tvecs = topic_svd.fit_transform(tvecs)

# Create the Bokeh columnar data source with our various elements. 
# Note the resize/normalization of the topics so the radius of our
# topic circles fits int he graph a bit better. 
tsource = ColumnDataSource(
        data=dict(
            x=svd_tvecs[:, 0],
            y=svd_tvecs[:, 1],
            w=[model.print_topic(tid, 10) for tid in range(10)],
            c=brewer['Spectral'][10],
            r=[tsize[idx]/700000.0 for idx in range(10)],
        )
    )

# Create the hover tool so that we can visualize the topics. 
hover = HoverTool(
        tooltips=[
            ("Words", "@w"),
        ]
    )


# Create the figure to draw the graph on. 
plt = figure(
    title="Topic Model Decomposition", 
    width=960, height=540, 
    tools="pan,box_zoom,reset,resize,save"
)

# Add the hover tool 
plt.add_tools(hover)

# Plot the SVD topic dimensions as a scatter plot 
plt.scatter(
    'x', 'y', source=tsource, size=9,
    radius='r', line_color='c', fill_color='c',
    marker='circle', fill_alpha=0.85,
)

# Show the plot to render the JavaScript 
show(plt)

# In order to use Scikit-Learn we need to transform Gensim vectors into a numpy Matrix. 
docarr = np.array([sparse2full(vec, len(lexicon)) for vec in tfidfvecs])

# Import the model family 
from sklearn.decomposition import PCA 

# Instantiate the model form, fit and transform 
tfidf_pca = PCA(n_components=2)
pca_dvecs = topic_svd.fit_transform(docarr)

# Create a map using the ColorBrewer 'Paired' Palette to assign 
# Topic IDs to specific colors. 
cmap = {
    i: brewer['Paired'][10][i]
    for i in range(10)
}

# Create a tokens listing for our hover tool. 
tokens = [
    " ".join([
        lexicon[tid] for tid, _ in sorted(doc, key=itemgetter(1), reverse=True)
    ][:10])
    for doc in tfidfvecs
]

# Create a Bokeh tabular data source to describe the data we've created. 
source = ColumnDataSource(
        data=dict(
            x=pca_dvecs[:, 0],
            y=pca_dvecs[:, 1],
            w=tokens,
            t=topics,
            c=[cmap[t] for t in topics],
        )
    )

# Create an interactive hover tool so that we can see the document. 
hover = HoverTool(
        tooltips=[
            ("Words", "@w"),
            ("Topic", "@t"),
        ]
    )

# Create the figure to draw the graph on. 
plt = figure(
    title="PCA Decomposition of BoW Space", 
    width=960, height=540, 
    tools="pan,box_zoom,reset,resize,save"
)

# Add the hover tool to the figure 
plt.add_tools(hover)

# Create the scatter plot with the PCA dimensions as the points. 
plt.scatter(
    'x', 'y', source=source, size=9,
    marker='circle_x', line_color='c', 
    fill_color='c', fill_alpha=0.5,
)

# Show the plot to render the JavaScript 
show(plt)

# Import the TSNE model family from the manifold package 
from sklearn.manifold import TSNE 
from sklearn.pipeline import Pipeline

# Instantiate the model form, it is usually recommended 
# To apply PCA (for dense data) or TruncatedSVD (for sparse)
# before TSNE to reduce noise and improve performance. 
tsne = Pipeline([
    ('svd', TruncatedSVD(n_components=75)),
    ('tsne', TSNE(n_components=2)),
])
                     
# Transform our TF-IDF vectors.
tsne_dvecs = tsne.fit_transform(docarr)

# Create a map using the ColorBrewer 'Paired' Palette to assign 
# Topic IDs to specific colors. 
cmap = {
    i: brewer['Paired'][10][i]
    for i in range(10)
}

# Create a tokens listing for our hover tool. 
tokens = [
    " ".join([
        lexicon[tid] for tid, _ in sorted(doc, key=itemgetter(1), reverse=True)
    ][:10])
    for doc in tfidfvecs
]

# Create a Bokeh tabular data source to describe the data we've created. 
source = ColumnDataSource(
        data=dict(
            x=tsne_dvecs[:, 0],
            y=tsne_dvecs[:, 1],
            w=tokens,
            t=topics,
            c=[cmap[t] for t in topics],
        )
    )

# Create an interactive hover tool so that we can see the document. 
hover = HoverTool(
        tooltips=[
            ("Words", "@w"),
            ("Topic", "@t"),
        ]
    )

# Create the figure to draw the graph on. 
plt = figure(
    title="TSNE Decomposition of BoW Space", 
    width=960, height=540, 
    tools="pan,box_zoom,reset,resize,save"
)

# Add the hover tool to the figure 
plt.add_tools(hover)

# Create the scatter plot with the PCA dimensions as the points. 
plt.scatter(
    'x', 'y', source=source, size=9,
    marker='circle_x', line_color='c', 
    fill_color='c', fill_alpha=0.5,
)

# Show the plot to render the JavaScript 
show(plt)

