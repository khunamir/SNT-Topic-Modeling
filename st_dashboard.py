import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import gensim
import pickle
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
from plotly.subplots import make_subplots
from pandasgui import show
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from gensim.parsing.preprocessing import STOPWORDS
from wordcloud import WordCloud

colors = ['peachpuff','lightskyblue','turquoise','darkorange','purple','olive','lightgreen','darkseagreen','maroon','teal',
          'deepskyblue','red','mediumblue','indigo','goldenrod','mediumvioletred','pink','beige','rosybrown']

st.set_page_config(layout="wide")

st.markdown("<h1 style='font-weight: normal'><b>Topic Model</b>: Science and Technology News</h1>", unsafe_allow_html=True)

def load_mpmt(site):    
    with open(f'./Models/{site}Models/{site.lower()}_lda_passes_train.pickle', 'rb') as file:
        model_passes = pickle.load(file)

    with open(f'./Models/{site}Models/{site.lower()}_lda_topics_train.pickle', 'rb') as file:
        model_topics = pickle.load(file)

    mp_df = pd.DataFrame(model_passes)
    mp_df = mp_df.transpose()
    mp_df = mp_df.iloc[0:50]
    mp_df['coherence'] = mp_df['coherence'].astype(float)

    mt_df = pd.DataFrame(model_topics)
    mt_df = mt_df.transpose()
    mt_df = mt_df.iloc[0:50]
    mt_df['coherence'] = mt_df['coherence'].astype(float)

    return mp_df, mt_df

def load_ex(site):
    with open(f'./Models/{site}Models/{site.lower()}_extreme2.pickle', 'rb') as file:
        model_extreme = pickle.load(file)

    ex_df = pd.DataFrame(model_extreme)
    ex_df = ex_df.transpose()
    ex_df['coherence'] = ex_df['coherence'].astype(float)
    ex_df = ex_df.reset_index()

    best_model = ex_df.iloc[ex_df['coherence'].idxmax()]['model']
    bow_corpus = ex_df.iloc[ex_df['coherence'].idxmax()]['corpus']
    dictionary = ex_df.iloc[ex_df['coherence'].idxmax()]['dictionary']

    return ex_df, best_model, bow_corpus, dictionary

def load_model(site):
    with open(f'./{site}Data/preprocessed_scitech.pkl', 'rb') as file:
        processed_series = pickle.load(file)

    return processed_series

def load_related(site, bow_corpus, highest_top):
    with open(f"./{site}Data/SciTechData.pkl", "rb") as file:
        news = pickle.load(file)

    dm_topic = []

    for i, corp in enumerate(bow_corpus):
        topic_percs = best_model.get_document_topics(corp)
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dm_topic.append(dominant_topic)

    news['dominant_topic'] = dm_topic

    return news[news['dominant_topic'] == highest_top]['url'][:10]

def load_evaluation_graph(data, xlabel, ylabel, title):
    if (len(data) > 25):
        fig = px.line(data, x=range(1, len(data)+1), y='coherence', title=title, labels={'x': xlabel, 'y': ylabel})
        fig.add_hline(y=data['coherence'].max())
        try:
            vert_value = int(data['coherence'].idxmax().split('a')[1])
        except:
            vert_value = int(data['coherence'].idxmax().split('s')[1])
    else:
        fig = px.line(data[::-1], x=range(30, 100, 10), y='coherence', title=title, labels={'x': xlabel, 'y': ylabel})
        vert_value = int(data.reset_index()['coherence'].idxmax())
        fig.update_xaxes(range=[30, 90])

    fig.add_vline(x=vert_value)

    return fig, vert_value

def load_cloud(processed_series):
    all_words = ''
    stopwords = set(STOPWORDS)

    for val in processed_series:
        all_words += ' '.join(val)+' '

    wordcloud = WordCloud(width = 1800, height = 1600,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(all_words)
                    
    # fig = plt.figure(figsize = (8, 8), facecolor = None)
    # ax = fig.add_axes([2, 2, 10, 10])
    # ax.imshow(wordcloud)
    # ax.axis("off")
    # fig.tight_layout(pad = 0)

    fig = px.imshow(wordcloud)

    return fig

def load_cloud_each(model, site):
    if site == 'Popular Science' or site == 'Cosmos Magazine':
        words = ['u']
    elif site == 'Discover Magazine':
        words = ['nt', 'u', 've', 'm', 'll', 'd', 'rofl']

    stopwords = set(STOPWORDS)

    for i in words:
        stopwords.add(i)

    num_topics = len(model.get_topics())

    topic_top3words = [(i, topic) for i, topics in model.show_topics(formatted=False, num_topics=num_topics) for j, (topic, wt) in enumerate(topics) if j < 3]

    k=0
    new_list = []
    new_new_list = []

    j = 0
    while (j < len(topic_top3words)):
        i = topic_top3words[j][1]

        if(j == len(topic_top3words)-1):
            new_new_list.append(new_list)

        if(k<3):
            j += 1
        else:
            new_new_list.append(new_list)
            new_list = []
            k = 0
            continue
        new_list.append(i)
        k += 1

    cloud = WordCloud(stopwords=stopwords,
                    background_color='white',
                    width=750,
                    height=750,
                    max_words=10,
                    colormap='tab10',
                    color_func=lambda *args, **kwargs: color_func(*args, **kwargs, n=n, topics=new_new_list[n]),
                    prefer_horizontal=1.0)

    topics = model.show_topics(num_topics=num_topics, formatted=False)

    j = 0
    n = 0
    col1, col2, col3, col4, col5 = st.columns(5)

    while n < num_topics:        
        if (j < 5):
            if (j == 0):
                col = col1
            elif (j == 1):
                col = col2
            elif (j == 2):
                col = col3
            elif (j == 3):
                col = col4
            elif (j == 4):
                col = col5
        else:
            j = 0
            col1, col2, col3, col4, col5 = st.columns(5)
            continue

        with col:
            fig = plt.figure(figsize=(1.5,1.5))
            plt.title('Topic ' + str(n+1), fontdict=dict(size=6))
            plt.axis('off')
            topic_words = dict(topics[n][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=400)
            plt.imshow(cloud)
            st.write(fig)

        j += 1
        n += 1        

def load_LDAvis(model, corpus, dictionary):
    vis = gensimvis.prepare(model, corpus, dictionary)
    html_string = pyLDAvis.prepared_data_to_html(vis)

    return html_string

def load_topic_document_count(best_model, bow_corpus):
    dm_topic = []

    for i, corp in enumerate(bow_corpus):
        topic_percs = best_model.get_document_topics(corp)
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dm_topic.append(dominant_topic)
        
    dm_df = pd.DataFrame(dm_topic, columns=['dominant_topic'])

    topic_top3words = [(i, topic) for i, topics in best_model.show_topics(formatted=False, num_topics=-1) for j, (topic, wt) in enumerate(topics) if j < 3]
    
    df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
    df_top3words = df_top3words_stacked.groupby('topic_id').agg(', '.join)
    df_top3words.reset_index(level=0,inplace=True)

    count_df = pd.DataFrame(dm_df.groupby('dominant_topic').dominant_topic.agg('count').to_frame('COUNT').reset_index()['COUNT'])
    count_df['top3'] = list(df_top3words['words'])

    fig = px.histogram(dm_df, 
                        x='dominant_topic',
                        labels={'dominant_topic': 'Dominant topic', 'count': 'Number of Documents'}, 
                        height=500,
                        width=1400,
                        title='Documents Count by Dominant Topic')
    fig.update_layout(yaxis_title='Number of Documents', bargap=0.2)
    fig.update_layout(
        margin=dict(b=40),
        xaxis = dict(
            tickmode = 'array',
            tickvals = list(range(dm_df['dominant_topic'].max()+1)),
            ticktext = df_top3words['words']
        )
    )

    return fig, count_df[count_df['COUNT'] == count_df['COUNT'].max()]['top3'].values[0], count_df['COUNT'].idxmax()

def load_document_count(data):
    doc_len = [len(d) for d in data]

    fifth = round(np.quantile(doc_len, q=0.05))
    ninefifth = round(np.quantile(doc_len, q=0.95))

    text = "Mean  : " + str(round(np.mean(doc_len))) \
       + "<br>Median  : " + str(round(np.median(doc_len))) \
       + "<br>Std dev.  : " + str(round(np.std(doc_len))) \
       + "<br>5th percentile  : " + str(round(np.quantile(doc_len, q=0.05))) \
       + "<br>95th percentile  : " + str(round(np.quantile(doc_len, q=0.95)))

    fig = px.histogram(doc_len, labels={"value": "Document Word Count"}, height=500, width=1400, title='Distribution of Documents Word Count')
    fig.add_annotation(x=0.95, xref='paper', y=0.95, yref='paper', text=text, showarrow=False, bgcolor="#F4F4F4", opacity=0.8, borderpad=8, borderwidth=2, bordercolor="#DDDDDD", align='left')
    fig.update_layout(yaxis_title='Number of Documents', showlegend=False)

    return fig, fifth, ninefifth

def color_func(word, font_size, position, orientation, font_path, random_state, n, topics):
    if word in topics:
        return colors[n]
    else:
        return 'lightgrey'

def load_topic_word_prob(best_model):
    topic_prob_list = [i[1].split(',') for i in best_model.show_topics(num_topics=-1)]

    prob_list = []
    words_list = []

    for i in topic_prob_list:
        num_list = re.findall(r'[\d]*[.][\d]+', *i)
        conv = [float(j) for j in num_list]
        prob_list.append(conv)

        words = re.findall(r'"(.*?)"', *i)
        words_list.append(words)

    def flatten(l):
        return [item for sublist in l for item in sublist]

    words_list = flatten(words_list)
    topnum_list = sorted(list(range(best_model.num_topics)) * 10)
    prob_list = flatten(prob_list)

    data = {
        "topic": topnum_list,
        "words": words_list,
        "probability": prob_list
    }

    topic_prob = pd.DataFrame(data)
    new_df = topic_prob.set_index(['topic'])

    rows = math.ceil(best_model.num_topics / 5)

    fig = make_subplots(
        rows=rows,
        cols=5, 
        shared_yaxes=True,
        subplot_titles=[f'Topic {n}' for n in range(1, best_model.num_topics+1)]
    )

    j = 1
    n = 0

    for i in range(1, rows+1):
        for j in range(1, 6):
            if (n < best_model.num_topics):
                fig.add_trace(
                    go.Bar(x=new_df.loc[n]['words'], y=new_df.loc[n]['probability']),
                    row=i, col=j
                )
                
                n += 1

    fig.update_layout(height=1000, width=1400, title_text="Topic Word Probabilities", showlegend=False, margin=dict(b=5))

    return fig

def load_tSNE(best_model, bow_corpus):
    # Get topic weights
    topic_weights = []
    for i, row_list in enumerate(best_model[bow_corpus]):
        topic_weights.append([w for i, w in row_list])

    # Array of topic weights    
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    # Plot the Topic Clusters using Bokeh
    colors = ['peachpuff','lightskyblue','turquoise','darkorange','purple','olive','lightgreen','darkseagreen','maroon','teal',
            'deepskyblue','red','mediumblue','indigo','goldenrod','mediumvioletred','pink','beige','rosybrown']
    n_topics = 4
    mycolors = np.array([color for color in colors])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
                plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])

    return plot

site = st.selectbox(
    'Select which site to analyze topics',
    ('Popular Science', 'Discover Magazine', 'Cosmos Magazine'),
)

vert_space = '<div style="padding: 20px 5px;"></div>'
st.markdown(vert_space, unsafe_allow_html=True)

if site:
    if site == 'Popular Science':
        site = 'PopSci'
    elif site == 'Discover Magazine':
        site = 'Discover'
    elif site == 'Cosmos Magazine':
        site = 'Cosmos'

    mp_df, mt_df = load_mpmt(site)

    st.subheader("How good is the model?")

    passes_graph, passes_vert = load_evaluation_graph(mp_df, 'Number of Passes', 'Topic Coherence', 'Topic Coherence vs Number of Passes' )
    passes_graph.update_layout(width=650)

    topics_graph, topics_vert = load_evaluation_graph(mt_df, 'Number of Topics', 'Topic Coherence', 'Topic Coherence vs Number of Topics' )
    topics_graph.update_layout(width=650)

    mdt_best = round(mt_df['coherence'].max(),4)

    st.markdown(f"The **:blue[best performing model]** obtained a coherence score of **:blue[{mdt_best}]** !  \n \
        The model performed best with {passes_vert} iterations over the whole corpus and {topics_vert} number of topics.")

    col1, col2 = st.columns(2)

    with col1:
        st.write(passes_graph)

    with col2:
        st.write(topics_graph)

    ex_df, best_model, bow_corpus, dictionary = load_ex(site)

    st.subheader("The model were also found to be performing better when extreme word occurrences are filtered!")

    ex_best = round(ex_df['coherence'].max(), 4)
    imp = round(ex_best / mdt_best, 4)

    st.markdown(f"This time, the **:blue[best performing model]** obtained a coherence score of **:blue[{ex_best}]**. \n \
        An increase of another **:blue[{imp}]**% !")

    best_graph, best_vert = load_evaluation_graph(ex_df, 'Percentage of Documents Used to Filter', 'Topic Coherence', 'Topic Coherence vs Percentage of Documents' )

    best_graph.update_layout(width=1400)

    st.write(best_graph)

    #col1, col2 = st.columns(2)

    processed_series = load_model(site)

    if site == 'PopSci':
        site = 'Popular Science'
    elif site == 'Discover':
        site = 'Discover Magazine'
    elif site == 'Cosmos':
        site = 'Cosmos Magazine'

    document_count, fifth, ninefifth = load_document_count(processed_series)
    topic_document_count, top_3, top_i = load_topic_document_count(best_model, bow_corpus)

    top_3 = top_3.split(',')
    
    st.subheader("How long are the documents?")

    st.markdown(f"Most documents in {site} are between **:blue[{fifth}]** and **:blue[{ninefifth}]** words long!")             

    st.write(document_count)    

    st.subheader(f"What are the most discussed topics in {site}?")

    st.markdown(f"The most discussed topics are related to the keywords **:blue[{top_3[0].upper()}]**, **:blue[{top_3[1].upper()}]** and **:blue[{top_3[2].upper()}]**")
    st.write(topic_document_count)

    if site == 'Popular Science':
        site = 'PopSci'
    elif site == 'Discover Magazine':
        site = 'Discover'
    elif site == 'Cosmos Magazine':
        site = 'Cosmos'

    related_url = load_related(site, bow_corpus, top_i)

    st.subheader("These articles have the highest probability of having above topic!")

    st.markdown('<div style="padding: 25px 5px;"></div>', unsafe_allow_html=True)

    st.write(related_url, width=1000)

    st.markdown('<div style="padding: 25px 5px;"></div>', unsafe_allow_html=True)

    st.subheader("Explore the topics below!")

    st.markdown(vert_space, unsafe_allow_html=True)

    if site == 'PopSci':
        site = 'Popular Science'
    elif site == 'Discover':
        site = 'Discover Magazine'
    elif site == 'Cosmos':
        site = 'Cosmos Magazine'

    load_cloud_each(best_model, site)

    st.markdown('<div style="padding: 40px 5px;"></div>', unsafe_allow_html=True)

    lda_vis = load_LDAvis(best_model, bow_corpus, dictionary)
    #st.write(lda_vis)

    st.subheader("LDAVis Visualization")
    st.markdown('<div style="padding: 20px 5px;"></div>', unsafe_allow_html=True)
    st.components.v1.html(lda_vis, height=1100, width=1400)