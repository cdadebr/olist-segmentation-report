import re
import umap
import pickle
import base64
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from translators import translate_text

# Opening relevant files
with open('one_reviews_df.pkl', 'rb') as file:
    one_reviews_df = pickle.load(file)

# Ngrams
with open('data/ngrams/top10_tokens.pkl', 'rb') as file:
    top10_tokens = pickle.load(file)

with open('data/ngrams/reduced_ngram_embeddings.pkl', 'rb') as file:
    reduced_ngram_embeddings = pickle.load(file)

with open('data/ngrams/ngram_results.pkl', 'rb') as file:
    ngram_results = pickle.load(file)

with open('data/ngrams/translated_ngrams.pkl', 'rb') as file:
    ngram_translations = pickle.load(file)

with open('data/ngrams/token2comments.pkl', 'rb') as file:
    token2comments = pickle.load(file)

# Comments
with open('data/comments/all_comments.pkl', 'rb') as file:
    comment_list = pickle.load(file)

with open('data/comments/translated_comments.pkl', 'rb') as file:
    comment_translations = pickle.load(file)

# Predefining functions
def plot_umap_3d(ngrams_pca, excerpts, n_neighbors, min_dist):
    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(ngrams_pca)

    # Create DataFrame for easier plotting with Plotly
    df_3d = pd.DataFrame(embedding, columns=['x', 'y', 'z'])
    df_3d['excerpt'] = excerpts

    # Plot using Plotly
    fig = px.scatter_3d(df_3d, x='x', y='y', z='z',
                        hover_data=['excerpt'])
    fig.update_traces(marker=dict(size=3), selector=dict(mode='markers'))
    st.plotly_chart(fig)#, use_container_width=True)

def matched_word(subword, idx):
        pattern = r'\b\w*' + re.escape(subword) + r'\w*\b'
        match = re.findall(pattern, token2comments[subword][idx][0])
        return match

def contextualise_word(subword, fullword, idx):
    sentence = token2comments[subword][idx][0].split(' ')
    for i, word in enumerate(sentence):
        if fullword in word:
            if i-4 > 0 and i+4 <= len(sentence):
                return translate_text(' '.join(sentence[i-4:i+4]), 'google')
            if i-4 <= 0:
                return  translate_text(' '.join(sentence[:7]), 'google')
            if i+4 > len(sentence):
                return translate_text(' '.join(sentence[-7:]), 'google')
            if i-4 < 0 and i+4 > len(sentence):
                return translate_text(' '.join(sentence), 'google')

def print_five(key):
    word_col, trans_col, context_col = [], [], []
    for example_num in range(5):
        word = matched_word(key, example_num)[0]
        english_trans = translate_text(word, 'google')
        context = contextualise_word(key, word, example_num)
        word_col.append(word)
        trans_col.append(english_trans)
        context_col.append(context)
    df_dict = {'Matched word': word_col, 'Translated word': trans_col, 'In context': context_col}
    df = pd.DataFrame(df_dict)
    df.index.name = 'Comment'
    return st.table(df)

# Compile app
def streamlit_app():    
    st.title('E-commerce Review Analysis')

    # Part 1: High-Level Overview
    st.header('Overview')
    st.write("""
    Customer comments with low review scores (1-star) were examined on the Olist platform - a service that helps businesses set up online stores in Brazil.
    \nThe aim was to provide business owners in the Olist ecosystem with clarity and direction on how to enhance customer satisfaction. 
    \nThis was done by grouping similar types of complaints together and identifying which types were most common or impactful.
    """)

    # Part 2: Data Preview
    st.header('Sample Reviews')
    st.write(f"""
    There are {len(one_reviews_df)} total 1-star reviews.
    \nThe majority of review comments were written in Portuguese.
    \nSome keywords to take note of: "waiting", "return", "not received", "refund", "differed", "not coming".
    """)
    st.table(one_reviews_df.head()[['translated comment']])


    # Part 3: Explain Methodology 1 (Clustering Shap Value Ngrams)
    st.header('Key Issues in 1-Star Reviews')
    st.write(f"""
    Tokens are the basic units of text or code used by a Large Language Model AI (think ChatGPT) to process and generate language. 
    \nIn this case, tokens could be either partial or whole words depending on word length.
    \nA Large Language Model (LLM) was trained to predict the review score for comments in the Olist dataset.
    \nIn order for the model to improve in its predictions, it had to learn the tokens most strongly correlated with 1-star reviews.
    \nHere are the model's top 10 most predictive tokens:""")
    
        # Visualise Shap Values
    roboto = {'fontname':'Roboto'}
    fig, ax = plt.subplots(figsize=(5,7.5))
    top10_tokens_list = list(top10_tokens.items())
    ax.barh([item[0] for item in top10_tokens_list], [item[1] for item in top10_tokens_list])
    ax.set_xlabel('Average SHAP Value', **roboto)
    ax.set_ylabel('Tokens', **roboto)
    ax.set_title('Top Tokens in Low Score Reviews', **roboto)
    ax.invert_yaxis()
    st.pyplot(fig)

    keys = list(top10_tokens.keys())
    
        # Create Token-to-Comment DataFrames
    st.write(f"""
    Tokens remain in English because some are subwords and translate poorly. 
    \nProvided below are the first three tokens in context: the matched words containing these tokens, their direct English translation, and their use in a comment.
    \nThe direct translation and the translation in context may slightly differ, but the substitute word will have similar semantic meaning.""")

    st.write(f"""**{keys[0]}**""")
    print_five(keys[0])

    st.write(f"""**{keys[1]}**""")
    print_five(keys[1])

    st.write(f"""**{keys[2]}**""")
    print_five(keys[2])

    #st.write(f"""**{keys[3]}**""")
    #print_five(keys[3])

    #st.write(f"""**{keys[4]}**""")
    #print_five(keys[4])
    
    #st.write(f"""**{keys[5]}**""")
    #print_five(keys[5])

    #st.write(f"""**{keys[6]}**""")
    #print_five(keys[6])

    #st.write(f"""**{keys[7]}**""")
    #print_five(keys[7])

    #st.write(f"""**{keys[8]}**""")
    #print_five(keys[8])

    #st.write(f"""**{keys[9]}**""")
    #print_five(keys[9

    
    # Explain Ngram Extraction
    st.subheader(f"""Ngram Extraction""")
    st.write(f"""   
    Comments from the dataset were chosen if they contained one or more of these tokens.
    \nThe idea being: these tokens linked to business issues that easily identified comments as 1-star reviews.
    \nThe token's position in a comment was found and a 7-token (approximately 5-word) long sliding window iterated over the token from left to right, capturing 
    snapshots (known as Ngrams) of the token in slightly different contexts.
    \n""")

    # Embed Ngram Gif
    ngram_file = open(r"media/videos/ngram_visualisation/NgramVisualization.gif", 'rb')
    ngram_contents = ngram_file.read()
    data_url = base64.b64encode(ngram_contents).decode('utf-8')
    ngram_file.close()
    st.markdown(f'<div align="center"><img src="data:image/gif;base64,{data_url}" alt="cat gif"><div>', unsafe_allow_html=True)

    video_file = open('media/videos/ngram_visualisation/1080p60/NgramVisualization.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

    # Explain Sliding Window Process
    st.write(f"""
    For example's sake, say "This is an example sentence." was a comment in the review dataset, and "an" was the high shap-value token.
    \nThe sliding window, of 3-word length in this example, would feature "an" at its 3rd, 2nd, and 1st positions as it iterates over the sentence one word at a time.
    \nWith each iteration, the sliding window saves the snapshot taken and groups it with its respective comment.""") 
    
    st.write(f"""
    \n
    \nThese Ngrams were grouped with the original comment and fed through the same LLM to create a numerical representation of the Ngram's contextual meaning.
    \nFor example, two sentences using the word bank in different contexts (say, a financial institution and the 'bank' of a river) would have completely 
    different numerical representations.
    \nHowever, the numerical representations for two sentences, one talking about financial banks and the other brokers, would be very similar.
    \nUsing cosine similarity, the Ngram with the closest numerical representation to its original comment was chosen, effectively selecting the best summary of the comment's sentiment.
    \nThere was now a corresponding concentrated Ngram summary for each comment.""")

   
    # Explain Ngram Clustering
    st.subheader(f"""Ngram Clustering""")
    st.write(f"""
    The numerical representations for summary Ngrams were plotted on a high-dimensional space.
    \nEach dimension in this space represents a feature the LLM learnt to successfully predict 1-star reviews.
    \nThe values in the numerical representations of each Ngram represent the degree to which a feature is present.
    \nOne such feature the LLM could have learnt is the concept of delivery time, or the concept of broken products.
    \nThe numerical representations for Ngrams with similar features, meaning similar problems, should cluster together when plotted in this high dimensional space.
    \nUsing a dimensionality reduction algorithm called UMAP, you can visualize the high dimensional representation for Ngrams in a 3-dimensional space:""")

    plot_umap_3d(reduced_ngram_embeddings, ngram_translations, ngram_results[0]['umap_n_neighbors'], ngram_results[0]['umap_min_dist'])  

streamlit_app()
