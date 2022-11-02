
def token(senten, word_tokenize):
    results = []
    for sentence in senten:
        results.append(word_tokenize(sentence))
    return results

def topic_model(reviews_lemmatized, gensim, np, MovieGroupProcess, int_val):
    np.random.seed(0)

    # initialize GSDMM
    gsdmm = MovieGroupProcess(K=int_val, alpha=0.1, beta=0.3, n_iters=int_val)

    # create dictionary of all words in all documents
    dictionary = gensim.corpora.Dictionary(reviews_lemmatized)

    # filter extreme cases out of dictionary
    dictionary.filter_extremes(no_below=int_val, no_above=0.5, keep_n=100000)

    # create variable containing length of dictionary/vocab
    n_terms = len(dictionary)

    # fit GSDMM model
    model = gsdmm.fit(reviews_lemmatized, n_terms)
    doc_count = np.array(gsdmm.cluster_doc_count)

    # topics sorted by the number of document they are allocated to
    top_index = doc_count.argsort()[-int_val:][::-1]

    # show the top 20 words in term frequency for each cluster
    ans = []
    ans = top_words(gsdmm, gsdmm.cluster_word_distribution, top_index, 15)
    return top_index, gsdmm, ans

def top_words(gsdmm, cluster_word_distribution, top_cluster, values):
    ans = []
    for cluster in top_cluster:
        sort_dicts =sorted(gsdmm.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
        ans.append("\nCluster %s : %s"%(cluster,sort_dicts))
    return ans

def create_topics_dataframe(pd, data_text,  mgp, threshold, topic_dict, lemma_text):
    result = pd.DataFrame(columns=['Text', 'Topic', 'Lemma-text'])
    for i, text in enumerate(data_text):
        result.at[i, 'Text'] = text
        result.at[i, 'Lemma-text'] = lemma_text[i]
        prob = mgp.choose_best_label(lemma_text[i])
        if prob[1] >= threshold:
            result.at[i, 'Topic'] = topic_dict[prob[0]]
        else:
            result.at[i, 'Topic'] = 'Other'
    return result

def processing(data, gensim, malaya, word_tokenize, np, MovieGroupProcess, pd, WordCloud, int_val, stopwords, list_stop):
    df = data.iloc[:, 0]

    # remove characters and turn to lower case
    df1 = df.str.lower().str.replace('[^\w\s]','')

    # change text abbreviations to original word
    df1 = df1.str.replace(r'\bx\b', 'tidak')
    df1 = df1.str.replace(r'\btak\b', 'tidak')
    df1 = df1.str.replace(r'\borg\b', 'orang')
    df1 = df1.str.replace(r'\bdgn\b', 'dengan')
    df1 = df1.str.replace(r'\bmora\b', 'moratorium')
    df1 = df1.str.replace(r'\bni\b', 'ini')
    df1 = df1.str.replace(r'\btu\b', 'itu')

    # remove unwanted word
    df1 = df1.str.replace('\n', '')
    df1 = df1.str.replace(r'\bla\b', '')
    df1 = df1.str.replace(r'\bje\b', '') 

    # remove stopword
    stop_words = malaya.text.function.STOPWORDS
    stops = set(stopwords.words('english'))
    stop_words.update(stops)
    stop_words.update(list_stop)
    df2 = df1.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    # dataframe change to list
    list_dat = df2.values.tolist()
    
    # tokenize word
    reviews_lemmatized = token(list_dat, word_tokenize)

    # GSDMM for the topic modeling
    ans = []
    top_index, gsdmm, ans = topic_model(reviews_lemmatized, gensim, np, MovieGroupProcess, int_val)

    # give name to the cluster
    list_topic = list(top_index)
    topic_dict = {}
    topic_names = []
    for i in range(len(list_topic)):
        topic_names.append("Cluster " + str(list_topic[i]))

    for i, topic_num in enumerate(top_index):
        topic_dict[topic_num]=topic_names[i]

    # create dataframe with topic
    result = create_topics_dataframe(pd, data_text=df1, mgp=gsdmm, threshold=0.3, topic_dict=topic_dict, lemma_text=reviews_lemmatized)
    result['Lemma_text'] = result['Lemma-text'].apply(lambda row: ' '.join(row))
    result = result.drop('Lemma-text', axis=1)

    # create dataframe with label
    final_df = pd.concat([df, result['Topic']], axis=1)

    # create word clouds
    wc = []
    for i in range(int_val):
        wc.append(create_WordCloud(WordCloud, result['Lemma_text'].loc[result.Topic == topic_names[i]], title=("Most used words in "+topic_names[i])))
    return wc, ans, final_df

def create_WordCloud(WordCloud, data, title=None):
    wordcloud = WordCloud(width = 400, height = 400,
                          collocations = False,
                          background_color ='white',
                          min_font_size = 14
                          ).generate(" ".join(data.values))

    return wordcloud
