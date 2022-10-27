import pandas as pd
import os
from expertai.nlapi.cloud.client import ExpertAiClient
from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize
import warnings
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.lsimodel import LsiModel
from gensim import corpora
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

os.environ["EAI_USERNAME"] = 'megha.naik@rakuten.com'
os.environ["EAI_PASSWORD"] = 'Rakuten@123'

client = ExpertAiClient()

warnings.filterwarnings('ignore')

def get_dataset():
    datafile = 'data/scraped_data.csv'
    dataset =  pd.read_csv(datafile)
    document = dataset.text 
    news_df = pd.DataFrame({'document':document})
    tokenized_doc = news_df['document'].str.replace("[^a-zA-Z#]"," ")
    return tokenized_doc

def clean_doc(text):
    text = " ".join([word.lower() for word in text.split() if len(word)>3])
    return text

def remove_stopwords(text):
    stopwords_set = set(stopwords.words('english'))
    custom = list(stopwords_set)+list(punctuation)
    text = word_tokenize(text)
    text = " ".join([word for word in text if word not in custom])
    return text

def lemma_data(text):
    #Initialize wordnet lemmatizer
    lemmatizer = WordNetLemmatizer()
    text = word_tokenize(text)
    text = " ".join([lemmatizer.lemmatize(word) for word in text])
    return text

def tm_truncated_svd():
    tokenized_doc = get_dataset()
    
    tokenized_doc = tokenized_doc.apply(clean_doc)
    tokenized_doc = tokenized_doc.apply(remove_stopwords)
    tokenized_doc = tokenized_doc.apply(lemma_data)

    #initialize the tfidf vectorizer with default stopword list
    tfidf = TfidfVectorizer(stop_words="english",max_features=1000,max_df=0.5,smooth_idf=True)

    #Vectorizing 'doucument' columns
    vector = tfidf.fit_transform(tokenized_doc)

    #COnvert vector into an array
    X = vector.toarray()

    svd_model = TruncatedSVD(n_components=5,algorithm='randomized',n_iter=100,random_state=122)
    svd_model.fit(X)
    print("The number of topics chosen are ",len(svd_model.components_))

    terms =  tfidf.get_feature_names()
    topics = []
    for i,comp in enumerate(svd_model.components_):
        terms_comp = zip(terms,comp)
        sorted_terms = sorted(terms_comp,key=lambda x : x[1] , reverse=True)[:10]
        topics.append("Topic "+str(i)+": ")
        for t in sorted_terms:
            topics.append(t[0])

    final_topic_list = [topics[i:i+11] for i in range(0, len(topics),11)]

    for x in final_topic_list:
        print(x)


def lemma_data_expert_ai(text):
    output = client.specific_resource_analysis(
        body={"document": {"text": text}},
        params={'language': language, 'resource': 'disambiguation'
                })

    text = " ".join([token.lemma for token in output.tokens])
    return text

# text = lemma_data_expert_ai(text)
# print(text)



# Function to lemmatize and remove the stopwords
def clean(doc):
    stop = set(stopwords.words('english'))
    exclude = set(punctuation)
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    # normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    list_of_docs = punc_free.tolist()
    new_list = []
    for row in list_of_docs:
        text = sent_tokenize(row)
        for doc in text:
            t = lemma_data_expert_ai(doc)
            new_list.append(t)
    return new_list


def tm_gensim_lda():
    tokenized_doc = get_dataset()
    # Creating a list of documents from the complaints column
    list_of_docs = tokenized_doc.tolist()

    # Implementing the function for all the complaints of list_of_docs
    doc_clean = [clean(doc).split() for doc in list_of_docs]

    # Creating the dictionary id2word from our cleaned word list doc_clean
    dictionary = corpora.Dictionary(doc_clean)

    # Creating the corpus
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # Creating the LSi model
    lsimodel = LsiModel(corpus=doc_term_matrix, num_topics=5, id2word=dictionary)
    pprint(lsimodel.print_topics())

    # Creating the dictionary id2word from our cleaned word list doc_clean
    dictionary = corpora.Dictionary(doc_clean)

    # Creating the corpus
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # Creating the LDA model
    ldamodel = LdaModel(corpus=doc_term_matrix, num_topics=5,id2word=dictionary, random_state=20, passes=30)

    # printing the topics
    pprint(ldamodel.print_topics())

    # Compute Perplexity
    perplexity_lda = ldamodel.log_perplexity(doc_term_matrix)
    print('\nPerplexity: ', perplexity_lda)  


    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=ldamodel, texts=doc_clean, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stopwords,
                    background_color='white',
                    width=2500,
                    height=1800,
                    max_words=10,
                    colormap='tab10',
                    color_func=lambda *args, **kwargs: cols[i],
                    prefer_horizontal=1.0)

    topics = ldamodel.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    fig.savefig('word_cloud.png')

