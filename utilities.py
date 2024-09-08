import re, pickle
import numpy as np
from numpy.linalg import norm
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

WINDOW = 8
THRESHOLD = 0.9
STOPWORDS = stopwords.words("english")
glove_vecs = KeyedVectors.load_word2vec_format('glove.6B.100d.txt.word2vec', binary=False)

def get_sents(text):
    sents = [s for s in sent_tokenize(clean_text(text))]
    return [[w for w in word_tokenize(s) if w not in STOPWORDS] for s in sents], sents

def cos_sim(a, b):
    A = np.array(a)
    B = np.array(b)

    return np.dot(A,B)/(norm(A)*norm(B))

def clean_text(text):
    if not isinstance(text, str):
        text = ""

    text = re.sub(r'[\n|\r]', ' ', text)
    # remove space between ending word and punctuations
    text = re.sub(r'[ ]+([\.\?\!\,]{1,})', r'\1 ', text)
    # remove duplicated spaces
    text = re.sub(r' +', ' ', text)
    # add space if no between punctuation and words
    text = re.sub(r'([a-z|A-Z]{2,})([\.\?\!]{1,})([a-z|A-Z]{1,})', r'\1\2\n\3', text)
    # handle case "...\" that" that the sentence spliter cannot do
    text = re.sub(r'([\?\!\.]+)(\")([\s]+)([a-z|A-Z]{1,})', r'\1\2\n\3\4', text)
    # remove space between letter and punctuation
    text = re.sub(r'([a-z|A-Z]{2,})([ ]+)([\.\?\!])', r'\1\3', text)
    # handle case '\".word' that needs space after '.'
    text = re.sub(r'([\"\']+\.)([a-z|A-Z]{1,})', r'\1\n\2', text)
    # handle case '.\"word' that needs space after '\"'
    text = re.sub(r'(\.[\"\']+)([a-z|A-Z]{1,})', r'\1\n\2', text)
    # Remove all punctuation that isn't a period
    text = re.sub(r"[!#$%&\(\)*+,\-/:;<=>?@\[\]^_`~]", "", text)
    text = text.lower()

    if len(text) > 0 and text[-1].isalpha():
        text += "."
    return text

def get_similarity(sent):
    vectorizer = CountVectorizer()
    vectorizer.fit(mvp + [sent])
    mvp_vecs = vectorizer.transform(mvp).toarray()
    sent_vec = vectorizer.transform([sent]).toarray()[0]

    mx = 0
    for vec in mvp_vecs:
        sim = cos_sim(vec, sent_vec)
        mx = sim if sim > mx else mx
    
    return mx

def predict_quote(sent):
    with open(f"model.pkl", "rb") as handle:
        model = pickle.load(handle)

    return model.predict([get_similarity(sent)])

def get_single_topic_dependent(window):
    window = [w for w in window if w not in stopwords.words("english")]
    topic = None
    for word in window:
        tmp = check_dependent_keyword(word)
        if tmp != None:
            ind = window.index(word)
            after = window[ind+1] if len(window) > ind+1 else window[ind-1]
            max_top = get_max_topic(window)

            if after in inv_keywords_split:
                topic = inv_keywords_split[after]
            elif max_top in tmp:
                topic = max_top
        else:
            for keyword in inv_keywords_split.keys():
                try:
                    if keyword == word or isPlural(keyword, word) or glove_vecs.similarity(word, keyword) >= THRESHOLD:
                        topic = inv_keywords_split[keyword]
                except:
                    continue
    
    # if topic == "hospital" and "malaria" in window:
    #     return "malaria"
    # else:
    return topic

## Could probably just do this while finding the first topic, but this is a quick solution
def remove_topic(window, topic):
    topic_words = unique_keywords[topic] + context_keywords[topic]
    first_pass = [w for w in window if w not in topic_words]
    final = []

    for word in first_pass:
        found = False

        for keyword in topic_words:
            try:
                if isPlural(keyword, word) or glove_vecs.similarity(word, keyword) >= THRESHOLD:
                    found = True
                    break
            except:
                continue

        if not found:
            final.append(word)
            
    return final

def isPlural(w1, w2):
    return (w1 + "s" == w2 or w2 + "s" == w1)

def get_max_topic(window):
    topics = {"hospital" : 0, "malaria" : 0, "farming" : 0, "school" : 0}

    for word in window:
        for keyword in inv_keywords_split.keys():
            try:
                if glove_vecs.similarity(word, keyword) >= THRESHOLD or keyword == word:
                    topics[inv_keywords_split[keyword]] += 1
            except:
                continue

    mx = max(topics, key=(lambda x: topics[x]))

    return mx if topics[mx] > 1 else None

def check_dependent_keyword(word):
    tk_topics = None
    ## For each topic and keyword in the topic
    for topic in context_keywords.keys():
        for keyword in context_keywords[topic]:
            try:
                if glove_vecs.similarity(word, keyword) >= THRESHOLD or keyword == word:
                    if tk_topics == None:
                        tk_topics = [topic]
                    else:
                        tk_topics.append(topic if topic not in tk_topics else None)
            except:
                continue
    
    return tk_topics

context_keywords = {"hospital" : ["care", "water", "generator", "die", "bed", "running", "disease"],\
                    "malaria"  : ["bed", "sleeping", "die", "cheap", "disease"],\
                    "farming"  : ["dying", "water"],\
                    "school"   : ["supplies", "fee", "midday", "supply", "energy", "free", "children", "kid"]}

unique_keywords = {"hospital" : ["health", "hospital", "treatment", "doctor", "medicine", "patient", "clinical", "officer"],\
                   "malaria"  : ["net", "malaria", "infect", "bednet", "mosquito", "bug", "mosquitoes", "biting"],\
                   "farming"  : ["farmer", "fertilizer", "irrigation", "crop", "seed", "harvest"],\
                   "school"   : ["school", "student", "meal", "lunch", "book", "paper", "pencil", "attend"]}

inv_keywords_split = {w : k for k, v in unique_keywords.items() for w in v}

mvp = get_sents(clean_text("".join(open("mvp.txt").readlines())))[1]