import gensim
from gensim.models.ldamulticore import LdaMulticore


def get_lda_model(index_tokens, num_topics=6, passes=3):
    print('Getting gensim LDA topic model')
    dictionary = gensim.corpora.Dictionary(index_tokens)
    corpus = [dictionary.doc2bow(text) for text in index_tokens]
    lda_model = LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
    return lda_model, dictionary, corpus


def get_top_topic(lda_model, dictionary, tokens):
    bow = dictionary.doc2bow(tokens)
    topic_probs = lda_model[bow]
    if not topic_probs or len(topic_probs) == 0:
        return -1
    topic_probs.sort(key=lambda tup: tup[1], reverse=True)
    return topic_probs[0][0]


def get_topic_vector(lda_model, dictionary, tokens):
    bow = dictionary.doc2bow(tokens)
    fingerprint = [0] * lda_model.num_topics
    for topic, prob in lda_model[bow]:
        fingerprint[topic] = prob
    return fingerprint