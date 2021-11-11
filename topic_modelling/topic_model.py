from top2vec import Top2Vec
from sklearn.datasets import fetch_20newsgroups
#https://github.com/ddangelov/Top2Vec#pretrained

#download dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

#build topic model
model = Top2Vec(documents=newsgroups.data, speed="learn", workers=8)

#to get number of topics
model.get_num_topics()

#topic number and its document frequency
topic_sizes, topic_nums = model.get_topic_sizes()

#To get the desired number of topic from documents
topic_words, word_scores, topic_nums = model.get_topics(77)

#To search a desired word in
topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=["prayer"], num_topics=5)

#wordcloud plot
for topic in topic_nums:
    model.generate_topic_wordcloud(topic)

#search documents by topic

documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=0, num_docs=5)
for doc, score, doc_id in zip(documents, document_scores, document_ids):
    print(f"Document: {doc_id}, Score: {score}")
    print("-----------")
    print(doc)
    print("-----------")
    print()

#semantic search by keyword

documents, document_scores, document_ids = model.search_documents_by_keywords(keywords=["christ", "heaven"], num_docs=5)
for doc, score, doc_id in zip(documents, document_scores, document_ids):
    print(f"Document: {doc_id}, Score: {score}")
    print("-----------")
    print(doc)
    print("-----------")
    print()

#search for similar words

words, word_scores = model.similar_words(keywords=["god"], keywords_neg=[], num_words=20)
for word, score in zip(words, word_scores):
    print(f"{word} {score}")