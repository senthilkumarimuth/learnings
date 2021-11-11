from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups(subset='all')['data']

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(data, show_progress_bar=True)