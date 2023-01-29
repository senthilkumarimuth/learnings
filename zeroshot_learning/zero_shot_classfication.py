from transformers import pipeline

pipe  = pipeline(model = 'facebook/bart-large-mnli')
pipe('hi there', candidate_labels = ['travel', 'greeting', 'angry'])