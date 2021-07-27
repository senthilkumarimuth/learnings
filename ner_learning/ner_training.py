from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import spacy
from tqdm import tqdm
from spacy.training.example import Example

'''
# to be used when the sample.json is created by the below link
https://github.com/ManivannanMurugavel/spacy-ner-annotator

import json
f = open('sample.json')
data = json.load(f)
print(len(data))

TRAIN_DATA =[]

for dict in data:
    temp_list=[]
    temp_list.append(dict['content'])
    temp_ent = []
    for j in dict["entities"]:
            j = j[:3]
            k=[j[1],j[0],j[2]]
            temp_ent.append(tuple(k))
    temp_list.append({'entities':temp_ent})
    temp_tuple = tuple(temp_list)
    TRAIN_DATA.append(temp_tuple)
print(TRAIN_DATA)

'''
TRAIN_DATA = [

    ('Update links diameter from t12 to t10', {

        'entities': [(0, 6, 'T-Update'), (27, 30, 'O-target'), (34, 37, 'N-target')]

    }),

    ('Update links diameter from t15 to t50', {

        'entities': [(0, 6, 'T-Update'), (27, 30, 'O-target'), (34, 37, 'N-target')]

    }),

    ('Edit links diameter from t10 to t20', {

        'entities': [(0, 4, 'T-Update'), (25, 28, 'O-target'), (32, 36, 'N-target')]

    }),

    ('Replace word columns with sections', {

        'entities': [(0, 7, 'T-Update'), (13, 21, 'O-target'), (26, 35, 'N-target')]

    }),

    ('Replace text columns with sections', {

        'entities': [(0, 7, 'T-Update'), (13, 21, 'O-target'), (26, 35, 'N-target')]

    }),

    ('Edit word rows as lines', {

        'entities': [(0, 4, 'T-Update'), (10, 14, 'O-target'), (18, 23, 'N-target')]

    }),

    ('Edit word rows as lines', {

        'entities': [(0, 4, 'T-Update'), (10, 14, 'O-target'), (18, 23, 'N-target')]

    }),

    ('Rephrase text centimeter as cm', {

        'entities': [(0, 8, 'T-Update'), (13, 24, 'O-target'), (28, 30, 'N-target')]

    }),

    ('Rephrase word vert as verticle', {

        'entities': [(0, 8, 'T-Update'), (13, 18, 'O-target'), (22, 30, 'N-target')]

    }),

    ('Change text scale to measurement', {

        'entities': [(0, 6, 'T-Update'), (12, 16, 'O-target'), (20, 31, 'N-target')]

    }),

    ('Change scale to measurement', {

        'entities': [(0, 6, 'T-Update'), (7, 12, 'O-target'), (16, 27, 'N-target')]

    }),

    ('Update word scale as measurement', {

        'entities': [(0, 6, 'T-Update'), (12, 17, 'O-target'), (21, 31, 'N-target')]

    }),

    ('Add word sections at the end', {

        'entities': [(0, 3, 'T-Insert'), (9, 17, 'target'), (25, 28, 'position')]

    }),

    ('Insert word diameter at the beginning', {

        'entities': [(0, 6, 'T-Insert'), (11, 20, 'target'), (27, 37, 'position')]

    }),

    ('Add text centimeter after 80', {

        'entities': [(0, 3, 'T-Insert'), (9, 19, 'target'), (20, 28, 'position')]

    }),

    ('Insert text circle before diameter', {

        'entities': [(0, 6, 'T-Insert'), (12, 18, 'target'), (19, 34, 'position')]

    }),

    ('Add word graphics at the end', {

        'entities': [(0, 3, 'T-Insert'), (9, 17, 'target'), (25, 28, 'position')]

    }),

    ('Insert word circle at the beginning', {

        'entities': [(0, 6, 'T-Insert'), (12, 18, 'target'), (26, 35, 'position')]

    }),

    ('Add text centimeter after 100', {

        'entities': [(0, 3, 'T-Insert'), (9, 19, 'target'), (20, 300, 'position')]

    }),

    ('Insert text verticle before diameter', {

        'entities': [(0, 6, 'T-Insert'), (12, 20, 'target'), (21, 36, 'position')]

    }),

    ('Add text verticle after 80', {

        'entities': [(0, 3, 'T-Insert'), (9, 17, 'target'), (18, 26, 'position')]

    }),

    ('Insert text circle before rotor', {

        'entities': [(0, 6, 'T-Insert'), (12, 18, 'target'), (19, 31, 'position')]

    }),

    ('Delete text t12', {

        'entities': [(0, 6, 'T-Delete'), (12, 15, 'target')]

    }),

    ('Remove text t12', {

        'entities': [(0, 6, 'T-Delete'), (12, 15, 'target')]

    }),

    ('Erase text t12', {

        'entities': [(0, 5, 'T-Delete'), (11, 14, 'target')]

    }),

    ('Remove word links', {

        'entities': [(0, 6, 'T-Delete'), (12, 17, 'target')]

    }),

    ('Delete text t15', {

        'entities': [(0, 6, 'T-Delete'), (12, 15, 'target')]

    }),

    ('Remove text t15', {

        'entities': [(0, 6, 'T-Delete'), (12, 15, 'target')]

    }),

    ('Erase text vert', {

        'entities': [(0, 5, 'T-Delete'), (11, 15, 'target')]

    }),

    ('Remove word circle', {

        'entities': [(0, 6, 'T-Delete'), (12, 18, 'target')]

    }),

    ('Erase text 80', {

        'entities': [(0, 5, 'T-Delete'), (11, 13, 'target')]

    }),

    ('Remove word column', {

        'entities': [(0, 6, 'T-Delete'), (12, 17, 'target')]

    }),

    ('Delete text diameter', {

        'entities': [(0, 6, 'T-Delete'), (12, 20, 'target')]

    }),

    ('Remove text verticle', {

        'entities': [(0, 6, 'T-Delete'), (12, 20, 'target')]

    }),

    ('Erase text section', {

        'entities': [(0, 5, 'T-Delete'), (11, 18, 'target')]

    }),

    ('Remove word assembly', {

        'entities': [(0, 6, 'T-Delete'), (12, 20, 'target')]

    }),

    ('Delete text parallel', {

        'entities': [(0, 6, 'T-Delete'), (12, 20, 'target')]

    }),

    ('Remove text links', {

        'entities': [(0, 6, 'T-Delete'), (12, 17, 'target')]

    }),

    ('Erase text cover', {

        'entities': [(0, 5, 'T-Delete'), (11, 16, 'target')]

    }),

    ('Remove word grid', {

        'entities': [(0, 6, 'T-Delete'), (12, 17, 'target')]

    }),

    ('Erase text scale', {

        'entities': [(0, 5, 'T-Delete'), (11, 16, 'target')]

    }),

    ('Remove word refer', {

        'entities': [(0, 6, 'T-Delete'), (12, 16, 'target')]

    })

]

TRAIN_DATA1 = [
    ('replace diameter from 1500 to 800', {
        'entities': [(0, 5, 'action'),(22,25, "source"),(30,33, "target")]
    }),
     ('erase word connect', {
        'entities': [(0, 5, 'action'), (12,18,"target")]
    }),
    ('include word square in the middle', {
        'entities': [(0, 2, 'action'), (9, 14, 'target')]
    })
]

''''
TRAIN_DATA = [
    ('Who is Nishanth?', {
        'entities': [(7, 15, 'PERSON')]
    }),
     ('Who is Kamal Khumar?', {
        'entities': [(7, 19, 'PERSON')]
    }),
    ('I like London and Berlin.', {
        'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
    })
]
'''
model = None
output_dir=Path(r'.\\ner_model')
n_iter=10

#load the model
""""
if model is not None:
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
else:
"""

nlp = spacy.blank('en')
print("Created blank 'en' model")

#set up the pipeline

if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe('ner')

for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):   #Todo: use tqdm to show the progress
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in tqdm(TRAIN_DATA):
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update(
                [example],
                drop=0.2,
                sgd=optimizer,
                losses=losses)
        print(losses)

for text, _ in TRAIN_DATA1:
    doc = nlp(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])

if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)

