#Hashing embedding helps us to share some vectors for a few vocabulary so that it helps for dimentionality rediction
"""
We show that models trained using hash embeddings exhibit at least the same level of
performance as models trained using regular embeddings across a wide range of
tasks. Furthermore, the number of parameters needed by such an embedding is
only a fraction of what is required by a regular embedding. Since standard embeddings and embeddings constructed
using the hashing trick are actually just special
cases of a hash embedding, hash embeddings can be considered an extension and
improvement over the existing regular embedding types.

# if just use one hash function(key1) alone that is called feature hashing. But if we use more than one hash function
which means we are hash embeddings.

"""
import numpy
import mmh3

vocab = ['apple', 'strawberry', 'orange', 'juice', 'drink', 'smoothie',
         'eat', 'fruit', 'health', 'wellness', 'steak', 'fries', 'ketchup',
         'burger', 'chips', 'lobster', 'caviar', 'service', 'waiter', 'chef']

normal = numpy.random.uniform(-0.1, 0.1, (20, 2))
hashed = numpy.random.uniform(-0.1, 0.1, (15, 2))

for word in vocab:
    key1 = mmh3.hash(word, 0) % 15
    key2 = mmh3.hash(word, 1) % 15
    vector = hashed[key1] + hashed[key2]
    print(word, '%.3f %.3f' % tuple(vector))


# calculate loss for hashing embeded vectors

import numpy
import numpy.random as random
import mmh3

random.seed(0)

nb_epoch = 50
learn_rate = 0.001
nr_hash_vector = 15

words = [str(i) for i in range(20)]
true_vectors = numpy.random.uniform(-0.1, 0.1, (len(words), 2))
hash_vectors = numpy.random.uniform(-0.1, 0.1, (nr_hash_vector, 2))
examples = list(zip(words, true_vectors))

for epoch in range(nb_epoch):
    random.shuffle(examples)
    loss=0.
    for word, truth in examples:
        key1 = mmh3.hash(word, 0) % nr_hash_vector
        key2 = mmh3.hash(word, 1) % nr_hash_vector

        hash_vector = hash_vectors[key1] + hash_vectors[key2]

        diff = hash_vector - truth

        hash_vectors[key1] -= learn_rate * diff
        hash_vectors[key2] -= learn_rate * diff
        loss += (diff**2).sum()
    print(epoch, loss)
