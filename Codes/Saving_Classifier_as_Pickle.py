
import pickle

#Saving pickle file - Pickling

'''
Pickling is a way to convert a python object (list, dict, etc.) into a character stream. 
The idea is that this character stream contains all the information necessary to reconstruct the object in another python script.
'''

pickle.dump(classifier, open('classifier', 'wb'))


#Loading pickle file - Unpickling

'''
While the process of retrieving original Python objects from the stored character stream representation is called unpickling
'''

classifier = pickle.load(open('classifier', 'rb'))