import pandas as pd

df=pd.DataFrame({'a':10,'b':20},index=[0,1])

#apply square root on full DataFrame

import numpy as np

df.apply(np.sqrt)

#Using a reducing function on either axis

df.apply(np.sum, axis=0)

#Using a reducing function on either axis

df.apply(np.sum, axis=1)

#Returning a list-like will result in a Series

df.apply(lambda x: [1, 2], axis=1)

# Passing result_type=’expand’ will expand list-like results to columns of a Dataframe

df.apply(lambda x: [1, 2], axis=1, result_type='expand')

#Returning a Series inside the function is similar to passing result_type='expand'. The resulting column names will be the Series index.

df.apply(lambda x: pd.Series([1, 2], index=['foo', 'bar']), axis=1)

#Passing result_type='broadcast' will ensure the same shape result, whether list-like or scalar is returned by the function, and broadcast it along the axis. The resulting column names will be the originals.

df.apply(lambda x: [1, 2], axis=1, result_type='broadcast')