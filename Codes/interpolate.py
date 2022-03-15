import numpy as np

import matplotlib.pyplot as plt
from scipy import interpolate
x = np.arange(0, 10)
y = np.exp(-x/3.0)
f = interpolate.interp1d(x, y)

xnew = np.arange(0, 2, 0.1)
ynew = f(xnew)   # use interpolation function returned by `interp1d`
plt.plot(xnew, ynew, 'o', xnew, ynew, '-')
plt.show()


#Pandas


# importing pandas as pd
import pandas as pd

# Creating the dataframe
df = pd.DataFrame({"A": [12, 4, 5, None, 1],
                   "B": [None, 2, 54, 3, None],
                   "C": [20, 16, None, 3, 8],
                   "D": [14, 3, None, None, 6]})

# Print the dataframe
df

# to interpolate the missing values
df.interpolate(method ='linear', limit_direction ='forward')


























