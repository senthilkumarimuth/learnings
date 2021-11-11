#Bernoulli distribution
from scipy.stats import bernoulli
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)



p = 0.5
#Generate random variable using bernoulli
data =  bernoulli.rvs(p,size=10)
#calculating four moments
mean, var, skew, kurt = bernoulli.stats(p, moments='mvsk')
#Display the distribution
x = sns.displot(data,kde = False, color = 'blue')
x.set(xlabel = 'Bernoulli Distribution', ylabel = 'Frequency')
#calulate probability mass function
a = np.arange(bernoulli.ppf(0.01, p),
              bernoulli.ppf(0.99, p))
bernoulli.pmf(a, p)
ax.plot(x, bernoulli.pmf(a, p), 'bo', ms=8, label='bernoulli pmf')

