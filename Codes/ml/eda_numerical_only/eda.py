import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read the dataset
df = pd.read_csv('export-33.csv')

#mean,min,max,percentile,count,std, 0.50 is median
result = df.describe([0.01, 0.05, 0.10, 0.50,0.90, 0.95,0.99])

#null values count
na_series = df.isnull().sum(axis = 0)
na_df = pd.DataFrame(na_series)
na_df = na_df.transpose()
na_df = na_df.rename(index={0: 'Null values'})
result = pd.concat([result, na_df])

#histogram
for column in df.columns:
   #fig, ax = plt.subplots()
   sns.histplot(data= df, x = column)
   plt.savefig('./histogram/'+column+'_histplot.png')
   plt.close()
   plt.clf()

#boxplot
for column in df.columns:
   sns.boxplot(data= df, x = column)
   plt.savefig('./boxplot/'+column+'_boxplot.png')
   plt.close()
   plt.clf()

#save the results to CSV
result.to_csv('export-33_descriptive_statistics.csv')

