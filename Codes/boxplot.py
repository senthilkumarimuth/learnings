import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style = 'whitegrid')
tips = sns.load_dataset('tips')
import pandas as pd
df = pd.DataFrame({'num':[0.1,100,200,300,400,500,600,2000]})
ax = sns.boxplot(y=df['num'])

plt.show()



