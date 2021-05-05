import pandas as pd

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('file.tsv', sep='\t', names = column_names)
print(df.head())

movie_titles = pd.read_csv('Movie_Id_Titles.csv')
print(movie_titles.head())

data = pd.merge(df,movie_titles,on='item_id')
print(data.head())

data.groupby('title')['rating'].mean().sort_values(ascending=False)
data.groupby('title')['rating'].count().sort_values(ascending=False).head()

ratings = pd.DataFrame(data.groupby('title')['rating'].mean()) 
  
ratings['num of ratings'] = pd.DataFrame(data.groupby('title')['rating'].count())
  
print(ratings.head())

# Sorting values according to 
# the 'num of rating column'
moviemat = data.pivot_table(index ='user_id',
              columns ='title', values ='rating')
  
print(moviemat.head())
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
  

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)
  
corr_starwars = pd.DataFrame(similar_to_starwars, columns =['Correlation'])
corr_starwars.dropna(inplace = True)
  
print(corr_starwars.head())



