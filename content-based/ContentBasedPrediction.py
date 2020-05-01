import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('imdb_clean_1990.csv',index_col=0)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

def weighted_rating(x,m,C):
    v = x['numVotes']
    R = x['averageRating']
    return (v/(v+m) * R) + (m/(m+v) * C)

#create keywords
df['keywords'] = df['genres'] + ',' + df['directors'] + ',' + df['writers'] + ',' + df['actors']

# # cosine similarity on keywords
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
matrix = count.fit_transform(df['keywords'])
cosine_sim = cosine_similarity(matrix, matrix)

df = df.reset_index()
titles = df['primaryTitle']
indices = pd.Series(df.index, index=df['primaryTitle'])

def recommendation(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, reverse=True)
    sim_scores = sim_scores[1:51]
    movie_indices = [i[0] for i in sim_scores]
    
    # create recommendation table
    movies = df.iloc[movie_indices][['primaryTitle', 'numVotes', 'averageRating']]
    #IMDB's weighted rating formula
    vote_counts = movies[movies['numVotes'].notnull()]['numVotes'].astype('int')
    vote_averages = movies[movies['averageRating'].notnull()]['averageRating'].astype('int')
    C = vote_averages.mean()
    # m = vote_counts.quantile(0.60)
    m = 100
    qualified = movies[(movies['numVotes'] >= m) & (movies['numVotes'].notnull()) & (movies['averageRating'].notnull())]
    qualified['numVotes'] = qualified['numVotes'].astype('int')
    qualified['averageRating'] = qualified['averageRating'].astype('int')
    # qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified['wr'] = weighted_rating(qualified,m,C)
    qualified = qualified.sort_values('averageRating', ascending=False).head(10)
    return qualified

predict = recommendation('Shazam!')
print('-------------------------------------------------')
print('Movie Name: Shazam! ')
print(' predicting the scores.....')
print(predict)
predict_score = sum(predict['wr'])/len(predict['wr'])
print('predicted score: ',predict_score)
