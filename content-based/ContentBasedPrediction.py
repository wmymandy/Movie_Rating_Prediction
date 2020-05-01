import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('imdb.csv',index_col=0)
df = df.replace(r'\\N','unknown', regex=True) 
df = df.fillna('unknown')

#IMDB's weighted rating formula
# Claculation of c 
numVotes = df[df['numVotes'].notnull()]['numVotes'].astype('int')
averageRating = df[df['averageRating'].notnull()]['averageRating'].astype('int')
C = averageRating.mean()

# Claculation of m
m = numVotes.quantile(0.95)

def weighted_rating(x):
    v = x['numVotes']
    R = x['averageRating']
    return (v/(v+m) * R) + (m/(m+v) * C)

#create keywords
df['keywords'] = df['genres'] + ',' + df['directors'] + ',' + df['writers'] + ',' + df['actors']

# cosine similarity on keywords
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(df['keywords'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)


df = df.reset_index()
titles = df['title']
indices = pd.Series(df.index, index=df['title'])

def recommendation(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, reverse=True)
    sim_scores = sim_scores[1:51]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = df.iloc[movie_indices][['title', 'numVotes', 'averageRating']]
    vote_counts = movies[movies['numVotes'].notnull()]['numVotes'].astype('int')
    vote_averages = movies[movies['averageRating'].notnull()]['averageRating'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['numVotes'] >= m) & (movies['numVotes'].notnull()) & (movies['averageRating'].notnull())]
    qualified['numVotes'] = qualified['numVotes'].astype('int')
    qualified['averageRating'] = qualified['averageRating'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified

predict = recommendation('The Godfather')
print('predicting the scores....')
predic_score = sum(predict['wr'])/len(predict['wr'])
print(predic_score)

predict.toDF().toPandas().to_csv('predict.csv')
