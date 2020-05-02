import pyspark.sql.functions as sql_func
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import os

#os.environ['HADOOP_HOME'] = "c:\\hadoop-common"
#print(os.environ['HADOOP_HOME'])

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

data_schema = StructType([
    StructField('movie',IntegerType(), False),
    StructField('user',IntegerType(), False),
    StructField('rating',IntegerType(), False),
    StructField('date',TimestampType(), False)
])

final_stat = spark.read.csv(
    'small_data.csv', header=True, schema=data_schema
).cache()

print(final_stat.head(5))

ratings = (final_stat
    .select(
        'movie',
        'user',
        'rating'
    )
).cache()

(training, test) = ratings.randomSplit([0.8, 0.2])
print("Training Size: ", training.count(), training.columns )
print("Testing Size: ", test.count(), training.columns)

als = ALS(rank = 5,
          maxIter=5, regParam=0.01,
          userCol="user", itemCol="movie", ratingCol="rating",
          coldStartStrategy="drop",
          implicitPrefs=False)
model = als.fit(training)
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

#Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
print("Number of Unique Users: ", userRecs.count())
#Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)
print("Number of Unique Movies: ", movieRecs.count())

userRecs_df = userRecs.toPandas()
print(userRecs_df.shape)

movieRecs_df = movieRecs.toPandas()
print(movieRecs_df.shape)

userRecs_df.to_csv('UseRec.csv', index = False)
movieRecs_df.to_csv('MovieRec.csv', index = False)

print(userRecs_df.head(5))
print(movieRecs_df.head(5))

topredict=test[test['user']==1]
topredict.show()
pd = model.transform(topredict)
pd.show()

recs = userRecs_df[userRecs_df['user']==1] ['recommendations']
print("Top 10 recommendation movies for user 53832:")
print("Movie\t", "Score\t")
for rec in recs:
    for i in range(10):
        print(rec[i][0], "\t", rec[i][1])





