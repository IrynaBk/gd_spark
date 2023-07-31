from pyspark import SparkConf
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, BooleanType, ArrayType
from pyspark.sql import DataFrame
from pyspark.sql.functions import avg, col, sum, max, to_timestamp, explode, split, hour, count, rank, percent_rank, \
    cume_dist, size, year, datediff, current_date, to_date, dense_rank, row_number, upper

import matplotlib.pyplot as plt


def calculate_average_stars_by_state(df: DataFrame) -> DataFrame:
    return df.groupBy("state") \
        .agg(avg("stars").alias("avg_stars"))


def cumulative_reviews_over_cities(business_df: DataFrame, review_df: DataFrame) -> DataFrame:
    business_df = business_df.select('name', 'business_id', 'city')
    date_format = 'yyyy-MM-dd HH:mm:ss'
    review_df = review_df.drop('useful', 'funny', 'cool', 'stars', 'text')
    review_df = review_df.withColumn("date", to_date(col("date"), date_format))
    review_df = review_df.withColumn("year", year(review_df["date"]))
    joined_df = business_df.join(review_df, on='business_id', how='inner')

    window = Window.partitionBy(joined_df['city']).orderBy(joined_df['year'])
    cumul_reviews = joined_df.withColumn('cumulative_reviews', count('review_id').over(window))

    return cumul_reviews


def calculate_category_rank(df: DataFrame) -> DataFrame:
    df = df.withColumn("categories", split(col("categories"), ", "))
    df = df.filter(col("categories").isNotNull())
    df = df.withColumn("state", upper(df.state))

    exploded_df = df.select("business_id", "state", explode(df.categories).alias("category"))

    category_counts = exploded_df.groupBy("state", "category").agg(count("*").alias("business_count"))

    window = Window.partitionBy(category_counts['state']).orderBy(category_counts['business_count'].desc())

    category_counts = category_counts.withColumn('rank', row_number().over(window))

    category_counts.filter(category_counts.state == "LA").show()
    category_counts.filter(category_counts.rank == 1).show()

    return category_counts


def get_most_checkins(business_df: DataFrame, checkin_df: DataFrame, city: str) -> DataFrame:
    city_business_df = business_df.filter(business_df['city'] == city)

    checkin_df = checkin_df.withColumn('date', explode(split(checkin_df['date'], ', ')))

    checkin_df = checkin_df.withColumn('date', to_timestamp(checkin_df['date'], 'yyyy-MM-dd HH:mm:ss'))

    checkin_df = checkin_df.filter(hour(col('date')) >= 9).filter(hour(col('date')) < 18)
    joined_df = city_business_df.join(checkin_df, on='business_id', how='inner')

    checkin_counts = joined_df.groupBy('postal_code') \
        .agg(count('*').alias('checkin_count')) \
        .sort(col('checkin_count').desc())

    return checkin_counts


def get_closed_business_reviews_summary(business_df: DataFrame,
                                        review_df: DataFrame) -> DataFrame:
    review_df = review_df.drop('stars')
    closed_business_df = business_df.filter(business_df['is_open'] == 0)
    joined_df = closed_business_df.join(review_df, on='business_id', how='inner')
    result_df = joined_df.groupBy('business_id', 'name', 'city', 'stars') \
        .agg(sum('useful').alias('total_useful'),
             sum('funny').alias('total_funny'),
             sum('cool').alias('total_cool')) \
        .sort(col('stars').desc())
    result_df = result_df.agg(avg("stars").alias("avg_stars"),
                              avg("total_funny").alias("avg_funny"),
                              avg("total_cool").alias("avg_cool"),
                              avg("total_useful").alias("avg_useful"))
    return result_df


if __name__ == '__main__':
    spark = (SparkSession.builder
             .master('local')
             .appName('my_task')
             .config(conf=SparkConf())
             .getOrCreate())

    business_schema = StructType([
        StructField("business_id", StringType(), False),
        StructField("name", StringType(), True),
        StructField("address", StringType(), True),
        StructField("city", StringType(), True),
        StructField("state", StringType(), True),
        StructField("postal_code", StringType(), True),
        StructField("latitude", FloatType(), True),
        StructField("longitude", FloatType(), True),
        StructField("stars", FloatType(), True),
        StructField("review_count", IntegerType(), True),
        StructField("is_open", IntegerType(), True),
        StructField("attributes", StringType(), True),
        StructField("categories", StringType(), True),
        StructField("hours", StringType(), True)
    ])

    review_schema = StructType([
        StructField("review_id", StringType(), False),
        StructField("user_id", StringType(), True),
        StructField("business_id", StringType(), True),
        StructField("stars", FloatType(), True),
        StructField("date", StringType(), True),
        StructField("text", StringType(), True),
        StructField("useful", IntegerType(), True),
        StructField("funny", IntegerType(), True),
        StructField("cool", IntegerType(), True)
    ])

    user_schema = StructType([
        StructField("user_id", StringType(), False),
        StructField("name", StringType(), True),
        StructField("review_count", IntegerType(), True),
        StructField("yelping_since", StringType(), True),
        StructField("friends", StringType(), True),
        StructField("useful", IntegerType(), True),
        StructField("funny", IntegerType(), True),
        StructField("cool", IntegerType(), True),
        StructField("fans", IntegerType(), True),
        StructField("elite", StringType(), True),
        StructField("average_stars", FloatType(), True),
        StructField("compliment_hot", IntegerType(), True),
        StructField("compliment_more", IntegerType(), True),
        StructField("compliment_profile", IntegerType(), True),
        StructField("compliment_cute", IntegerType(), True),
        StructField("compliment_list", IntegerType(), True),
        StructField("compliment_note", IntegerType(), True),
        StructField("compliment_plain", IntegerType(), True),
        StructField("compliment_cool", IntegerType(), True),
        StructField("compliment_funny", IntegerType(), True),
        StructField("compliment_writer", IntegerType(), True),
        StructField("compliment_photos", IntegerType(), True)
    ])

    checkin_schema = StructType([
        StructField("business_id", StringType(), False),
        StructField("date", StringType(), True)
    ])

    tip_schema = StructType([
        StructField("text", StringType(), nullable=False),
        StructField("date", StringType(), nullable=False),
        StructField("compliment_count", IntegerType(), nullable=False),
        StructField("business_id", StringType(), nullable=False),
        StructField("user_id", StringType(), nullable=False)
    ])

    json_file_path = "file:///mnt/yelp_dataset/"
    business_df = spark.read.json(path=json_file_path + 'yelp_academic_dataset_business.json',
                                  schema=business_schema)
    user_df = spark.read.json(path=json_file_path + 'yelp_academic_dataset_user.json',
                              schema=user_schema)
    review_df = spark.read.json(path=json_file_path + 'yelp_academic_dataset_review.json',
                                schema=review_schema)
    tip_df = spark.read.json(path=json_file_path + 'yelp_academic_dataset_tip.json',
                             schema=tip_schema)
    checkin_df = spark.read.json(path=json_file_path + 'yelp_academic_dataset_checkin.json',
                                 schema=checkin_schema)


    average_stars_df = calculate_average_stars_by_state(business_df)
    average_stars_df.sort(col('avg_stars').asc()).show()

    closed_df = get_closed_business_reviews_summary(business_df, review_df)
    closed_df.show(truncate=False)
    checkins_res_df = get_most_checkins(business_df, checkin_df, 'Reno')
    checkins_res_df.show(3)

    result_df = calculate_category_rank(business_df)
    result_df.show()

    result_df.filter(result_df.state == "LA").show()
    result_df.filter(result_df.rank == 1).show()

    res_df = cumulative_reviews_over_cities(business_df, review_df)
    res_df.groupBy('year').count().show()
    res_df.filter((res_df.city == 'Riverside') | (res_df.city == 'riverside') | (res_df.city == 'RIVERSIDE')).groupBy(
        'year').count().show()

