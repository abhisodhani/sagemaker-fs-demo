from pyspark.sql.types import StructField, StructType, StringType, DoubleType, TimestampType, LongType
from pyspark.sql.functions import desc, dense_rank
from pyspark.sql import SparkSession, DataFrame
from  argparse import Namespace, ArgumentParser
from pyspark.sql.window import Window
import argparse
import logging
import boto3
import time
import sys
import os


TOTAL_UNIQUE_USERS = 10000
FEATURE_GROUP = 'cc-agg-batch-chime-fg'
TRAIN_FEATURE_GROUP = 'cc-train-chime-fg'

logger = logging.getLogger('sagemaker')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


feature_store_client = boto3.client(service_name='sagemaker-featurestore-runtime')


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Spark Job Input and Output Args')
    parser.add_argument('--s3_input_bucket', type=str, help='S3 Input Bucket')
    parser.add_argument('--s3_input_key_prefix', type=str, help='S3 Input Key Prefix')
    parser.add_argument('--s3_output_bucket', type=str, help='S3 Output Bucket')
    parser.add_argument('--s3_output_key_prefix', type=str, help='S3 Output Key Prefix')
    args = parser.parse_args()
    return args
    

def define_schema() -> StructType:
    schema = StructType([StructField('tid', StringType(), True),
                         StructField('datetime', TimestampType(), True),
                         StructField('cc_num', LongType(), True),
                         StructField('amount', DoubleType(), True),
                         StructField('fraud_label', StringType(), True)])
    return schema


def aggregate_features(args: Namespace, schema: StructType, spark: SparkSession) -> DataFrame:
    logger.info('[Read Raw Transactions Data as Spark DataFrame]')
    transactions_df = spark.read.csv(f's3a://{os.path.join(args.s3_input_bucket, args.s3_input_key_prefix)}', \
                                     header=False, \
                                     schema=schema)
    logger.info('[Aggregate Transactions to Derive New Features using Spark SQL]')
    query = """
    SELECT *, \
           avg_amt_last_10m/avg_amt_last_1w AS amt_ratio1, \
           amount/avg_amt_last_1w AS amt_ratio2, \
           num_trans_last_10m/num_trans_last_1w AS count_ratio \
    FROM \
        ( \
        SELECT *, \
               COUNT(*) OVER w1 as num_trans_last_10m, \
               AVG(amount) OVER w1 as avg_amt_last_10m, \
               COUNT(*) OVER w2 as num_trans_last_1w, \
               AVG(amount) OVER w2 as avg_amt_last_1w \
        FROM transactions_df \
        WINDOW \
               w1 AS (PARTITION BY cc_num order by cast(datetime AS timestamp) RANGE INTERVAL 10 MINUTE PRECEDING), \
               w2 AS (PARTITION BY cc_num order by cast(datetime AS timestamp) RANGE INTERVAL 1 WEEK PRECEDING) \
        ) 
    """
    transactions_df.registerTempTable('transactions_df')
    aggregated_features = spark.sql(query)
    return aggregated_features


def write_to_s3(args: Namespace, aggregated_features: DataFrame) -> None:
    logger.info('[Write Aggregated Features to S3]')
    aggregated_features.coalesce(1) \
                       .write.format('com.databricks.spark.csv') \
                       .option('header', True) \
                       .mode('overwrite') \
                       .option('sep', ',') \
                       .save('s3a://' + os.path.join(args.s3_output_bucket, args.s3_output_key_prefix))
    
def group_by_card_number(aggregated_features: DataFrame) -> DataFrame: 
    logger.info('[Group Aggregated Features by Card Number]')
    window = Window.partitionBy('cc_num').orderBy(desc('datetime'))
    sorted_df = aggregated_features.withColumn('rank', dense_rank().over(window))
    grouped_df = sorted_df.filter(sorted_df.rank == 1).drop(sorted_df.rank)
    sliced_df = grouped_df.select('cc_num', 'num_trans_last_1w', 'avg_amt_last_1w')
    return sliced_df


def transform_row(sliced_df: DataFrame) -> list:
    logger.info('[Transform Spark DataFrame Row to SageMaker Feature Store Record]')
    records = []
    for row in sliced_df.rdd.collect():
        record = []
        cc_num, num_trans_last_1w, avg_amt_last_1w = row
        if cc_num:
            record.append({'ValueAsString': str(cc_num), 'FeatureName': 'cc_num'})
            record.append({'ValueAsString': str(num_trans_last_1w), 'FeatureName': 'num_trans_last_1w'})
            record.append({'ValueAsString': str(round(avg_amt_last_1w, 2)), 'FeatureName': 'avg_amt_last_1w'})
            records.append(record)
    return records


def write_to_feature_store(records: list) -> None:
    logger.info('[Write Grouped Features to SageMaker Feature Store]')
    success, fail = 0, 0
    for record in records:
        event_time_feature = {
                'FeatureName': 'trans_time',
                'ValueAsString': str(int(round(time.time())))
            }
        record.append(event_time_feature)
        response = feature_store_client.put_record(FeatureGroupName=FEATURE_GROUP, Record=record)
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            success += 1
        else:
            fail += 1
    logger.info('Success = {}'.format(success))
    logger.info('Fail = {}'.format(fail))
    assert success == TOTAL_UNIQUE_USERS
    assert fail == 0

def transform_row_train(df: DataFrame) -> list:
    logger.info('[Transform Spark DataFrame Row to SageMaker Feature Store Record]')
    records = []
    for row in df.rdd.collect():
        record = []
        tid,cc_num, amount, fraud_label, num_trans_last_10m, avg_amt_last_10m, num_trans_last_1w, avg_amt_last_1w,datetime, amt_ratio1,amt_ratio2,count_ratio = row
        if tid:
            record.append({'ValueAsString': str(tid), 'FeatureName': 'tid'})
            record.append({'ValueAsString': str(cc_num), 'FeatureName': 'cc_num'})
            record.append({'ValueAsString': str(round(amount, 2)), 'FeatureName': 'amount'})
            record.append({'ValueAsString': str(fraud_label), 'FeatureName': 'fraud_label'})
            record.append({'ValueAsString': str(num_trans_last_1w), 'FeatureName': 'num_trans_last_1w'})
            record.append({'ValueAsString': str(round(avg_amt_last_1w, 2)), 'FeatureName': 'avg_amt_last_1w'})
            record.append({'ValueAsString': str(datetime), 'FeatureName': 'datetime'})
            record.append({'ValueAsString': str(num_trans_last_10m), 'FeatureName': 'num_trans_last_10m'})
            record.append({'ValueAsString': str(round(avg_amt_last_10m, 2)), 'FeatureName': 'avg_amt_last_10m'})
            record.append({'ValueAsString': str(round(amt_ratio1, 2)), 'FeatureName': 'amt_ratio1'})
            record.append({'ValueAsString': str(round(amt_ratio2, 2)), 'FeatureName': 'amt_ratio2'})
            record.append({'ValueAsString': str(round(count_ratio, 2)), 'FeatureName': 'count_ratio'})
            records.append(record)
    return records

def write_to_train_feature_store(records: list) -> None:
    logger.info('[Write Grouped Features to SageMaker Feature Store]')
    success, fail = 0, 0
    for record in records:
        event_time_feature = {
                'FeatureName': 'datetime',
                'ValueAsString': str(int(round(time.time())))
            }
        record.append(event_time_feature)
        response = feature_store_client.put_record(FeatureGroupName='cc-train-chime-fg', Record=record)
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            success += 1
        else:
            fail += 1
    logger.info('Success = {}'.format(success))
    logger.info('Fail = {}'.format(fail))
    assert success == TOTAL_UNIQUE_USERS
    assert fail == 0
    
def run_spark_job():
    spark = SparkSession.builder.appName('PySparkJob').getOrCreate()
    args = parse_args()
    schema = define_schema()
    aggregated_features = aggregate_features(args, schema, spark)
    records = transform_row_train(aggregated_features)
    write_to_train_feature_store(records)
#     write_to_s3(args, aggregated_features)
#     sliced_df = group_by_card_number(aggregated_features)
#     records = transform_row(sliced_df)
#     write_to_feature_store(records)

    
    
if __name__ == '__main__':
    run_spark_job()
