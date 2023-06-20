from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import length, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, VectorAssembler
from pyspark.ml.classification import NaiveBayes, MultilayerPerceptronClassifier, LogisticRegression, LinearSVC
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel

import json
import os
import sys

from pyspark.sql.types import StructType, StructField, StringType

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sc = SparkContext.getOrCreate()
spark = SparkSession.builder.appName('Spam Detector').getOrCreate()
data_path = 'SMSSpam/SMSSpam'
pipline_path = 'Pipeline_model/pipeline_model'
model_path = 'ML_model/ml_model'


def predict_spam(message, pipe, spam_detector):
    schema = StructType([StructField("text", StringType(), True)])
    new_data = spark.createDataFrame([(message,)], schema=schema)
    new_data = pipe.transform(new_data)
    new_data = new_data.select('features')

    prediction = spam_detector.transform(new_data)
    prediction = prediction.collect()[0]['prediction']

    if prediction == 0:
        print("This message is likely not spam (ham).")
    elif prediction == 1:
        print("This message is likely spam.")


if __name__ == "__main__":
    pipe = PipelineModel.load(pipline_path)

    spam_detector = LinearSVC.load(model_path)

    metadata_path = model_path + "/metadata/part-00000"
    metadata = sc.textFile(metadata_path)
    metadata_json = metadata.collect()[0]
    metadata_dict = json.loads(metadata_json)

    print("Model type:", metadata_dict['class'])

    message = str(input("Enter the suspicious message: "))
    predict_spam(message, pipe, spam_detector)
