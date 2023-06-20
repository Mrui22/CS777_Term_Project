from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import length, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, VectorAssembler
from pyspark.ml.classification import NaiveBayes, MultilayerPerceptronClassifier, LogisticRegression, LinearSVC
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import os
import sys
import os.path

from pyspark.sql.types import StructType, StructField, StringType

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

sc = SparkContext.getOrCreate()
spark = SparkSession.builder.appName('Spam Detector').getOrCreate()
data_path = 'SMSSpam/SMSSpam'
pipline_path = 'Pipeline_model/pipeline_model'
model_path = 'ML_model/ml_model'

def evaluate_model(classifier_name, test_result):
    evaluator_acc = MulticlassClassificationEvaluator(metricName='accuracy')
    evaluator_f1 = MulticlassClassificationEvaluator(metricName='f1')
    evaluator_recall = MulticlassClassificationEvaluator(metricName='weightedRecall')

    acc = evaluator_acc.evaluate(test_result)
    f1 = evaluator_f1.evaluate(test_result)
    recall = evaluator_recall.evaluate(test_result)

    print(f"{classifier_name} Accuracy: {acc}")
    print(f"{classifier_name} F1 score: {f1}")
    print(f"{classifier_name} Recall: {recall}")

    return f1

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
    # data = spark.read.option("header","true").csv(data_path)
    data = spark.read.csv(data_path, inferSchema=True, sep='\t')
    data.printSchema()

    print("With special characters")
    print("=======================================================================================")
    data = data.withColumnRenamed('_c0', 'class').withColumnRenamed('_c1', 'text')
    data = data.withColumn('length', length(data['text']))
    data.show()
    tokenizer = Tokenizer(inputCol='text', outputCol='tokens')
    stop_word_remover = StopWordsRemover(inputCol='tokens', outputCol='stop_tokens')
    count_vec = CountVectorizer(inputCol='stop_tokens', outputCol='count_vec')
    idf = IDF(inputCol='count_vec', outputCol='tf_idf')
    ham_spam_to_numeric = StringIndexer(inputCol='class', outputCol='label')
    assembler = VectorAssembler(inputCols=['tf_idf', 'length'], outputCol='features')

    classifier_list = ['NaiveBayes', 'LogisticRegression', 'SVC']
    max_f1 = 0
    best_pipeline_model = None
    best_model = None

    for classifier_name in classifier_list:
        classifier = None
        if classifier_name == 'NaiveBayes':
            classifier = NaiveBayes()
        elif classifier_name == 'LogisticRegression':
            classifier = LogisticRegression()
        elif classifier_name == 'SVC':
            classifier = LinearSVC()
        try:
            if classifier is not None:
                pipe = Pipeline(stages=[ham_spam_to_numeric, tokenizer, stop_word_remover, count_vec, idf, assembler])

                clean_data = pipe.fit(data)
                clean_data = clean_data.transform(data)
                clean_data = clean_data.select('label', 'features')

                train_data, test_data = clean_data.randomSplit([0.8, 0.2])
                spam_detector = classifier.fit(train_data)
                test_result = spam_detector.transform(test_data)

                f1 = evaluate_model(classifier_name, test_result)
                if f1 > max_f1:
                    best_pipeline_model = pipe
                    best_model = classifier
        except Exception as e:
            print(e)

    if not os.path.exists(pipline_path):
        print("Saving pipeline")
        best_pipeline_model.save(pipline_path)
    if not os.path.exists(model_path):
        print("Saving ML model")
        best_pipeline_model.save(model_path)

    print("Removing special characters")
    print("=======================================================================================")
    columns_with_special_chars = ["text"]
    for column in columns_with_special_chars:
        data = data.withColumn(column, regexp_replace(column, "[^a-zA-Z0-9 ]", ""))

    data = data.withColumn('length', length(data['text']))
    data.show()
    tokenizer = Tokenizer(inputCol='text', outputCol='tokens')
    stop_word_remover = StopWordsRemover(inputCol='tokens', outputCol='stop_tokens')
    count_vec = CountVectorizer(inputCol='stop_tokens', outputCol='count_vec')
    idf = IDF(inputCol='count_vec', outputCol='tf_idf')
    ham_spam_to_numeric = StringIndexer(inputCol='class', outputCol='label')
    assembler = VectorAssembler(inputCols=['tf_idf', 'length'], outputCol='features')

    for classifier_name in classifier_list:
        classifier = None
        if classifier_name == 'NaiveBayes':
            classifier = NaiveBayes()
        elif classifier_name == 'LogisticRegression':
            classifier = LogisticRegression()
        elif classifier_name == 'SVC':
            classifier = LinearSVC()
        try:
            if classifier is not None:
                pipe = Pipeline(stages=[ham_spam_to_numeric, tokenizer, stop_word_remover, count_vec, idf, assembler])

                clean_data = pipe.fit(data)
                clean_data = clean_data.transform(data)
                clean_data = clean_data.select('label', 'features')

                train_data, test_data = clean_data.randomSplit([0.8, 0.2])
                spam_detector = classifier.fit(train_data)
                test_result = spam_detector.transform(test_data)

                f1 = evaluate_model(classifier_name, test_result)
        except Exception as e:
            print(e)

    sc.stop()