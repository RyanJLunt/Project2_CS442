#https://piotrszul.github.io/spark-tutorial/notebooks/3.1_ML-Introduction.html
#https://github.com/PacktPublishing/Apache-Spark-2-for-Beginners/blob/master/Code_Chapter%207/Code/Python/PythonSparkMachineLearning.py



import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import Normalizer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA

import sys
import pandas as pd
import os

#delete prior data to store in new data.
if os.path.exists("results.txt"):
	os.remove("results.txt")

#print("Stage 1: Beginning set up")
#Spark / data set up.
spark = SparkSession.builder.appName('wineQuality-ml').getOrCreate()
DataFrame = spark.read.csv('winequality-white.csv', header=True, inferSchema=True, sep=';')
training_Data, testing_Data = DataFrame.randomSplit([0.8, 0.2])

#print("Stage 2:")
# set up structure from dataframe.
featureCols = [columns for columns in DataFrame.columns if columns != "quality"]
assembler = VectorAssembler(inputCols=['fixed acidity', \
'volatile acidity', \
'citric acid', \
'residual sugar', \
'chlorides', \
'free sulfur dioxide', \
'total sulfur dioxide', \
'density', \
'pH', \
'sulphates', \
'alcohol'], outputCol='features')
clean_Data = Normalizer(inputCol='features', outputCol='clean_Features')


if len(sys.argv) == 2:
	if sys.argv[1] == 'lr':
		print("Solving for only LR")
		Linear_Regression = LinearRegression(featuresCol='clean_Features', labelCol='quality', regParam=0.0, maxIter=250, elasticNetParam=0.3)
		grid = ParamGridBuilder() \
			.addGrid(Linear_Regression.regParam, [0.0, 0.3, 0.6]) \
			.addGrid(Linear_Regression.elasticNetParam, [0.2, 0.5, 0.8]).build()
		
		RSME_Evaluate = RegressionEvaluator(labelCol='quality', predictionCol='prediction', metricName='rmse')
		R2_Evaluate = RegressionEvaluator(labelCol='quality', predictionCol='prediction', metricName='r2')
		LR_Pipeline = Pipeline(stages=[assembler, clean_Data, Linear_Regression])
		RSME_Cross = CrossValidator(estimator = LR_Pipeline, estimatorParamMaps = grid, evaluator = RSME_Evaluate, numFolds = 4)
		R2_Cross = CrossValidator(estimator = LR_Pipeline, estimatorParamMaps = grid, evaluator = R2_Evaluate, numFolds = 5)

		print("Machine learning pt. 1: Training models...")
		LR_Model= LR_Pipeline.fit(training_Data)
		RMSE_Model = RSME_Cross.fit(training_Data)
		R2_Model = R2_Cross.fit(training_Data)

		print("Machine learning pt. 2: Testing models...")
		LR_Data = LR_Model.transform(testing_Data)
		RSME_Data = RMSE_Model.transform(testing_Data)
		R2_Data = R2_Model.transform(testing_Data)

		print("Machine learning pt. 3: Readying data...")
		final_RMSE_Data = str(RSME_Data)
		final_R2_Data = str(R2_Data)

		final_LR_RSME = str(RSME_Evaluate.evaluate(LR_Data))
		final_LR_R2 = str(R2_Evaluate.evaluate(LR_Data))

		LR_Accuracy1 = (1 - RSME_Evaluate.evaluate(LR_Data))
		LR_Accuracy2 = float("{0:.2f}".format(LR_Accuracy1))
		LR_Accuracy3 = str(LR_Accuracy2)

		print("RESULTS FOR BOTH MODELS:")
		print("***********************************************************************************************************************************")
		print("Linear regression:")
		print("Linear regression RSMRE score: " + final_LR_RSME)
		print("Linear regression accuracy: " + LR_Accuracy3)
		print("Linear regression R^2 score: " + final_LR_R2)
		print("***********************************************************************************************************************************")

		f = open("results.txt", "a")
		f.write("Linear regression:")
		f.write("\n")
		f.write("Linear regression RSMRE score: " + final_LR_RSME)
		f.write("\n")
		f.write("Linear regression accuracy: " + LR_Accuracy3)
		f.write("\n")
		f.write("Linear regression R^2 score: " + final_LR_R2)
		f.close()

		print("Stored data in results.txt")
		print("Ending program - successfully returned 0.")
		spark.stop()
		exit()

	if sys.argv[1] == 'rf':
		print("Solving for only RF")
		
		Linear_Regression = LinearRegression(featuresCol='clean_Features', labelCol='quality', regParam=0.0, maxIter=250, elasticNetParam=0.3)
		Random_Forest = RandomForestRegressor(featuresCol='clean_Features', labelCol='quality', numTrees=150, maxBins=100, maxDepth=25)
		grid = ParamGridBuilder() \
			.addGrid(Linear_Regression.regParam, [0.0, 0.3, 0.6]) \
			.addGrid(Linear_Regression.elasticNetParam, [0.2, 0.5, 0.8]).build()

		print("Preparing to utilize machine learning...")
		RSME_Evaluate = RegressionEvaluator(labelCol='quality', predictionCol='prediction', metricName='rmse')
		R2_Evaluate = RegressionEvaluator(labelCol='quality', predictionCol='prediction', metricName='r2')
		RF_Pipeline = Pipeline(stages=[assembler, clean_Data, Random_Forest])
		LR_Pipeline = Pipeline(stages=[assembler, clean_Data, Linear_Regression])
		RSME_Cross = CrossValidator(estimator = LR_Pipeline, estimatorParamMaps = grid, evaluator = RSME_Evaluate, numFolds = 4)
		R2_Cross = CrossValidator(estimator = LR_Pipeline, estimatorParamMaps = grid, evaluator = R2_Evaluate, numFolds = 5)

		print("Machine learning pt. 1: Training models...")
		RF_Model = RF_Pipeline.fit(training_Data)
		RMSE_Model = RSME_Cross.fit(training_Data)
		R2_Model = R2_Cross.fit(training_Data)

		print("Machine learning pt. 2: Testing models...")
		RF_Data = RF_Model.transform(testing_Data)
		RSME_Data = RMSE_Model.transform(testing_Data)
		R2_Data = R2_Model.transform(testing_Data)

		print("Machine learning pt. 3: Readying data...")
		final_RMSE_Data = str(RSME_Data)
		final_R2_Data = str(R2_Data)
		final_RF_RSME = str(RSME_Evaluate.evaluate(RF_Data))
		final_RF_R2 = str(R2_Evaluate.evaluate(RF_Data))
		RF_Accuracy1 = (1 - RSME_Evaluate.evaluate(RF_Data))
		RF_Accuracy2 = float("{0:.2f}".format(RF_Accuracy1))
		RF_Accuracy3 = str(RF_Accuracy2)

		print("***********************************************************************************************************************************")
		print("Random Forest:")
		print("Random Forest RSMRE score: " + final_RF_RSME)
		print("Random Forest accuracy: " + RF_Accuracy3)
		print("Random Forest R^2 score: " + final_RF_R2)
		print("***********************************************************************************************************************************")

		f = open("results.txt", "a")
		f.write("Random Forest:")
		f.write("\n")
		f.write("Random Forest RSMRE score: " + final_RF_RSME)
		f.write("\n")
		f.write("Random Forest accuracy: " + RF_Accuracy3)
		f.write("\n")
		f.write("Random Forest R^2 score: " + final_RF_R2)
		f.close()

		print("Stored data in results.txt")
		print("Ending program - successfully returned 0.")
		spark.stop()
		exit()

else:
	print("No command line arguments given - performing analysis for both modes.")

print("Beginning set up of methods; grid.")

Linear_Regression = LinearRegression(featuresCol='clean_Features', labelCol='quality', regParam=0.0, maxIter=250, elasticNetParam=0.3)
Random_Forest = RandomForestRegressor(featuresCol='clean_Features', labelCol='quality', numTrees=150, maxBins=100, maxDepth=25)

#grid creation with parameters for evaluation / valudators.
grid = ParamGridBuilder() \
	.addGrid(Linear_Regression.regParam, [0.0, 0.3, 0.6]) \
	.addGrid(Linear_Regression.elasticNetParam, [0.2, 0.5, 0.8]).build()

print("Preparing to utilize machine learning...")
# creating pipelines, evaluators, and cross validation

RSME_Evaluate = RegressionEvaluator(labelCol='quality', predictionCol='prediction', metricName='rmse')
R2_Evaluate = RegressionEvaluator(labelCol='quality', predictionCol='prediction', metricName='r2')
LR_Pipeline = Pipeline(stages=[assembler, clean_Data, Linear_Regression])
RF_Pipeline = Pipeline(stages=[assembler, clean_Data, Random_Forest])
RSME_Cross = CrossValidator(estimator = LR_Pipeline, estimatorParamMaps = grid, evaluator = RSME_Evaluate, numFolds = 4)
R2_Cross = CrossValidator(estimator = LR_Pipeline, estimatorParamMaps = grid, evaluator = R2_Evaluate, numFolds = 5)


print("Machine learning pt. 1: Training models...")
LR_Model= LR_Pipeline.fit(training_Data)
RF_Model = RF_Pipeline.fit(training_Data)
RMSE_Model = RSME_Cross.fit(training_Data)
R2_Model = R2_Cross.fit(training_Data)


print("Machine learning pt. 2: Testing models...")
RF_Data = RF_Model.transform(testing_Data)
LR_Data = LR_Model.transform(testing_Data)
RSME_Data = RMSE_Model.transform(testing_Data)
R2_Data = R2_Model.transform(testing_Data)

print("Machine learning pt. 3: Readying data...")
final_RMSE_Data = str(RSME_Data)
final_R2_Data = str(R2_Data)

final_RF_RSME = str(RSME_Evaluate.evaluate(RF_Data))
final_LR_RSME = str(RSME_Evaluate.evaluate(LR_Data))
final_RF_R2 = str(R2_Evaluate.evaluate(RF_Data))
final_LR_R2 = str(R2_Evaluate.evaluate(LR_Data))


LR_Accuracy1 = (1 - RSME_Evaluate.evaluate(LR_Data))
RF_Accuracy1 = (1 - RSME_Evaluate.evaluate(RF_Data))
LR_Accuracy2 = float("{0:.2f}".format(LR_Accuracy1))
RF_Accuracy2 = float("{0:.2f}".format(RF_Accuracy1))
LR_Accuracy3 = str(LR_Accuracy2)
RF_Accuracy3 = str(RF_Accuracy2)
error_Output = "% accurate."

print("RESULTS FOR BOTH MODELS:")
print("***********************************************************************************************************************************")
print("Linear regression:")
print("Linear regression RSMRE score: " + final_LR_RSME)
print("Linear regression accuracy: " + LR_Accuracy3)
print("Linear regression R^2 score: " + final_LR_R2)
print("Random Forest:")
print("Random Forest RSMRE score: " + final_RF_RSME)
print("Random Forest accuracy: " + RF_Accuracy3)
print("Random Forest R^2 score: " + final_RF_R2)
print("***********************************************************************************************************************************")

#saving results with file I/o

f = open("results.txt", "a")
f.write("Linear regression:")
f.write("\n")
f.write("Linear regression RSMRE score: " + final_LR_RSME)
f.write("\n")
f.write("Linear regression accuracy: " + LR_Accuracy3)
f.write("\n")
f.write("Linear regression R^2 score: " + final_LR_R2)
f.write("\n")

f.write("Random Forest:")
f.write("\n")
f.write("Random Forest RSMRE score: " + final_RF_RSME)
f.write("\n")
f.write("Random Forest accuracy: " + RF_Accuracy3)
f.write("\n")
f.write("Random Forest R^2 score: " + final_RF_R2)
f.close()

print("Stored data in results.txt")
print("Ending program - successfully returned 0.")
spark.stop()




