# cell 1 install pyspark
!pip install pyspark


# cell 2 imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan
from pyspark.sql.types import NumericType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator
import time
import json
import pandas as pd
import builtins

print("libraries imported")


# cell 3 upload and load data
from google.colab import files

print("upload your data file (csv, json, or txt):")
uploaded = files.upload()

filename = list(uploaded.keys())[0]
print(filename + " has been uploaded")

spark = SparkSession.builder.appName("dataprocessing").master("local[*]").config("spark.driver.memory", "4g").getOrCreate()

if filename.endswith(".csv"):
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(filename)
    print("loaded as csv")
elif filename.endswith(".json"):
    df = spark.read.option("multiLine", "true").json(filename)
    print("loaded as json")
elif filename.endswith(".txt"):
    df = spark.read.option("header", "true").option("inferSchema", "true").option("delimiter", "\t").csv(filename)
    print("loaded as txt with tab delimiter")
else:
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(filename)
    print("loaded as csv by default")

print("data loaded")
print("rows " + str(df.count()))
print("columns " + str(len(df.columns)))
df.printSchema()
df.show(5)


# cell 4 user selects processing options
print("select processing options")

numclusters = int(input("enter number of clusters for kmeans (2 to 10): "))
numtrees = int(input("enter number of trees for random forest (10 to 100): "))
testsplit = int(input("enter test data percentage (10 to 40): "))

print("selected options:")
print("kmeans clusters: " + str(numclusters))
print("random forest trees: " + str(numtrees))
print("test split: " + str(testsplit) + " percent")


# cell 5 descriptive statistics
print("descriptive statistics")

numrows = df.count()
numcols = len(df.columns)
print("stat 1 data shape")
print("total rows " + str(numrows))
print("total columns " + str(numcols))

print("stat 2 data types")
for colname, dtype in df.dtypes:
    print(colname + " " + dtype)

print("stat 3 null values")
numericcols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
stringcols = [f.name for f in df.schema.fields if not isinstance(f.dataType, NumericType)]

nullexprs = []
for c in numericcols:
    nullexprs.append(count(when(col(c).isNull() | isnan(c), c)).alias(c))
for c in stringcols:
    nullexprs.append(count(when(col(c).isNull(), c)).alias(c))

if len(nullexprs) > 0:
    nullcounts = df.select(nullexprs).collect()[0]
    for colname in df.columns:
        if colname in nullcounts.asDict():
            nullcount = nullcounts[colname]
            nullpct = (nullcount / numrows) * 100
            print(colname + " " + str(nullcount) + " nulls " + str(round(nullpct, 2)) + " percent")

print("stat 4 numeric summary")
if len(numericcols) > 0:
    summarydf = df.select(numericcols).summary("min", "max", "mean", "stddev")
    summarydf.show()

print("all 4 stats completed")


# cell 6 data preparation
print("data preparation")

dfclean = df.dropna()
cleanrowcount = dfclean.count()
rowsremoved = numrows - cleanrowcount
print("rows after cleaning " + str(cleanrowcount))
print("rows removed " + str(rowsremoved))

if len(numericcols) < 2:
    print("error: need at least 2 numeric columns for ml")
else:
    targetcol = numericcols[-1]
    featurecols = numericcols[:-1]
    
    print("target column " + targetcol)
    print("feature columns " + str(featurecols))
    
    assembler = VectorAssembler(inputCols=featurecols, outputCol="features", handleInvalid="skip")
    dfml = assembler.transform(dfclean)
    
    trainsplit = (100 - testsplit) / 100
    testsplitval = testsplit / 100
    traindata, testdata = dfml.randomSplit([trainsplit, testsplitval], seed=42)
    print("training samples " + str(traindata.count()))
    print("testing samples " + str(testdata.count()))


# cell 7 ml job 1 linear regression
print("ml job 1 linear regression")

starttime = time.time()

lr = LinearRegression(featuresCol="features", labelCol=targetcol, maxIter=10)
lrmodel = lr.fit(traindata)

lrpredictions = lrmodel.transform(testdata)
evaluator = RegressionEvaluator(labelCol=targetcol, predictionCol="prediction", metricName="rmse")
lrrmse = evaluator.evaluate(lrpredictions)
lrr2 = evaluator.evaluate(lrpredictions, {evaluator.metricName: "r2"})

lrtime = time.time() - starttime

print("rmse " + str(round(lrrmse, 2)))
print("r2 score " + str(round(lrr2, 4)))
print("training time " + str(round(lrtime, 2)) + " seconds")

print("sample predictions")
lrpredictions.select(targetcol, "prediction").show(5)


# cell 8 ml job 2 kmeans clustering
print("ml job 2 kmeans clustering")

starttime = time.time()

clusterfeatures = featurecols[:2] if len(featurecols) >= 2 else featurecols
clusterassembler = VectorAssembler(inputCols=clusterfeatures, outputCol="clusterfeatures", handleInvalid="skip")
dfcluster = clusterassembler.transform(dfclean)

kmeans = KMeans(featuresCol="clusterfeatures", k=numclusters, seed=42)
kmeansmodel = kmeans.fit(dfcluster)

clustercenters = kmeansmodel.clusterCenters()
wssse = kmeansmodel.summary.trainingCost

kmeanstime = time.time() - starttime

print("number of clusters " + str(numclusters))
print("wssse " + str(round(wssse, 2)))
print("training time " + str(round(kmeanstime, 2)) + " seconds")

print("cluster centers")
for i, center in enumerate(clustercenters):
    print("cluster " + str(i) + " " + str([round(c, 4) for c in center]))

print("cluster distribution")
kmpredictions = kmeansmodel.transform(dfcluster)
kmpredictions.groupBy("prediction").count().orderBy("prediction").show()


# cell 9 ml job 3 random forest
print("ml job 3 random forest")

starttime = time.time()

rf = RandomForestRegressor(featuresCol="features", labelCol=targetcol, numTrees=numtrees, seed=42)
rfmodel = rf.fit(traindata)

rfpredictions = rfmodel.transform(testdata)
rfrmse = evaluator.evaluate(rfpredictions)
rfr2 = evaluator.evaluate(rfpredictions, {evaluator.metricName: "r2"})

rftime = time.time() - starttime

print("rmse " + str(round(rfrmse, 2)))
print("r2 score " + str(round(rfr2, 4)))
print("number of trees " + str(numtrees))
print("training time " + str(round(rftime, 2)) + " seconds")

print("feature importances")
importances = rfmodel.featureImportances.toArray()
for i, colname in enumerate(featurecols):
    if i < len(importances):
        print(colname + " " + str(round(importances[i], 4)))


# cell 10 ml job 4 correlation analysis
print("ml job 4 correlation analysis")

starttime = time.time()

print("correlation with target " + targetcol)
correlations = {}
for colname in featurecols[:10]:
    corr = dfclean.stat.corr(colname, targetcol)
    correlations[colname] = corr
    if abs(corr) > 0.5:
        strength = "strong"
    elif abs(corr) > 0.3:
        strength = "moderate"
    else:
        strength = "weak"
    print(colname + " " + str(round(corr, 4)) + " " + strength)

corrtime = time.time() - starttime

if len(correlations) > 0:
    strongest = builtins.max(correlations, key=lambda x: abs(correlations[x]))
    print("strongest predictor " + strongest + " correlation " + str(round(correlations[strongest], 4)))
print("analysis time " + str(round(corrtime, 2)) + " seconds")

print("all 4 ml jobs completed")


# cell 11 scalability benchmarking
print("scalability benchmarking")
print("testing with different core counts")

def runbenchmark(numcores, datapath, nclusters, ntrees):
    sparkbench = SparkSession.builder.appName("benchmark" + str(numcores)).master("local[" + str(numcores) + "]").config("spark.driver.memory", "4g").config("spark.sql.shuffle.partitions", str(numcores * 2)).getOrCreate()
    
    dfb = sparkbench.read.option("header", "true").option("inferSchema", "true").csv(datapath)
    dfb = dfb.dropna()
    
    numcols = [f.name for f in dfb.schema.fields if isinstance(f.dataType, NumericType)]
    featcols = numcols[:-1]
    targetc = numcols[-1]
    
    asm = VectorAssembler(inputCols=featcols, outputCol="features", handleInvalid="skip")
    dfmlb = asm.transform(dfb)
    
    start = time.time()
    
    LinearRegression(featuresCol="features", labelCol=targetc, maxIter=10).fit(dfmlb)
    
    clusterfeat = featcols[:2] if len(featcols) >= 2 else featcols
    clusterasm = VectorAssembler(inputCols=clusterfeat, outputCol="clusterfeatures", handleInvalid="skip")
    dfclusterb = clusterasm.transform(dfb)
    KMeans(featuresCol="clusterfeatures", k=nclusters, seed=42).fit(dfclusterb)
    
    RandomForestRegressor(featuresCol="features", labelCol=targetc, numTrees=ntrees, seed=42).fit(dfmlb)
    
    totaltime = time.time() - start
    
    return totaltime

corecounts = [1, 2, 4, 8]
benchmarkresults = {}

for cores in corecounts:
    print("testing with " + str(cores) + " cores")
    exectime = runbenchmark(cores, filename, numclusters, numtrees)
    benchmarkresults[cores] = exectime
    print("completed in " + str(round(exectime, 2)) + " seconds")

print("benchmark results")
basetime = benchmarkresults[1]
resultstable = []

print("cores\ttime\tspeedup\tefficiency")
for cores in corecounts:
    exectime = benchmarkresults[cores]
    speedup = basetime / exectime
    efficiency = (speedup / cores) * 100
    print(str(cores) + "\t" + str(round(exectime, 2)) + "\t" + str(round(speedup, 2)) + "\t" + str(round(efficiency, 1)) + "%")
    resultstable.append({"cores": cores, "time": round(exectime, 2), "speedup": round(speedup, 2), "efficiency": round(efficiency, 1)})

benchmarkdf = pd.DataFrame(resultstable)
print(benchmarkdf)


# cell 12 save results to google drive
print("saving results to google drive")

from google.colab import drive
drive.mount('/content/drive')

allresults = {
    "dataset": {
        "filename": str(filename),
        "rows": int(numrows),
        "columns": int(numcols)
    },
    "options": {
        "clusters": int(numclusters),
        "trees": int(numtrees),
        "testsplit": int(testsplit)
    },
    "statistics": {
        "numrows": int(numrows),
        "numcols": int(numcols),
        "numericcols": int(len(numericcols)),
        "cleanrows": int(cleanrowcount)
    },
    "mlresults": {
        "linearregression": {"rmse": float(round(lrrmse, 2)), "r2": float(round(lrr2, 4)), "time": float(round(lrtime, 2))},
        "kmeans": {"wssse": float(round(wssse, 2)), "clusters": int(numclusters), "time": float(round(kmeanstime, 2))},
        "randomforest": {"rmse": float(round(rfrmse, 2)), "r2": float(round(rfr2, 4)), "trees": int(numtrees), "time": float(round(rftime, 2))},
        "correlation": {"strongest": str(strongest) if len(correlations) > 0 else "none", "value": float(round(correlations.get(strongest, 0), 4)) if len(correlations) > 0 else 0.0}
    },
    "benchmark": resultstable
}

import os
savepath = "/content/drive/MyDrive/sparkresults"
os.makedirs(savepath, exist_ok=True)

with open(savepath + "/results.json", "w") as f:
    json.dump(allresults, f, indent=2)

benchmarkdf.to_csv(savepath + "/benchmark.csv", index=False)

print("results saved to " + savepath)
print("files created: results.json, benchmark.csv")


# cell 13 download results
print("download processed results")

from google.colab import files as colabfiles

colabfiles.download(savepath + "/results.json")
colabfiles.download(savepath + "/benchmark.csv")

print("project complete")
