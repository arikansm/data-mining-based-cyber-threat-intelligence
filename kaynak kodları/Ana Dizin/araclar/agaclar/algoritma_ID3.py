from math import log
import operator
import time
import os
import sys
from araclar.agaclar.agac_gorsellestir import *
import ayarlar.ayarlar as settings
def warn(*args, **kwargs):
	pass
import warnings
warnings.warn = warn
from sklearn.metrics import classification_report

"""ağacı dosyadan okur"""
def saveTreeToFile(sma_tree, filename):
	import pickle
	fw = open(filename, 'wb')
	pickle.dump(sma_tree, fw)
	fw.close()

"""ağacı dosyadan okur"""
def getTreeFromFile(filename):
	import pickle
	fr = open(filename, 'rb')
	sma_tree = pickle.load(fr)
	return sma_tree

"""ağaç oluşturma için main fonksiyonudur."""
def Tree(data,t_data, clmns, save_operation="kdd"):
	#öğrenme ve sınıflandırma süresini ölçmek için başlangıç noktası olarak kullanılacaktır.
	learningStartTime = time.time()

	#sütun nesnesi değerlerinin değişmemesi için geçici bir değişken üzerinden ağaç oluşturulur.
	tempC = clmns[:]

	#ağaç oluşturulur.
	sma_tree = buildTree(data, tempC)

	#öğrenme süresi hesaplanır.
	learningEndTime = time.time()
	learningTime = learningEndTime - learningStartTime

	if save_operation == "kdd":
		#oluşturulan ağaç dosyaya kaydedilir.
		treeFileName = settings.id3_tree_file_path
		saveTreeToFile(sma_tree, treeFileName)
		# sınıflandırma yapılır ve doğruluk ile f1-score gibi bilgiler hesaplanır.
		accuracy, confisionMatrix = prediction(treeFileName, clmns, t_data)
	if save_operation == "nsl":
		treeFileName = settings.nsl_id3_tree_file_path
		saveTreeToFile(sma_tree, treeFileName)
		accuracy, confisionMatrix = prediction(treeFileName, clmns, t_data)
	if save_operation == "fold":
		accuracy, confisionMatrix = prediction_with_tree(sma_tree, clmns, t_data)


	#sınıflandırma süresi hesaplanır.
	testingTime = time.time() - learningEndTime

	return sma_tree, accuracy, confisionMatrix, learningTime, testingTime

"""ağaç sözlük (dictionary) veri yapısı ile oluşturulur."""
def buildTree(data, clmns):
	#son indeksteki değerler sınıftır. Bunlar sınıf dizinine aktarılır.
	siniflar = [d[-1] for d in data]

	#tüm sınıflar aynı ise geriye bu sınıf döndürülür. Çünkü üzerinde işlem yapmaya gerek kalmamıştır.
	if siniflar.count(siniflar[0]) == len(siniflar):
		return siniflar[0]

	#incelenecek sütun kalmamış ve sınıf değerlerinin hepside aynı değilse en çok sayıda olan döndürülür.
	if len(data[0]) == 1:
		return vote_class(siniflar)

	#en iyi özellik seçilerek indeks döndürülür.
	selectedFeature_index = selectBestFeature(data)

	#ağaçta göstermek için en iyi indeksin adı bulunur.
	selectedFeatureLabel = clmns[selectedFeature_index]

	#en iyi özellik adı köke konularak ağaç oluşturmaya başlanır.
	sma_tree = {selectedFeatureLabel: {}}

	#kullanılan özellik silinir.
	del (clmns[selectedFeature_index])

	#seçilen özelliğin değerleri bir değişkene kaydedilir.
	featureValues = [d[selectedFeature_index] for d in data]

	#özelliğin benzersiz olan değerleri bir değişkene atılır.
	uniqueFeatureValues = set(featureValues)

	#her benzersiz olan değer, sözlük veri tipinde olan ağaca eklenir.
	for v in uniqueFeatureValues:
		_clmns = clmns[:]
		#eklenen değerler için veri parçalanır ve yine buildTree fonksiyonu için gönderilir. Böylece yinelemeli olarak ağaç oluşturulur.
		sma_tree[selectedFeatureLabel][v] = buildTree(splitData(data, selectedFeature_index, v), _clmns)
	return sma_tree

"""Tüm özellikler kullanılmış ancak sınıf değerlerinin hepsi aynı değilse en çok olan sınıf değeri seçilir."""
def vote_class(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	
	max_class_label = "None"
	max_class_count = -1
	for k, v in classCount.items():
		if v > max_class_count:
			max_class_label = k
			max_class_count = v

	return max_class_label
	
"""Son indeksinde sınıf bilgisi bulunan veri için en iyi kazancı elde ettirecek özellik seçilerek indeksi döndürülür."""
def selectBestFeature(data):
	numberOfFeatures = len(data[0]) - 1

	#genel entropi hesaplanır.
	INFO_D = calc_I(data)

	#her özellik için kazanç hesaplanır. en iyi olanın indeksi döndürülür.
	bestGain = 0.0
	bestFeature = -1
	for i in range(numberOfFeatures):
		#analiz edilecek indeksin benzersiz değerleri bulunur.
		featureValues = [d[i] for d in data]
		uniqueFeaturesValues = set(featureValues)

		INFO_feature_D = 0.0
		#her benzersiz değer için entropi hesaplanır.
		for value in uniqueFeaturesValues:
			subData = splitData(data, i, value)
			p = len(subData) / float(len(data))
			INFO_feature_D = INFO_feature_D + p * calc_I(subData)

		#kazanç hesaplanır
		infoGain = INFO_D - INFO_feature_D

		#en iyi olan seçilir.
		if infoGain > bestGain:
			bestGain = infoGain
			bestFeature = i

	return bestFeature

"""veri için entropi hesaplanır"""
def calc_I(data):
	recordNumber = len(data)
	classCounts = {}
	for line in data:
		currentClass= line[-1]
		if currentClass not in classCounts.keys():
			classCounts[currentClass] = 0
		classCounts[currentClass] += 1

	I = 0.0
	# -p(xi)*log2p(xi)
	for key in classCounts: 
		p = float(classCounts[key]) / recordNumber  
		I = I - p * log(p, 2)
	return I

"""veriyi özelliğin değerine göre ayırır"""
def splitData(data, feature, val):
	new_data = []
	for line in data:
		if line[feature] == val:
			new_line = line[:feature]
			new_line.extend(line[feature + 1:])
			new_data.append(new_line)
	return new_data

"""ağaç derinliğini getirir."""
def getTreeDepth(sma_tree):
	maxDepth=0
	firstStr=list(sma_tree.keys())[0]
	secondDict=sma_tree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			thisDepth=1+getTreeDepth(secondDict[key])
		else:
			thisDepth=1
		if thisDepth>maxDepth:
			maxDepth=thisDepth
	return maxDepth

"""toplam yaprak sayısını hesaplar"""
def getNumLeafs(sma_tree):
	numLeafs=0
	firstStr=list(sma_tree.keys())[0]
	secondDict=sma_tree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			numLeafs+= getNumLeafs(secondDict[key])
		else:
			numLeafs+=1
	return numLeafs

"""öğrenilmiş ağaç ile sınıflandırma yapar"""
def prediction(storedTreeFile, labels, data):
	trainingTree = getTreeFromFile(storedTreeFile)

	correctNum = 0
	totalNum = 0

	class_true = []
	class_predicted = []

	for line in data:
		totalNum += 1

		d = line[:-1]
		c = line[-1]

		#sınıf tahmin edilir.
		predictedClass = classifyLine(trainingTree, labels, d)

		#doğru tahmin edilenlerin sayısı hesaplanır.
		if c == predictedClass:
			correctNum += 1
		
		#gerçek sınıflar ve tahminler ayrı dizilere aktarılır.
		class_true.append(c)
		class_predicted.append(predictedClass)

	#doğruluk hesaplanır
	accuracy = correctNum / float(totalNum)

	#f1-score gibi bilgiler hesaplanır
	confisionMatrix = classification_report(class_true, class_predicted, digits=4)
	confisionMatrixA = confisionMatrix[:confisionMatrix.find(settings.unknown_class_value)]
	confisionMatrixB = confisionMatrix[confisionMatrix.find(settings.unknown_class_value):][confisionMatrix[confisionMatrix.find(settings.unknown_class_value):].find("\n")+1:]
	confisionMatrix = confisionMatrixA + confisionMatrixB
	confisionMatrix = confisionMatrix[:confisionMatrix.find("avg / total")]
	
	return accuracy,confisionMatrix


def prediction_with_tree(trainingTree, labels, data):
	correctNum = 0
	totalNum = 0

	class_true = []
	class_predicted = []

	for line in data:
		totalNum += 1

		d = line[:-1]
		c = line[-1]

		# sınıf tahmin edilir.
		predictedClass = classifyLine(trainingTree, labels, d)

		# doğru tahmin edilenlerin sayısı hesaplanır.
		if c == predictedClass:
			correctNum += 1

		# gerçek sınıflar ve tahminler ayrı dizilere aktarılır.
		class_true.append(c)
		class_predicted.append(predictedClass)

	# doğruluk hesaplanır
	accuracy = correctNum / float(totalNum)

	# f1-score gibi bilgiler hesaplanır
	confisionMatrix = classification_report(class_true, class_predicted, digits=4)
	confisionMatrixA = confisionMatrix[:confisionMatrix.find(settings.unknown_class_value)]
	confisionMatrixB = confisionMatrix[confisionMatrix.find(settings.unknown_class_value):][
					   confisionMatrix[confisionMatrix.find(settings.unknown_class_value):].find("\n") + 1:]
	confisionMatrix = confisionMatrixA + confisionMatrixB
	confisionMatrix = confisionMatrix[:confisionMatrix.find("avg / total")]

	return accuracy, confisionMatrix

"""öğrenilmiş ağaç içinde yinelemeli gezerek ilgili veri için sınıfı döndürür"""
def classifyLine(sma_tree, clmns, test_line):
	firstStr = list(sma_tree.keys())[0]
	secondDict = sma_tree[firstStr]
	featatureIndex = clmns.index(firstStr)
	predictedClass = settings.unknown_class_value
	for key in secondDict.keys():
		if test_line[featatureIndex] == key:
			sub_key = secondDict[key]
			if type(sub_key).__name__ == 'dict':
				predictedClass = classifyLine(sub_key, clmns, test_line)
			elif (sub_key != "" and len(sub_key) > 0):
				predictedClass = sub_key

	return predictedClass
