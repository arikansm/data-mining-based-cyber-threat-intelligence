from math import log
import operator
import time
import os
import sys
from collections import OrderedDict
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
	#ağaç üzerinde oluşan gereksiz nesneler silinir.
	pruneTree(sma_tree)

	#öğrenme süresi hesaplanır.
	learningEndTime = time.time()
	learningTime = learningEndTime - learningStartTime

	if save_operation == "kdd":
		#oluşturulan ağaç dosyaya kaydedilir.
		treeFileName = settings.cart_tree_file_path
		saveTreeToFile(sma_tree, treeFileName)
		# sınıflandırma yapılır ve doğruluk ile f1-score gibi bilgiler hesaplanır.
		accuracy, confisionMatrix = prediction(treeFileName, clmns, t_data)
	if save_operation == "nsl":
		treeFileName = settings.nsl_cart_tree_file_path
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
	split = selectBestFeature(data)
	selectedFeature = split['index']

	#ağaçta göstermek için en iyi indeksin adı bulunur.
	selectedFeatureLabel = clmns[selectedFeature]
	
	#ağaçta göstermek için en iyi bölme noktasının değeri alınır.
	v = split['value']

	#en iyi özellik adı köke konularak ağaç oluşturmaya başlanır.
	sma_tree = {selectedFeatureLabel: {}}

	#kullanılan özellik silinir.
	del (clmns[selectedFeature])


	_clmns = clmns[:]
	__clmns = clmns[:]

	#en iyi özellik ve bölme noktası ile veri ikiye ayrılır.
	lower, eq_or_upper = splitData(data, selectedFeature, v)

	#eğer ikiye ayırmak üzere en iyi değer seçildiğinde veri ikiye ayrılmıyorsa en çok sınıf değerini geri döndür.
	if len(lower) == 0:
		return vote_class([row[-1] for row in eq_or_upper])

	if len(eq_or_upper) == 0:
		return vote_class([row[-1] for row in lower])

	#bölme noktasından veri parçalanır ve yine buildTree fonksiyonu için gönderilir. Böylece yinelemeli olarak ağaç oluşturulur.
	sma_tree[selectedFeatureLabel]["<" + str(v)] = buildTree(lower, _clmns)
	sma_tree[selectedFeatureLabel][">" + str(v)] = buildTree(eq_or_upper, __clmns)

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

"""veri seçilmiş özelliğin belirlenen değerine göre ikiye ayrılır."""
def splitData(data, feature, val):
	lower = []
	eq_or_upper = []
	for line in data:
		if line[feature] < val:
			new_line = line[:feature]
			new_line.extend(line[feature + 1:])
			lower.append(new_line)
		else:
			new_line = line[:feature]
			new_line.extend(line[feature + 1:])
			eq_or_upper.append(new_line)
	return lower,eq_or_upper

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
		if key[0] == "<":
			if float(test_line[featatureIndex]) < float(key[1:]):
				if type(secondDict[key]).__name__ == 'dict':
					predictedClass = classifyLine(secondDict[key], clmns, test_line)
				else:
					predictedClass = secondDict[key]
		elif key[0] == ">":
			if float(test_line[featatureIndex]) >= float(key[1:]):
				if type(secondDict[key]).__name__ == 'dict':
					predictedClass = classifyLine(secondDict[key], clmns, test_line)
				else:
					predictedClass = secondDict[key]
	return predictedClass

#veri ikiye bölünür.
def splitDataForTest(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

"""veri için gini indeksi hesaplanır"""
def calc_G(groups, classes):
	#toplam kayıt sayısı hesaplanır.
	total_record_number = float(sum([len(group) for group in groups]))

	gini = 0.0
	for group in groups:
		#grup için toplam kayıt sayısı hesaplanır.
		size = float(len(group))

		if size == 0:
			continue

		score = 0.0
		for class_val in classes:
			#p hesaplanır. (sınıf sayısı / toplam sayı)
			p = [row[-1] for row in group].count(class_val) / size
			#p^2 değerleri hesaplanır.
			score += p * p
		#(1 - p1^2 - p2^2) * (size/total) hesaplanır.
		#iki grup için hesaplanan değerler toplanır.
		gini += (1.0 - score) * (size / total_record_number)
	return gini

"""Son indeksinde sınıf bilgisi bulunan veri için en iyi kazancı elde ettirecek özellik seçilerek indeksi döndürülür."""
def selectBestFeature(dataset):
	#sınıf değerleri bulunur.
	class_values = list(set(row[-1] for row in dataset))

	selected_index, selected_value, selected_score = None, None, float("inf")
	#tüm özellikler gezilir.
	#her özelliğin benzersiz değerleri bulunur. 
	#Bu değerler ile veriyi ikiye ayırıp gini indeksi hesaplanır.
	#en iyi gini indeksi için değerler döndürülür.
	for index in range(len(dataset[0])-1):
		dataColumn = [d[index] for d in dataset]
		dataColumnUnique = list(set(dataColumn))
		for unique_value in dataColumnUnique:
			groups = splitDataForTest(index, unique_value, dataset)
			gini = calc_G(groups, class_values)
			if gini < selected_score:
				selected_index, selected_value, selected_score = index, unique_value, gini
	return {'index':selected_index, 'value':selected_value, 'gini':gini}

"""verilen ağaç kesiti içinde gezinerek benzersiz olan sınıf sayılarını döndürür."""
def get_uniques(data):
	cls = []
	for key in data.keys():
		if type(data[key]).__name__ == 'dict':
			uniques = get_uniques(data[key])
			for unique in uniques:
				if unique not in cls:
					cls.append(unique)
		else:
			if len(cls) == 0:
				cls.append(data[key])
			else:
				if data[key] not in cls:
					cls.append(data[key])
	return cls 

"""ağaç içinde yinelemeli olarak gezinir. Her yinelemede tüm sınıfların aynı olup olmadığını kontrol eder. Eğer aynı ise direk olarak bu değeri kullanır."""
def pruneTree(data):
	for key in data.keys():
		if type(data[key]).__name__ == 'dict':
			uniques = get_uniques(data[key])
			if len(uniques) != 1:
				pruneTree(data[key])
			else:
				data[key] = uniques[0]