import subprocess
import os
import sys
import time
import random
import numpy as np
import araclar.db_sql as sql
from sklearn.ensemble import ExtraTreesClassifier
from ascii_graph import Pyasciigraph
from ascii_graph.colors import *
from ascii_graph.colordata import vcolor
import inquirer
from tqdm import tqdm
import araclar.agaclar.algoritma_ID3 as SmaTree_Id3
import araclar.agaclar.algoritma_CART as SmaTree_Cart
import araclar.agaclar.agac_gorsellestir as SmaVisualize
from colorama import Fore, Back, Style, init
import ayarlar.ayarlar as settings
from numpy.core import records
from random import shuffle
from copy import copy, deepcopy
import datetime

init()
#############################################################################################
#	EKRAN TEMİZLEME
#############################################################################################
def clear_screen():
	os.system('cls' if os.name == 'nt' else 'clear')

#############################################################################################
#	MENU FONKSIYONU
#############################################################################################
def menu():

	control = "t"
	text = " "

	while True:
		if control == "t":
			text = "Seçim"
		else:
			text = "Hatalı giriş, Seçim"

		print("\nMENU")
		print("--------------------------------")
		menu = [
				inquirer.List('secim',
			                	message=text,
			                	choices=[
			                		Fore.YELLOW + '1.' + Style.RESET_ALL + ' Eğitim Kümesi Oluştur', 
			                		Fore.YELLOW + '2.' + Style.RESET_ALL + ' Eğitim Kümesi İçin Sapan Veri Analizi', 
			                		Fore.YELLOW + '3.' + Style.RESET_ALL + ' Eğitim Kümesi İçin Veri Formatı Dönüştürme', 
			                		Fore.YELLOW + '4.' + Style.RESET_ALL + ' Eğitim Kümesi İçin Öznitelik - Sınıf İlişkisi', 
			                		Fore.YELLOW + '5.' + Style.RESET_ALL + ' Eğitim Kümesi İçin Sürekli Verileri Grupla', 
			                		Fore.YELLOW + '6.' + Style.RESET_ALL + ' Test Kümesi Oluştur',
			                		Fore.YELLOW + '7.' + Style.RESET_ALL + ' Test Kümesi İçin Önişleme Ve Dönüştürme Adımları',
			                		Fore.YELLOW + '8.' + Style.RESET_ALL + ' Eğitim ve Test Kümeleri Üzerinde Algoritmaları Eğit',
			                		Fore.YELLOW + '9.' + Style.RESET_ALL + ' Test Kümesini Sınıflandır',
			                		Fore.CYAN + '10.' + Style.RESET_ALL + ' Mevcut Trafik Kayıtlarını Sınıflandır',
			                		Fore.CYAN + '11.' + Style.RESET_ALL + ' Kaydedilmiş Trafik Kayıtları Üzerinden Veri Tabanını Güncelle',
			                		Fore.CYAN + '12.' + Style.RESET_ALL + ' Canlı Trafik Kayıtları Üzerinden Veri Tabanını Güncelle & Sınıflandır',
			                		Fore.MAGENTA + '13.' + Style.RESET_ALL + ' KDD CUP 99 %10 Veri Kümesi İçin 10 Katlamalı Çapraz Doğrulama',
									Fore.MAGENTA + '14.' + Style.RESET_ALL + ' NSL-KDD Eğitim Veri Kümesi İçin 10 Katlamalı Çapraz Doğrulama',
			                		Fore.MAGENTA + '15.' + Style.RESET_ALL + ' NSL-KDD Eğitim Ve Test Veri Kümeleri Üzerinde Algoritmaların Başarımları',
									Fore.YELLOW + '16.' + Style.RESET_ALL + ' NSL-KDD Eğitim Ve Test Veri Kümeleri Üzerinde Algoritmaların Eğitimi',
									Fore.BLUE + '17.'+ Style.RESET_ALL + ' NSL-KDD Eğitim Veri Kümesi "portsweep" Saldırısı Güncellik Testi',
			                		Fore.CYAN + '18.' + Style.RESET_ALL + ' Mevcut Trafik Kayıtlarını Sınıflandır (NSL)',
			                		Fore.CYAN + '19.' + Style.RESET_ALL + ' Canlı Trafik Kayıtları Üzerinden Veri Tabanını Güncelle & Sınıflandır (NSL)',
			                		Fore.RED + '20.' + Style.RESET_ALL + ' Sonlandır'		                		
			                		],
		            		),
				]
		secim = inquirer.prompt(menu)['secim']

		clear_screen()

		if secim == Fore.YELLOW + '1.' + Style.RESET_ALL + ' Eğitim Kümesi Oluştur':
			control = "t"
			build_data(settings.configs_operations[0])
		elif secim == Fore.YELLOW + '2.' + Style.RESET_ALL + ' Eğitim Kümesi İçin Sapan Veri Analizi':
			control = "t"
			analyze_outliers()
		elif secim == Fore.YELLOW + '3.' + Style.RESET_ALL + ' Eğitim Kümesi İçin Veri Formatı Dönüştürme':
			control = "t"
			change_data_format(settings.sample_data_table_name, settings.configs_operations[0])
		elif secim == Fore.YELLOW + '4.' + Style.RESET_ALL + ' Eğitim Kümesi İçin Öznitelik - Sınıf İlişkisi':
			control = "t"
			class_feature_relation(settings.sample_data_table_name)
		elif secim == Fore.YELLOW + '5.' + Style.RESET_ALL + ' Eğitim Kümesi İçin Sürekli Verileri Grupla':
			control = "t"
			discretization(settings.sample_data_table_name, settings.configs_operations[0])
		elif secim == Fore.YELLOW + '6.' + Style.RESET_ALL + ' Test Kümesi Oluştur':
			control = "t"
			build_data(settings.configs_operations[1])
		elif secim == Fore.YELLOW + '7.' + Style.RESET_ALL + ' Test Kümesi İçin Önişleme Ve Dönüştürme Adımları':
			control = "t"
			format_test_data()
		elif secim == Fore.YELLOW + '8.' + Style.RESET_ALL + ' Eğitim ve Test Kümeleri Üzerinde Algoritmaları Eğit':
			control = "t"
			teach_algorithms()
		elif secim == Fore.YELLOW + '9.' + Style.RESET_ALL + ' Test Kümesini Sınıflandır':
			control = "t"
			classify("kdd", settings.test_data_table_name, settings.test_data_primarykey_column, settings.mining_columns, None, [])
		elif secim == Fore.CYAN + '11.' + Style.RESET_ALL + ' Kaydedilmiş Trafik Kayıtları Üzerinden Veri Tabanını Güncelle':
			control = "t"
			update_traffic_records()
		elif secim == Fore.CYAN + '10.' + Style.RESET_ALL + ' Mevcut Trafik Kayıtlarını Sınıflandır':
			control = "t"
			classify_traffic_records("kdd")
		elif secim == Fore.CYAN + '12.' + Style.RESET_ALL + ' Canlı Trafik Kayıtları Üzerinden Veri Tabanını Güncelle & Sınıflandır':
			control = "t"
			realtime_classify_traffic_records("kdd")
		elif secim == Fore.MAGENTA + '13.' + Style.RESET_ALL + ' KDD CUP 99 %10 Veri Kümesi İçin 10 Katlamalı Çapraz Doğrulama':
			control = "t"
			ten_fold(settings.kdd10_data_table_name)
		elif secim == Fore.MAGENTA + '14.' + Style.RESET_ALL + ' NSL-KDD Eğitim Veri Kümesi İçin 10 Katlamalı Çapraz Doğrulama':
			control = "t"
			ten_fold(settings.nsl_train_data_table_name)
		elif secim == Fore.MAGENTA + '15.' + Style.RESET_ALL + ' NSL-KDD Eğitim Ve Test Veri Kümeleri Üzerinde Algoritmaların Başarımları':
			control = "t"
			nsl_train("fold")
		elif secim == Fore.YELLOW + '16.' + Style.RESET_ALL + ' NSL-KDD Eğitim Ve Test Veri Kümeleri Üzerinde Algoritmaların Eğitimi':
			control = "t"
			nsl_train("nsl")
		elif secim == Fore.BLUE + '17.'+ Style.RESET_ALL + ' NSL-KDD Eğitim Veri Kümesi "portsweep" Saldırısı Güncellik Testi':
			control = "t"
			portsweep_accuracy()
		elif secim == Fore.CYAN + '18.' + Style.RESET_ALL + ' Mevcut Trafik Kayıtlarını Sınıflandır (NSL)':
			control = "t"
			classify_traffic_records("nsl")
		elif secim == Fore.CYAN + '19.' + Style.RESET_ALL + ' Canlı Trafik Kayıtları Üzerinden Veri Tabanını Güncelle & Sınıflandır (NSL)':
			control = "t"
			realtime_classify_traffic_records("nsl")
		elif secim == Fore.RED + '20.' + Style.RESET_ALL + ' Sonlandır':
			break
		else:
			control = "f"

#############################################################################################
#	BAŞLANGIÇ FONKSİYONU (main FONKSIYONU)
#############################################################################################
def initialize():
	clear_screen()

	print("\n\n-----------------------------------------------------------")
	print("Veri Ambarı ile bağlantı kuruluyor...")
	connectionStartTime = time.time()
	status, err_message = sql.connect_db()
	if status:
		print(Fore.YELLOW + Fore.YELLOW + "--> " + Style.RESET_ALL + "" + Style.RESET_ALL + "%.3f saniyede veri tabanına bağlantı sağlandı." % (time.time() - connectionStartTime))
		print("-----------------------------------------------------------\n")
		menu()
	else:
		print(Fore.YELLOW + Fore.YELLOW + "--> " + Style.RESET_ALL + "" + Style.RESET_ALL + "Bağlantı sırasında hata ile karşılaşıldı.\nHata: " + err_message)	
		print("-----------------------------------------------------------\n")
		questions = [ inquirer.Confirm('continue', message="Bağlanmayı tekrar denemek ister misiniz?") ]
		answer = inquirer.prompt(questions)['continue']
		if answer:
			initialize()
		else:
			print('Veri tabanı ile olan bağlantı sorunlarından dolayı program sonlandırıldı.')

#############################################################################################
#	AYARLARI VERİ TABANINDAN ALMA VE KAYDETME
#############################################################################################
def get_config(parameter):
	query = ("select " + settings.configs_table_value_column + " from " + settings.configs_table_name + " where " + settings.configs_table_parameter_column + " = \"{}\"".format(parameter))
	cursor = sql.query_db(query)
	if cursor.rowcount == 0:
		set_config(parameter, settings.configs_sample_status_values[0])
		return settings.configs_sample_status_values[0]
	else:
		return cursor.fetchone()[0]

def set_config(parameter, value):
	control_query = ("select count(*) from " + settings.configs_table_name + " where " + settings.configs_table_parameter_column + " = \"{}\"".format(parameter))
	control_cursor = sql.query_db(control_query)
	count = control_cursor.fetchone()[0]

	if count == 0:
		query = ("insert into " + settings.configs_table_name + " (" + settings.configs_table_parameter_column + ", " + settings.configs_table_value_column + ") values (\"{}\", \"{}\")".format(parameter, value))
		cursor = sql.query_db(query)
	else:
		query = ("update " + settings.configs_table_name + " set " + settings.configs_table_value_column + " = \"{}\" where ".format(value) + settings.configs_table_parameter_column + " = \"{}\"".format(parameter))
		cursor = sql.query_db(query)
	sql.query_commit()
#############################################################################################
#	ÖRNEKLEMDE İLGİLİ SINIF İÇİN OLACAK NESNE SAYISI
#############################################################################################
def get_rate(toplam, yeni_toplam, sayi):
	result = (sayi/toplam)*yeni_toplam
	return round(result)

#############################################################################################
#	İLGİLİ SINIF İÇİN İD BİLGİLERİ GETİRME
#############################################################################################
def get_data(class_value, operation):

	if operation == settings.configs_operations[0]:
		query = ("select " + settings.raw_data_primarykey_column + " from " + settings.raw_data_table_name + " where " + settings.raw_data_class_column + ' = "{}"'.format(class_value))
	elif operation == settings.configs_operations[1]:
		query = ("select " + settings.raw_data_primarykey_column + " from " + settings.raw_data_table_name + " where " + settings.raw_data_class_column + ' = "{}" and '.format(class_value) + settings.raw_data_primarykey_column + " not in (select " + settings.raw_data_primarykey_column + " from " + settings.sample_data_table_name + " where " + settings.raw_data_class_column + ' = "{}")'.format(class_value))
	cursor = sql.query_db(query)
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Kayıtlar getirildi. Diziye aktarılıyor...")
	#data = cursor.fetchall()
	data = []
	total = cursor.rowcount
	pbar = tqdm(total=total, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} | {elapsed} ")
	pbar.set_description("    İşleniyor")
	for record in cursor:
		data.append(record)
		pbar.update(1)
	pbar.close()

	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Kayıtlar diziye aktarıldı.",flush=True)
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Kayıt Sayısı: {}".format(len(data)))
	return data

#############################################################################################
#	RASTGELE OLARAK BELLİ ARALIKTA BELİRLİ BİR SAYIDA SAYI ÜRETME
#############################################################################################
def random_number_generate_in_range(len_data, len_):
	randomNumbers = random.sample(range(0, len_data - 1), len_)
	return randomNumbers

#############################################################################################
#	İLGİLİ SINIFIN İDLERİNDEN BELİRLİ SAYIDA RASTGELE OLARAK ID SEÇİMİ VE ÖRNEKLEM TABLOSUNA
#	KAYDEDİLMESİ
#	İLGİLİ SINIFIN İD BİLGİLERİ, BU VERİNİN UZUNLUĞU VE ÖRNEKLEM SAYISI
#############################################################################################
def copy_data(data, len_data, len_, operation):
	randomNumbers = random_number_generate_in_range(len_data, len_)
	total = len(randomNumbers)
	pbar = tqdm(total=total, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} | {elapsed} ")
	pbar.set_description("    İşleniyor")
	index = 0
	for r in randomNumbers:
		index = index + 1
		relative_id = (data[r])[0]
		
		if operation == settings.configs_operations[0]:
			query = ("insert into " + settings.sample_data_table_name + " select * from " + settings.raw_data_table_name + " where " + settings.raw_data_primarykey_column + "={}".format(relative_id))
		elif operation == settings.configs_operations[1]:
			query = ("insert into " + settings.test_data_table_name + " select *, 'None' from " + settings.raw_data_table_name + " where " + settings.raw_data_primarykey_column + "={}".format(relative_id))

		sql.query_db(query)
		pbar.update(1)
	pbar.close()

#############################################################################################
#	ÖRNEKLEM ÜRETME
#############################################################################################
def build_data(operation):
	if control_build_data(operation):
		if operation == settings.configs_operations[0]:
			tag = "örneklem"
		elif operation == settings.configs_operations[1]:
			tag = "test"

		class_datas = []
		toplam_kayit = 0
		for cls_value in settings.classes_values:
			index = settings.classes_values.index(cls_value)
			cls_name = settings.classes_names[index]

			print("\n\n-----------------------------------------------------------")
			if operation == settings.configs_operations[0]:
				print("\n" + "[ " + Fore.BLUE + str(index + 1) + Style.RESET_ALL + " | " + Fore.BLUE + str(len(settings.classes_values)) + Style.RESET_ALL + " ] " + Fore.GREEN + cls_name + Style.RESET_ALL + " veri tabanından getiriliyor...")
			elif operation == settings.configs_operations[1]:
				print("\n" + "[ " + Fore.BLUE + str(index + 1) + Style.RESET_ALL + " | " + Fore.BLUE + str(len(settings.classes_values)) + Style.RESET_ALL + " ] " + Fore.GREEN + cls_name + Style.RESET_ALL + " veri tabanından getiriliyor..." + Style.DIM + "(Örneklemde olmayan kayıtlar)" + Style.RESET_ALL)
			
			cls_data = get_data(cls_value, operation)
			class_datas.append(cls_data)
			toplam_kayit = toplam_kayit + len(cls_data)

		print("\nToplam kayıt sayısı: {}".format(toplam_kayit))

		while True:
			questions = [inquirer.Text('record_number', message="Yeni " + tag + " kümesinde olmasını istediğiniz kayıt sayısını giriniz")
						]
			while True:
				try:
					new_record_number = int(inquirer.prompt(questions)['record_number'])
					if new_record_number > 0 and new_record_number <= toplam_kayit:
						break
					else:
						print("    Lütfen sıfırdan büyük ve toplam kayıt sayısından (" + str(toplam_kayit) + ") küçük bir sayı giriniz!")
				except:
					print("    Lütfen sayı giriniz!")
			
			print("\nİstenilen veri kümesi kayıt sayısı: {}".format(new_record_number))

			class_new_lengths = []
			for cls_data in class_datas:
				new_len = get_rate(toplam_kayit, new_record_number, len(cls_data))
				print(new_len)
				class_new_lengths.append(new_len)

			print("\n-----------------------------------------------------------")

			for cls_value in settings.classes_values:
				index = settings.classes_values.index(cls_value)
				cls_name = settings.classes_names[index]
				new_len = class_new_lengths[index]
				if new_len != 0:
					print(Fore.GREEN + cls_name + Style.RESET_ALL + " için " + tag + " sayısı: " + Fore.YELLOW + "{}".format(new_len) + Style.RESET_ALL)
				else:
					print(Fore.RED + cls_name + Style.RESET_ALL + " için " + tag + " sayısı: " + Fore.RED + "{}".format(new_len) + Style.RESET_ALL)

			yeni_toplam_kayit = 0
			for new_len in class_new_lengths:
				yeni_toplam_kayit = yeni_toplam_kayit + new_len
			print("\nToplam " + tag + " sayısı {} olacaktır.".format(yeni_toplam_kayit))

			print("-----------------------------------------------------------\n")

			answers = ["Evet, yeni kayıt sayıları ile devam etmek istiyorum","Hayır, tekrar kayıt sayısı girmek istiyorum"]
			questions = [
							inquirer.List('record_option',
							message="Devam etmek istiyor musunuz?",
							choices=[Fore.YELLOW + 'Evet' + Style.RESET_ALL + ', bu kayıt sayıları ile devam et.', Fore.RED + 'Hayır' + Style.RESET_ALL + ', yeni kayıt sayısı girmek istiyorum.'],
							),
						]
			answer = inquirer.prompt(questions)["record_option"]
			if "Evet" in answer:
				break

		print(tag + " çıkarma işlemi yapılıyor...")

		if operation == settings.configs_operations[0]:
			set_config(settings.configs_sample_status_parameter, settings.configs_sample_status_values[2])
		elif operation == settings.configs_operations[1]:
			set_config(settings.configs_test_status_parameter, settings.configs_test_status_values[2])

		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Eski " + tag + " verisi temizleniyor.")

		if operation == settings.configs_operations[0]:
			query = "delete from " + settings.sample_data_table_name
		elif operation == settings.configs_operations[1]:
			query = "delete from " + settings.test_data_table_name

		sql.query_db(query)
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Eski " + tag + " verisi temizlendi.")

		if operation == settings.configs_operations[0]:
			set_config(settings.configs_sample_status_parameter, settings.configs_sample_status_values[3])
		elif operation == settings.configs_operations[1]:
			set_config(settings.configs_test_status_parameter, settings.configs_test_status_values[3])

		for cls_value in settings.classes_values:
			index = settings.classes_values.index(cls_value)
			cls_name = settings.classes_names[index]
			new_len = class_new_lengths[index]
			cls_data = class_datas[index]
			if new_len > 0:
				print(Fore.YELLOW + "--> " + Style.RESET_ALL + Fore.GREEN + cls_name + Style.RESET_ALL + " için " + tag + " verisi oluşturuluyor...")
				copy_data(cls_data, len(cls_data), new_len, operation)
				print(Fore.YELLOW + "    " + Style.RESET_ALL + tag + " verisi oluşturuldu.")
			else:
				print(Fore.YELLOW + "--> " + Style.RESET_ALL + Fore.RED + cls_name + Style.RESET_ALL + " için yeni " + tag + " verisi kayıt sayısı yeterli değil!")

		sql.query_commit()

		if operation == settings.configs_operations[0]:
			set_config(settings.configs_sample_status_parameter, settings.configs_sample_status_values[1])
			set_config(settings.configs_data_format_status_parameter, settings.configs_data_format_status_values[1])
			set_config(settings.configs_test_status_parameter, settings.configs_test_status_values[0])
			set_config(settings.configs_discretization_status_parameter, settings.configs_discretization_status_values[1])
			set_config(settings.configs_algorithm_status_parameter, settings.configs_algorithm_status_values[0])
		elif operation == settings.configs_operations[1]:
			set_config(settings.configs_test_status_parameter, settings.configs_test_status_values[1])
			set_config(settings.configs_algorithm_status_parameter, settings.configs_algorithm_status_values[0])

		print("-----------------------------------------------------------")

#############################################################################################
#	10 FOLD KDD 10 Per
#############################################################################################

def tarih():
	return "(zaman: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") +")"


def preprocess_test(data):
	d = deepcopy(data)
	d_transformed = preprocess_operation_for_fold_transform(d)
	d_preprocessed, cutoffs = preprocess_operation_for_fold_disc(d_transformed)

def nsl_train(save_operation):
	ar = []

	questions = [
		inquirer.List('expert_choice',
					  message="Uzman görüşü ile türüne karar verilen saldırılar öğrenme kümesine dahil edilsin mi?",
					  choices=[Fore.YELLOW + 'Evet' + Style.RESET_ALL,
							   Fore.RED + 'Hayır' + Style.RESET_ALL],
					  ),
	]
	answer = inquirer.prompt(questions)["expert_choice"]
	if "Evet" in answer:
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Uzman görüşü ile türüne karar verilen saldırılar getiriliyor...")
		query = ("select " + settings.mining_columns + "," + settings.expert_prediction_column_name + " from " + settings.expert_data_table_name + ' where prediction != "' + settings.unknown_class_value  + '"')
		cursor = sql.query_db(query)
		for record in cursor:
			ar.append(list(record))

	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "NSL-KDD eğitim kümesi kayıtları getiriliyor...")
	query = ("select " + settings.mining_columns + "," + settings.mining_class + " from " + settings.nsl_train_data_table_name)
	cursor = sql.query_db(query)
	for record in cursor:
		ar.append(list(record))

	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "NSL-KDD test kümesi kayıtları getiriliyor...")
	query = ("select " + settings.mining_columns + "," + settings.mining_class + " from " + settings.nsl_test_data_table_name)
	cursor = sql.query_db(query)
	artest = []
	for record in cursor:
		artest.append(list(record))

	arr = deepcopy(ar)
	arrtest = deepcopy(artest)

	preprocess_test(arr)

	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Önişlem adımları gerçekleştiriliyor...")
	train, test = preprocess_operation_for_fold(arr, arrtest)
	if save_operation == "fold":
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Başarımlar hesaplanıyor...")
	algorithms_train_test(train, test, save_operation)
	if save_operation == "fold":
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Başarımlar hesaplandı.")

def stratified_random_ten_folds(table_name):
	array_classes = []

	print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + "Her sınıf için kayıtlar getiriliyor...")
	for cls_value in settings.classes_values:
		print(Fore.YELLOW + "--> --> --> " + Style.RESET_ALL + cls_value + " sınıfı için kayıtlar getiriliyor...")
		query = ("select " + settings.mining_columns + "," + settings.mining_class + " from " + table_name + " where " + settings.mining_class + ' = "{}"'.format(cls_value))
		cursor = sql.query_db(query)
		data=[]
		for record in cursor:
			data.append(list(record))
		array_classes.append(data)

	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Tüm kayıtlar getirildi. Kümelere bölme işlemi başlatıldı.")
	folds = [[], [], [], [], [], [], [], [], [], []]

	class_index = 0
	for class_records in array_classes:
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + str(class_records[0][-1]) + " sınıfı kümelere bölünüyor...")
		shuffle(class_records)
		cur_length = len(class_records)
		new_length = int(cur_length / 10)

		fold_index = 0
		while (True):
			records = class_records[:new_length]
			class_records = class_records[new_length:]

			if fold_index == 9 and len(class_records) != 0:
				for cr in class_records:
					records.append(cr)
				class_records = []

			for r in records:
				folds[fold_index].append(r)
			fold_index = fold_index + 1
			if fold_index == 10:
				break
		array_classes[class_index] = []
		class_index = class_index + 1
	return folds


def algorithms_train_test(train, test, save_operation):
	columns_array = deepcopy(settings.mining_columns.split(","))

	trn_cart = deepcopy(train)
	te_cart = deepcopy(test)
	columns_array_cart = deepcopy(columns_array)
	trn_id3 = deepcopy(train)
	te_id3 = deepcopy(test)
	columns_array_id3 = deepcopy(columns_array)

	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "CART algoritması öğreniyor...")
	cart_agac, cart_dogruluk, cart_accuracy_matrix, cart_learning_time, cart_testing_time = SmaTree_Cart.Tree(trn_cart, te_cart, columns_array_cart, save_operation)
	if save_operation != "nsl":
		print(cart_dogruluk)
		print(cart_accuracy_matrix)
	if save_operation == "nsl":
		print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + "Ağaç görselleştiriliyor...")
		SmaVisualize.VisualizeSmaTree(cart_agac, "nsl_cart")
		print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + "Ağaç görselleştirildi...")
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "ID3 algoritması öğreniyor...")
	id3_agac, id3_dogruluk, id3_accuracy_matrix, id3_learning_time, id3_testing_time = SmaTree_Id3.Tree(trn_id3, te_id3, columns_array_id3, save_operation)
	if save_operation != "nsl":
		print(id3_dogruluk)
		print(id3_accuracy_matrix)
	if save_operation == "nsl":
		print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + "Ağaç görselleştiriliyor...")
		SmaVisualize.VisualizeSmaTree(id3_agac, "nsl_id3")
		print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + "Ağaç görselleştirildi...")
		set_config(settings.configs_nsl_algorithm_status_parameter, settings.configs_nsl_algorithm_status_values[1])

		id3_derinlik = SmaTree_Id3.getTreeDepth(id3_agac)
		id3_yaprak_sayisi = SmaTree_Id3.getNumLeafs(id3_agac)
		cart_derinlik = SmaTree_Cart.getTreeDepth(cart_agac)
		cart_yaprak_sayisi = SmaTree_Cart.getNumLeafs(cart_agac)
		id3 = [id3_dogruluk, id3_accuracy_matrix, id3_learning_time, id3_testing_time, id3_derinlik, id3_yaprak_sayisi]
		cart = [cart_dogruluk, cart_accuracy_matrix, cart_learning_time, cart_testing_time, cart_derinlik, cart_yaprak_sayisi]

		print(Fore.YELLOW + "\n\t--NSL-----------------------------------------------" + Style.RESET_ALL)
		print(Fore.BLUE + "\tÖzellik\t\t\tID3 (Ent.)\tCART (Gini)" + Style.RESET_ALL)
		print(Fore.YELLOW + "\t----------------------------------------------------" + Style.RESET_ALL)
		print("\tÖğrenme Süresi" + "\t\t" + "%.3f saniye" % id3[2] + "\t" + "%.3f saniye" % cart[2])
		print(Fore.YELLOW + "\t----------------------------------------------------" + Style.RESET_ALL)
		print("\tSınıflandırma Süresi" + "\t" + "%.3f saniye" % id3[3] + "\t" + "%.3f saniye" % cart[3])
		print(Fore.YELLOW + "\t----------------------------------------------------" + Style.RESET_ALL)
		print("\tYaprak Sayısı" + "\t\t" + str(id3[5]) + "\t\t" + str(cart[5]))
		print(Fore.YELLOW + "\t----------------------------------------------------" + Style.RESET_ALL)
		print("\tDerinlik" + "\t\t" + str(id3[4]) + "\t\t" + str(cart[4]))
		print(Fore.YELLOW + "\t----------------------------------------------------" + Style.RESET_ALL)
		print("\tDoğruluk (Accuracy)" + "\t" + "%.4f" % id3[0] + "\t\t" + "%.4f" % cart[0])
		print(Fore.YELLOW + "\t----------------------------------------------------" + Style.RESET_ALL)

		print("\tID3 doğruluk matrisi\n")
		print(id3[1])
		print("\t----------------------------------------------------")
		print("\tCART doğruluk matrisi\n")
		print(cart[1])
		print("\t----------------------------------------------------")
		print("\tPrecision = Tp / Tp + Fp\n\tRecall = Tp / Tp + Fn\n\tF1 = 2 * (Precision * Recall) / Precision + Recall")
		print("\t----------------------------------------------------")


	return [[id3_dogruluk, id3_accuracy_matrix], [cart_dogruluk, cart_accuracy_matrix]]


def preprocess_operation_for_fold_transform(data):
	columns = deepcopy(settings.columns_data_transformation)
	column_list = columns.split(',')

	index_line = 0
	for line in data:
		index_column = 0
		for column in column_list:
			data[index_line][index_column + 1] = settings.transform_to_dummy_data_process(column_list[index_column], data[index_line][index_column + 1], line)
			index_column = index_column + 1
		index_line = index_line + 1
	return data


def preprocess_operation_for_fold_disc(data_raw, cutoffs={}):
	columns = deepcopy(settings.discretization_columns)
	column_names = columns.split(',')
	mining_colums = deepcopy(settings.mining_columns.split(','))

	data = np.array(data_raw, dtype=object)

	i = 0
	ind = 0
	lim = len(column_names)
	while True:
		ind = mining_colums.index(column_names[i])
		trf = np.transpose(data[:, ind])
		trf = trf.astype(float)
		try:
			cutoff = cutoffs[column_names[i]]
		except:
			split = np.array_split(np.sort(trf), settings.discretization_max_split)
			cutoff = [x[-1] for x in split]
			cutoff = cutoff[:-1]
			cutoff = np.unique(cutoff)
			cutoffs[column_names[i]] = cutoff

		discrete = np.digitize(trf, cutoff, right=True)
		data[:, ind] = np.transpose(discrete)

		i = i + 1
		if i == lim:
			break
	data = data.tolist()
	return data, cutoffs


def preprocess_operation_for_fold(train, test):
	train_transformed = preprocess_operation_for_fold_transform(train)
	test_transformed = preprocess_operation_for_fold_transform(test)

	train_preprocessed, cutoffs = preprocess_operation_for_fold_disc(train_transformed)
	test_preprocessed, cutoffs = preprocess_operation_for_fold_disc(test_transformed, cutoffs)

	return train_preprocessed, test_preprocessed


def algorithm_results_on_ten_folds(table_name):
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Kümeler oluşturuluyor...")
	folds = stratified_random_ten_folds(table_name)
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "10 adet küme oluşturuldu.")
	result_list = []

	fold_index = 0
	for f in folds:
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + str(
			fold_index + 1) + ". küme test için seçildi. Diğer kümeler eğitim için kullanılacak.")
		test = deepcopy(f)

		trn = []
		for i in range(len(folds)):
			if i != fold_index:
				trn = trn + folds[i]
		train = deepcopy(trn)

		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Önişlem adımları gerçekleştiriliyor...")
		preprocess_test(train)
		train_preprocessed, test_preprocessed = preprocess_operation_for_fold(train, test)
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Başarımlar hesaplanıyor...")
		result = algorithms_train_test(train_preprocessed, test_preprocessed, "fold")
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Başarımlar hesaplandı.")
		result_list.append(result)

		fold_index = fold_index + 1

	return result_list

def ten_fold(table_name):
	results = algorithm_results_on_ten_folds(table_name)

	total_id3 = 0
	total_cart = 0
	for r in results:
		total_id3 = total_id3 + r[0][0]
		total_cart = total_cart + r[1][0]

	print("#################################################")
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "ID3 algoritması doğruluk ortalaması: " + str(
		round(total_id3 / 10, 4)))
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "CART algoritması doğruluk ortalaması: " + str(
		round(total_cart / 10, 4)) + " ...")
	print("#################################################")

#############################################################################################
#	KONTROL FONKSIYONLARI
#############################################################################################
def control_analyze_outliers():
	sample_status = get_config(settings.configs_sample_status_parameter)
	if sample_status == settings.configs_sample_status_values[0] or sample_status == settings.configs_sample_status_values[3]:
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Lütfen bu adımdan önce örneklem oluşturunuz!")
		return False
	elif sample_status == settings.configs_sample_status_values[2]:
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Örneklem üzerinde değişiklik yapılıyor. Lütfen daha sonra tekrar deneyiniz!")
		return False

	data_format = get_config(settings.configs_data_format_status_parameter)
	if data_format == settings.configs_data_format_status_values[2]:
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Veri formatı değiştirilmiş!")
		print(Fore.RED + "--> " + Style.RESET_ALL + "Sapan veri sonuçları tam doğrulukta olmayabilir!")

	data_discretization = get_config(settings.configs_discretization_status_parameter)
	if data_discretization == settings.configs_discretization_status_values[2]:
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Sürekli veriler gruplanmış!")
		print(Fore.RED + "--> " + Style.RESET_ALL + "Sapan veri sonuçları tam doğrulukta olmayabilir!")

	return True

def control_data_transform(operation):
	if operation == settings.configs_operations[0]:
		sample_status = get_config(settings.configs_sample_status_parameter)
		if sample_status == settings.configs_sample_status_values[0] or sample_status == settings.configs_sample_status_values[3]:
			print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Lütfen bu adımdan önce örneklem oluşturunuz!")
			return False
		elif sample_status == settings.configs_sample_status_values[2]:
			print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Örneklem üzerinde değişiklik yapılıyor. Lütfen daha sonra tekrar deneyiniz!")
			return False

		data_format = get_config(settings.configs_data_format_status_parameter)
		if data_format == settings.configs_data_format_status_values[2]:
			print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Veri formatı zaten değiştirilmiş durumda!")
			return False

	return True

def control_build_data(operation):
	if operation == settings.configs_operations[0]:
		sample_status = get_config(settings.configs_sample_status_parameter)
		if sample_status == settings.configs_sample_status_values[0] or sample_status == settings.configs_sample_status_values[3]:
			return True

		elif sample_status == settings.configs_sample_status_values[2]:
			print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Örneklem üzerinde değişiklik yapılıyor. Lütfen daha sonra tekrar deneyiniz!")
			return False

		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Hali hazırda bir örneklem verisi bulundu!")
		questions = [ inquirer.Confirm('continue', message=" Devam etmek istiyor musunuz?") ]
		return inquirer.prompt(questions)['continue']
	elif operation == settings.configs_operations[1]:
		test_status = get_config(settings.configs_test_status_parameter)
		if test_status == settings.configs_test_status_values[0] or test_status == settings.configs_test_status_values[3]:
			return True

		elif test_status == settings.configs_test_status_values[2]:
			print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Test üzerinde değişiklik yapılıyor. Lütfen daha sonra tekrar deneyiniz!")
			return False

		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Hali hazırda bir test verisi bulundu!")
		questions = [ inquirer.Confirm('continue', message=" Devam etmek istiyor musunuz?") ]
		return inquirer.prompt(questions)['continue']

def control_class_feature_relation():
	sample_status = get_config(settings.configs_sample_status_parameter)
	if sample_status == settings.configs_sample_status_values[0] or sample_status == settings.configs_sample_status_values[3]:
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Lütfen bu adımdan önce örneklem oluşturunuz!")
		return False
	elif sample_status == settings.configs_sample_status_values[2]:
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Örneklem üzerinde değişiklik yapılıyor. Lütfen daha sonra tekrar deneyiniz!")
		return False

	data_format = get_config(settings.configs_data_format_status_parameter)
	if data_format != settings.configs_data_format_status_values[2]:
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Veri formatı henüz değiştirilmemiş!")
		print(Fore.RED + "--> " + Style.RESET_ALL + "Veri sonuçlarında doğruluk için lütfen veri formatını değiştiriniz!")
		return False

	data_discretization = get_config(settings.configs_discretization_status_parameter)
	if data_discretization == settings.configs_discretization_status_values[2]:
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Sürekli veriler gruplanmış!")
		print(Fore.RED + "--> " + Style.RESET_ALL + "Veri sonuçları tam doğrulukta olmayabilir!")

	return True

def control_discretization(operation):
	if operation == settings.configs_operations[0]:
		sample_status = get_config(settings.configs_sample_status_parameter)
		if sample_status == settings.configs_sample_status_values[0] or sample_status == settings.configs_sample_status_values[3]:
			print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Lütfen bu adımdan önce örneklem oluşturunuz!")
			return False
		elif sample_status == settings.configs_sample_status_values[2]:
			print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Örneklem üzerinde değişiklik yapılıyor. Lütfen daha sonra tekrar deneyiniz!")
			return False

		data_discretization = get_config(settings.configs_discretization_status_parameter)
		if data_discretization == settings.configs_discretization_status_values[2]:
			print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Sürekli veriler zaten gruplanmış durumda!")
			return False

	return True

def control_format_test_data():
	test_status = get_config(settings.configs_test_status_parameter)
	if test_status == settings.configs_test_status_values[0] or test_status == settings.configs_test_status_values[3]:
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Lütfen daha önce test verisi oluşturunuz!")
		return False

	elif test_status == settings.configs_test_status_values[2]:
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Test üzerinde değişiklik yapılıyor. Lütfen daha sonra tekrar deneyiniz!")
		return False

	elif test_status == settings.configs_test_status_values[4]:
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Test verisi için önişleme zaten yapılmış!")
		return False

	return True

def control_id3():
	algorithm_status = get_config(settings.configs_algorithm_status_parameter)
	if algorithm_status == settings.configs_algorithm_status_values[0]:
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Seçilen algoritma henüz öğrenilmemiştir.")
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Lütfen ilgili algoritma üzerinde öğrenme gerçekleştiriniz!")
		return False

	elif algorithm_status == settings.configs_algorithm_status_values[1] or algorithm_status == settings.configs_algorithm_status_values[3]:
		return True

def control_cart():
	algorithm_status = get_config(settings.configs_algorithm_status_parameter)
	if algorithm_status == settings.configs_algorithm_status_values[0]:
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Seçilen algoritma henüz öğrenilmemiştir.")
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Lütfen ilgili algoritma üzerinde öğrenme gerçekleştiriniz!")
		return False
	elif algorithm_status == settings.configs_algorithm_status_values[2] or algorithm_status == settings.configs_algorithm_status_values[3]:
		return True

def control_nsl():
	algorithm_status = get_config(settings.configs_nsl_algorithm_status_parameter)
	if algorithm_status == settings.configs_nsl_algorithm_status_values[0]:
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Seçilen algoritma henüz öğrenilmemiştir.")
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Lütfen ilgili algoritma üzerinde öğrenme gerçekleştiriniz!")
		return False

	elif algorithm_status == settings.configs_nsl_algorithm_status_values[1]:
		return True

def control_teach_algorithms():
	test_status = get_config(settings.configs_test_status_parameter)
	if test_status == settings.configs_test_status_values[0] or test_status == settings.configs_test_status_values[3]:
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Lütfen daha önce test verisi oluşturunuz!")
		return False

	elif test_status == settings.configs_test_status_values[2]:
		print("\n" + Fore.RED + "--> " + Style.RESET_ALL + "Test üzerinde değişiklik yapılıyor. Lütfen daha sonra tekrar deneyiniz!")
		return False

	return True
#############################################################################################
#	VERİLEN ÖZELLİKLER İÇİN IQR İLE SAPAN VERİLERİN LİSTELENMESİ
#############################################################################################
def outliers_show(columns):
	for column in columns:
		print("\n" + Fore.CYAN + column  + Style.RESET_ALL + " bilgileri veri tabanından getiriliyor...")
		query = ("select " + column + " from " + settings.sample_data_table_name)
		cursor = sql.query_db(query)
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Kayıtlar getirildi. Hesaplama yapılıyor...")
		data_array = []
		for (c) in cursor:
			data_array.append(float(c[0]))
		quartile_1, quartile_3 = np.percentile(data_array, [25, 75])
		iqr = quartile_3 - quartile_1
		lower_bound = quartile_1 - (iqr * 1.5)
		upper_bound = quartile_3 + (iqr * 1.5)
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Q3: " + str(quartile_3))
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Q1: " + str(quartile_1))
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "IQR: " + str(iqr))
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Alt sınır: " + str(lower_bound))
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Üst sınır: " + str(upper_bound))
		query = ("select " + settings.sample_data_primarykey_column + " from " + settings.sample_data_table_name + " where " + column +" < " + str(lower_bound) + " and " + column + " > " + str(upper_bound))
		cursor = sql.query_db(query)
		count = cursor.rowcount

		result = Fore.YELLOW + "--> " + Style.RESET_ALL + "Sapan veri adedi: "
		if count == 0:
			result = result + Fore.GREEN
		else:
			result = result + Fore.RED
		result = result + str(count) + Style.RESET_ALL
		print(result)

		if count != 0:
			query = ("delete from " + settings.sample_data_table_name + " where " + column +" < " + str(lower_bound) + " and " + column + " > " + str(upper_bound))
			cursor = sql.query_db(query)
			print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Sapan veriler temizlendi!")
			sql.query_commit()


#############################################################################################
#	SAPAN VERİ KONTROLÜ
#############################################################################################
def analyze_outliers():
	if control_analyze_outliers():
		print("\n\n-----------------------------------------------------------")
		print("İncelenecek Sütunlar: ")
		
		for column in settings.outliers_columns:
			print(Fore.YELLOW + "--> " + Style.RESET_ALL + "" + column)
		print("-----------------------------------------------------------")

		questions = [ inquirer.Confirm('continue', message="Devam etmek istiyor musunuz?") ]
		answer = inquirer.prompt(questions)['continue']

		if answer:
			outliers_show(settings.outliers_columns)

#############################################################################################
#	VERİLEN DATA İÇİNDE BULUNAN İLGİLİ ÖZELLİKLER İÇİN ENCODE YAPILMASI
#############################################################################################
def transform_to_dummy_data(data_np, column_list, table_name, output = True):
	index_line = 0
	for line in data_np:
		index_column = 0
		for column in column_list:
			if index_column != 0:
				data_np[index_line][index_column] = settings.transform_to_dummy_data_process(column_list[index_column],data_np[index_line][index_column],line)
			index_column = index_column + 1
		index_line = index_line + 1

	update_records(data_np, column_list, table_name, output)

def change_data_format(table_name, operation, where_clause=None):
	if control_data_transform(operation):
		print("\nKayıtlar getiriliyor...")
		columns = settings.sample_data_primarykey_column + "," + settings.columns_data_transformation
		column_list = columns.split(',')
		if where_clause is not None:
			query = ("select " + columns + " from " + table_name + " " + where_clause + " order by " + settings.sample_data_primarykey_column)
		else:
			query = ("select " + columns + " from " + table_name + " order by " + settings.sample_data_primarykey_column)
		cursor_d = sql.query_db(query)
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Kayıtlar getirildi. Diziye aktarılıyor...")

		data_array = []

		for c in cursor_d:
			data_array.append(list(c))

		if len(data_array) != 0:
			data_np = np.array(data_array)
		
			print("\nVeri formatı değiştiriliyor.")
			print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Formatı değiştirilecek özellikler: ")

			for column in column_list:
				if column != settings.sample_data_primarykey_column:
					print("    " + Fore.YELLOW + "--> " + Style.RESET_ALL + column)
			answer = True
			if operation == settings.configs_operations[0]:
				questions = [ inquirer.Confirm('continue', message="Devam etmek istiyor musunuz?") ]
				answer = inquirer.prompt(questions)['continue']

			if answer:
				set_config(settings.configs_sample_status_parameter, settings.configs_sample_status_values[2])

				if (where_clause == None):
					output = True
				else:
					output = False

				transform_to_dummy_data(data_np, column_list, table_name, output)
				set_config(settings.configs_sample_status_parameter, settings.configs_sample_status_values[4])
				set_config(settings.configs_data_format_status_parameter, settings.configs_data_format_status_values[2])
				print("\n" + Fore.YELLOW + "--> " + Style.RESET_ALL + "İlgili adım tamamlandı.")
		else:
			print(Fore.RED + "--> " + Style.RESET_ALL + "Veri tabanında analiz edilecek kayıt bulunamadı!")
			return False

#############################################################################################
#	EKRANA BAR CHART ÇİZDİRME
#############################################################################################
def print_chart(importances,column_names,header):
	array_plt = []
	index = 0
	for importance in importances:
		array_plt.append((column_names[index], importance))
		index = index + 1
	graph = Pyasciigraph(float_format='{0:,.10f}')
	pattern = [Gre, Blu]
	data = vcolor(array_plt, pattern)
	for line in graph.graph(header, data):
		print(line)

#############################################################################################
#	KUVVETLER İÇİN IQR HESAPLAMA
#############################################################################################
def class_feature_relation_iqe_print(importances,clmnNames):
		print("\n" + Fore.YELLOW + "--> " + Style.RESET_ALL + "Kuvvetler için IQR Hesaplaması:")
		quartile_1, quartile_3 = np.percentile(importances, [25, 75])
		iqr = quartile_3 - quartile_1
		lower_bound = quartile_1 - (iqr * 1.5)
		print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + "Q3: " + str(quartile_3))
		print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + "Q1: " + str(quartile_1))
		print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + "IQR: " + str(iqr))
		print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + "Alt sınır: " + str(lower_bound))

		print("\n" + Fore.YELLOW + "--> " + Style.RESET_ALL + "Alt sınırdan küçük olan özellikler:")
		ind = 0
		lim = len(clmnNames)
		while True:
			if importances[ind] < lower_bound:
				print(Fore.RED + "--> --> " + Style.RESET_ALL + str(clmnNames[ind]) + ": " + str(importances[ind]))
			ind = ind + 1
			if ind == lim:
				break

		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Alt sınırdan küçük olmayan özellikler:")
		ind = 0
		while True:
			if importances[ind] >= lower_bound:
				print(Fore.GREEN + "--> --> " + Style.RESET_ALL + str(clmnNames[ind]) + ": " + str(importances[ind]))
			ind = ind + 1
			if ind == lim:
				break

#############################################################################################
#	VERİ VE SINIFLARININ ALINARAK ÖZELLİKLER ARASI İLİŞKİ HESAPLAMA
#############################################################################################
def class_feature_relation_get_importances(data_np, data_class):
	clf = ExtraTreesClassifier()
	clf = clf.fit(data_np, data_class)
	return clf.feature_importances_

#############################################################################################
#	ÖZELLİKLERİN İLİŞKİLERİ İÇİN ANALİZ YAPILMASI
#############################################################################################
def class_feature_relation(table):

	if control_class_feature_relation():
		print("\nKayıtlar getiriliyor...")
		query = ("select " + settings.class_feature_relation_columns + "," + settings.raw_data_class_column + " from " + table + " order by " + settings.sample_data_primarykey_column)
		#clmns = "duration,service,source_byte,destination_byte,count,same_srv_rate,serror_rate,srv_serror_rate,dst_host_count,dst_host_srv_count,dst_host_same_src_port_rate,dst_host_serror_rate,dst_host_srv_serror_rate,flag,source_port_number,destination_port_number,service_type,ids_detection,malware_detection,ashula_detection"
		cursor_d = sql.query_db(query)
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Kayıtlar getirildi. Hesaplama yapılıyor...")

		data_array = []
		data_cl = []

		for c in cursor_d:
			data_array.append(list(c[:-1 or None]))
			data_cl.append(c[-1])

		data_np = np.array(data_array)

		data_class = np.array(data_cl)
		targets = np.unique(data_class)
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Bulunan Sınıflar")
		for trg in targets:
			print(Fore.YELLOW + "--> --> " + Style.RESET_ALL +"Sınıf: " + trg)

		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "İlişkiler hesaplanıyor.(extremely randomized trees)")
		importances = class_feature_relation_get_importances(data_np,data_class)
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "İlişkiler hesaplandı.\n")

		column_names = settings.class_feature_relation_columns.split(",")

		max_importance = max(importances)
		min_importance = min(importances)

		index = 0
		for importance in importances:
			if importance == min_importance:
				column_names[index] = column_names[index] + Fore.RED + " --> min" + Style.RESET_ALL
			elif importance == max_importance:
				column_names[index] = column_names[index] + Fore.YELLOW + " --> max" + Style.RESET_ALL
			index = index + 1

		print_chart(importances, column_names, Fore.YELLOW + "--> " + Style.RESET_ALL +  'Sınıf özelliği için ilişki kuvvetleri:')

		questions = [ inquirer.Confirm('continue', message="Kuvvetler için IQR hesaplansın mı?") ]
		answer = inquirer.prompt(questions)['continue']
		if answer:
			class_feature_relation_iqe_print(importances, column_names)

#############################################################################################
#	TABLODA İLGİLİ SÜTUNLARI İLGİLİ DATA İLE GÜNCELLE
#	İD İLK İNDEXTE
#############################################################################################
def update_records(data, column_list, table_name, output = True):

	total = len(data)
	if (output == True):
		pbar = tqdm(total=total, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} | {elapsed} ")
		pbar.set_description(" İşleniyor")

		for d in data:
			index_column = 0
			lim = len(column_list)
			query1 = "UPDATE " + table_name + " SET "
			while True:
				if index_column != 0:
					query1 = query1 + column_list[index_column] + ' = "' + str(d[index_column]) + '",'
				index_column = index_column + 1
				if index_column == lim:
					break

			query1 = (query1[:-1] + " WHERE " + settings.raw_data_primarykey_column + " = " + str(int(d[0])))
			sql.query_db(query1)
			pbar.update(1)
		pbar.close()
		
	else:
		for d in data:
			index_column = 0
			lim = len(column_list)
			query1 = "UPDATE " + table_name + " SET "
			while True:
				if index_column != 0:
					query1 = query1 + column_list[index_column] + ' = "' + str(d[index_column]) + '",'
				index_column = index_column + 1
				if index_column == lim:
					break

			query1 = (query1[:-1] + " WHERE " + settings.raw_data_primarykey_column + " = " + str(int(d[0])))
			sql.query_db(query1)

	sql.query_commit()

#############################################################################################
#	SÜREKLİ VERİYİ AYRIK HALE GETİRME
#############################################################################################
def discretization(table, operation, where_clause=None):
	if control_discretization(operation):
		print("\n\n-----------------------------------------------------------")
		print("\nKategorik veriye dönüştürülecek özellikler: ")
		columns = settings.sample_data_primarykey_column + "," + settings.discretization_columns
		
		column_names = settings.discretization_columns.split(",")
		for c in column_names:
			print(Fore.YELLOW + "--> " + Style.RESET_ALL + c)
		print("-----------------------------------------------------------")
		answer = True
		if operation == settings.configs_operations[0]:
			questions = [ inquirer.Confirm('continue', message="Devam etmek istiyor musunuz?") ]
			answer = inquirer.prompt(questions)['continue']
		if answer:
			print("\nKayıtlar getiriliyor...")
			if where_clause is not None:
				query = ("select " + columns + " from " + table + " " + where_clause + " order by " + settings.sample_data_primarykey_column)
			else:
				query = ("select " + columns + " from " + table + " order by " + settings.sample_data_primarykey_column)
			cursor_d = sql.query_db(query)

			data_array = []

			for c in cursor_d:
				data_array.append(list(c))

			if len(data_array) != 0:
				data_np = np.array(data_array)
				print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Kayıtlar getirildi. Veri gruplandırılıyor...")

				if table == settings.sample_data_table_name:
					del_cuttoff()

				column_names = columns.split(",")

				ind = 0
				lim = len(column_names)
				while True:
					if ind != 0:
						trf = np.transpose(data_np[:,ind])

						if table == settings.sample_data_table_name:
							split = np.array_split(np.sort(trf), settings.discretization_max_split)
							cutoffs = [x[-1] for x in split]
							cutoffs = cutoffs[:-1]
							cutoffs = np.unique(cutoffs)

							set_cuttoff(column_names[ind], cutoffs)
						else:
							cutoffs = get_cuttoff(column_names[ind])

						discrete = np.digitize(trf, cutoffs, right=True)
						print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + column_names[ind] + ": " + str(cutoffs) + " ile gruplandı.")

						data_np[:,ind] = np.transpose(discrete)

					ind = ind + 1
					if ind == lim:
						break
				print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Veri gruplandı. Veri tabanı güncelleniyor...")

				if (where_clause == None):
					update_records(data_np, column_names, table)
				else:
					update_records(data_np, column_names, table, False)

				set_config(settings.configs_discretization_status_parameter, settings.configs_discretization_status_values[2])
				print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Veri tabanı güncellendi.")
			else:
				print(Fore.RED + "--> " + Style.RESET_ALL + "Veri tabanında analiz edilecek kayıt bulunamadı!")
				return False

def get_cuttoff(cat):
	query = ("select " + settings.discretization_table_value_column + " from " + settings.discretization_table_name + " where " + settings.discretization_table_parameter_column + "='" + cat + "'")
	cursor = sql.query_db(query)
	for item in cursor:
		arr_info = item[0]

	return np.loads(arr_info)

def set_cuttoff(cat,np_arr):
	dmp = np_arr.dumps()

	query = ("insert into " + settings.discretization_table_name + " (" + settings.discretization_table_parameter_column + "," + settings.discretization_table_value_column + ") values (%s, %s )")
	sql.query_db_with_blob(query, cat, dmp)
	sql.query_commit()

def del_cuttoff():
	query = ("delete from " + settings.discretization_table_name)
	sql.query_db(query)
	sql.query_commit()

#############################################################################################
#	TEST VERİSİ OLUŞTURMA
#############################################################################################

def format_test_data():
	if control_format_test_data():
		change_data_format(settings.test_data_table_name, settings.configs_operations[1])
		discretization(settings.test_data_table_name, settings.configs_operations[1])
		set_config(settings.configs_test_status_parameter, settings.configs_test_status_values[4])


#############################################################################################
#	ALGORİTMALAR
#############################################################################################

def algorithm_ID3(columns_array,data_array, data_array_test):
	columns_array = columns_array[:-1]

	agac, dogruluk, accuracyMatrix, learningTime, testingTime = SmaTree_Id3.Tree(data_array, data_array_test, columns_array)
	derinlik = SmaTree_Id3.getTreeDepth(agac)
	yaprakSayisi = SmaTree_Id3.getNumLeafs(agac)
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "ID3 algoritması öğrenmeyi tamamladı.")
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Ağaç dosyaya kaydedildi.")
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Doğruluk: " + str(dogruluk))
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Öğrenme Süresi: %.3f saniye" % learningTime)
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Sınıflandırma Süresi: %.3f saniye" % testingTime)
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Yaprak Sayısı: " + str(yaprakSayisi))
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Derinlik: " + str(derinlik))

	delete_tree_file("id3")
	questions = [ inquirer.Confirm('continue', message="ID3 ile oluşturulmuş ağaç ekrana basılsın mı?") ]
	answer = inquirer.prompt(questions)['continue']

	if answer:
		print("\n--> Ağaç: ")
		print(agac)

	questions = [ inquirer.Confirm('continue', message="Kaydedilmiş ağaç görselleştirilsin mi?") ]
	answer = inquirer.prompt(questions)['continue']

	if answer:
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Kaydedilmiş ağaç görselleştiriliyor...")
		SmaVisualize.VisualizeSmaTree(agac,"id3")
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Ağaç görselleştirildi ve kaydedildi.")

	return [dogruluk, accuracyMatrix, learningTime, testingTime, derinlik, yaprakSayisi]

def algorithm_CART(columns_array, data_array, data_array_test):
	columns_array = columns_array[:-1]

	agac, dogruluk, accuracyMatrix, learningTime, testingTime = SmaTree_Cart.Tree(data_array, data_array_test, columns_array)
	derinlik = SmaTree_Cart.getTreeDepth(agac)
	yaprakSayisi = SmaTree_Cart.getNumLeafs(agac)
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "CART algoritması öğrenmeyi tamamladı.")
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Ağaç dosyaya kaydedildi.")
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Doğruluk: " + str(dogruluk))
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Öğrenme Süresi: %.3f saniye" % learningTime)
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Sınıflandırma Süresi: %.3f saniye" % testingTime)
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Yaprak Sayısı: " + str(yaprakSayisi))
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Derinlik: " + str(derinlik))

	delete_tree_file("cart")
	questions = [ inquirer.Confirm('continue', message="CART ile oluşturulmuş ağaç ekrana basılsın mı?") ]
	answer = inquirer.prompt(questions)['continue']

	if answer:
		print("\n--> Ağaç: ")
		print(agac)

	questions = [ inquirer.Confirm('continue', message="Kaydedilmiş ağaç görselleştirilsin mi?") ]
	answer = inquirer.prompt(questions)['continue']

	if answer:
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Kaydedilmiş ağaç görselleştiriliyor...")
		SmaVisualize.VisualizeSmaTree(agac,"cart")
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Ağaç görselleştirildi ve kaydedildi.")

	return [dogruluk, accuracyMatrix, learningTime, testingTime, derinlik, yaprakSayisi]

def teach_algorithms():
	if control_teach_algorithms():
		print("\n\n-----------------------------------------------------------")
		columns = settings.mining_columns.split(",")
		questions = [
						inquirer.List('column_option',
						message="Veri madenciliği işlemlerinde kullanılacak özellikleri belirlemek ister misiniz?",
						choices=[Fore.YELLOW + 'Evet' + Style.RESET_ALL + ', özellik seç.', Fore.RED + 'Hayır' + Style.RESET_ALL + ', tümünü kullan.', Fore.BLUE + "İptal" + Style.RESET_ALL + ", ana menüye dön!"],
						),
					]
		answers = inquirer.prompt(questions)["column_option"]

		if "İptal" in answers:
			clear_screen()
			return

		if "Evet" in answers:
			questions = [
							inquirer.Checkbox('columns',
	                		message="Lütfen özellik seçiniz. Hiçbir özellik seçilmezse işlem iptal edilecektir",
	                		choices=settings.mining_columns.split(","),
	                		),
						]
			columns = inquirer.prompt(questions)["columns"]
		columns = ",".join(str(x) for x in columns)
		if len(columns) != 0:
			columns = columns + "," + settings.mining_class
			questions = [
						inquirer.List('operation_choise',
						message="Hangi algoritmaları öğrenme kümesi üzerinde eğitmek istiyorsunuz?",
						choices=[Style.RESET_ALL + "Her" + Fore.YELLOW + ' iki ' + Style.RESET_ALL + 'algoritma', Fore.RED + 'Yalnızca' + Style.RESET_ALL + ' ID3 algoritması', Fore.RED + 'Yalnızca' + Style.RESET_ALL + ' CART algoritması'],
						),
					]
			answers = inquirer.prompt(questions)["operation_choise"]

			op_count = 1
			op_index = 1
			if "ID3" in answers:
				op = "id3"
				op_count = 2
			elif "CART" in answers:
				op = "cart"
				op_count = 2
			else:
				op = "all"
				op_count = 3

			print("\n" + "[ " + Fore.BLUE + str(op_index) + Style.RESET_ALL + " | " + Fore.BLUE + str(op_count) + Style.RESET_ALL + " ] " + "Kayıtlar getiriliyor...")

			query = ("select " + columns + " from " + settings.sample_data_table_name + " order by " + settings.sample_data_primarykey_column)
			cursor_d = sql.query_db(query)

			data_array = []
			for c in cursor_d:
				data_array.append(list(c))

			query_test = ("select " + columns + " from " + settings.test_data_table_name + " order by " + settings.test_data_primarykey_column)
			cursor_t = sql.query_db(query_test)

			data_array_test = []
			for c_t in cursor_t:
				data_array_test.append(list(c_t))

			print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Kayıtlar getirildi.\n")

			columns_array = columns.split(",")

			if op == "id3" or op == "all":
				op_index = op_index + 1
				print("\n" + "[ " + Fore.BLUE + str(op_index) + Style.RESET_ALL + " | " + Fore.BLUE + str(op_count) + Style.RESET_ALL + " ] " + Fore.GREEN + "ID3 algoritması" + Style.RESET_ALL + " örneklem üzerinde öğreniyor...")
				id3 = algorithm_ID3(columns_array,data_array, data_array_test)
				if op != "all":
					set_config(settings.configs_algorithm_status_parameter, settings.configs_algorithm_status_values[1])

			if op == "cart" or op == "all":
				op_index = op_index + 1
				print("\n" + "[ " + Fore.BLUE + str(op_index) + Style.RESET_ALL + " | " + Fore.BLUE + str(op_count) + Style.RESET_ALL + " ] " + Fore.GREEN + "CART algoritması" + Style.RESET_ALL + " örneklem üzerinde öğreniyor...")
				cart = algorithm_CART(columns_array, data_array, data_array_test)
				if op != "all":
					set_config(settings.configs_algorithm_status_parameter, settings.configs_algorithm_status_values[2])

			if op == "all":
				questions = [ inquirer.Confirm('continue', message="Algoritma karşılaştırması bir tablo üzerinde gösterilsin mi?") ]
				answer = inquirer.prompt(questions)['continue']

				if answer:
					print(Fore.YELLOW + "\n\t--KDD-----------------------------------------------" + Style.RESET_ALL)
					print(Fore.BLUE + "\tÖzellik\t\t\tID3 (Ent.)\tCART (Gini)" + Style.RESET_ALL)
					print(Fore.YELLOW + "\t----------------------------------------------------" + Style.RESET_ALL)
					print("\tÖğrenme Süresi" + "\t\t" + "%.3f saniye" % id3[2] + "\t" + "%.3f saniye" % cart[2])
					print(Fore.YELLOW + "\t----------------------------------------------------" + Style.RESET_ALL)
					print("\tSınıflandırma Süresi" + "\t" + "%.3f saniye" % id3[3] + "\t" + "%.3f saniye" % cart[3])
					print(Fore.YELLOW + "\t----------------------------------------------------" + Style.RESET_ALL)
					print("\tYaprak Sayısı" + "\t\t" + str(id3[5]) + "\t\t" + str(cart[5]))
					print(Fore.YELLOW + "\t----------------------------------------------------" + Style.RESET_ALL)
					print("\tDerinlik" + "\t\t" + str(id3[4]) + "\t\t" + str(cart[4]))
					print(Fore.YELLOW + "\t----------------------------------------------------" + Style.RESET_ALL)
					print("\tDoğruluk (Accuracy)" + "\t" + "%.4f" % id3[0] + "\t\t" + "%.4f" % cart[0])
					print(Fore.YELLOW + "\t----------------------------------------------------" + Style.RESET_ALL)

					print("\tID3 doğruluk matrisi\n")
					print(id3[1])
					print("\t----------------------------------------------------")
					print("\tCART doğruluk matrisi\n")
					print(cart[1])
					print("\t----------------------------------------------------")
					print("\tPrecision = Tp / Tp + Fp\n\tRecall = Tp / Tp + Fn\n\tF1 = 2 * (Precision * Recall) / Precision + Recall")
					print("\t----------------------------------------------------")

				set_config(settings.configs_algorithm_status_parameter, settings.configs_algorithm_status_values[3])
		else:
			print(Fore.RED + '--> ' + Style.RESET_ALL + 'Hiçbir özellik seçilmediğinden işlem iptal edildi.')

def get_tree_from_file(filename):
	import pickle
	fr = open(filename, 'rb')
	sma_tree = pickle.load(fr)
	return sma_tree

def prediction_of_line(tree_file_name, columns, line, algorithm):
	if algorithm == "id3":
		firstStr = list(tree_file_name.keys())[0]
		secondDict = tree_file_name[firstStr]
		featatureIndex = columns.index(firstStr)
		predictedClass = settings.classes_values[0]
		for key in secondDict.keys():
			if line[featatureIndex] == key:
				if type(secondDict[key]).__name__ == 'dict':
					predictedClass = prediction_of_line(secondDict[key], columns, line, "id3")
				else:
					predictedClass = secondDict[key]
		return predictedClass
	elif algorithm == "cart":
		firstStr = list(tree_file_name.keys())[0]
		secondDict = tree_file_name[firstStr]
		featatureIndex = columns.index(firstStr)
		predictedClass = settings.classes_values[0]
		for key in secondDict.keys():
			if key[0] == "<":
				if float(line[featatureIndex]) < float(key[1:]):
					if type(secondDict[key]).__name__ == 'dict':
						predictedClass = prediction_of_line(secondDict[key], columns, line, "cart")
					else:
						predictedClass = secondDict[key]
			elif key[0] == ">":
				if float(line[featatureIndex]) >= float(key[1:]):
					if type(secondDict[key]).__name__ == 'dict':
						predictedClass = prediction_of_line(secondDict[key], columns, line, "cart")
					else:
						predictedClass = secondDict[key]
		return predictedClass

def classify(operation_type, classify_table, classify_table_primary_key, mining_columns, where_clause=None, answers = []):
	print("\n\n-----------------------------------------------------------")

	output = True

	if(len(answers) != 0):
		output = False

	if(output):
		columns = mining_columns.split(",")
		questions = [
						inquirer.List('column_option',
						message="Veri madenciliği işlemlerinde kullanılacak özellikleri belirlemek ister misiniz?",
						choices=[Fore.YELLOW + 'Evet' + Style.RESET_ALL + ', özellik seç.', Fore.RED + 'Hayır' + Style.RESET_ALL + ', tümünü kullan.', Fore.BLUE + "İptal" + Style.RESET_ALL + ", ana menüye dön!"],
						),
					]
		answer = inquirer.prompt(questions)["column_option"]

		if "İptal" in answer:
			clear_screen()
			return

		if "Evet" in answer:
			questions = [
							inquirer.Checkbox('columns',
							message="Lütfen özellik seçiniz. Hiçbir özellik seçilmezse işlem iptal edilecektir",
							choices=mining_columns.split(","),
							),
						]
			columns = inquirer.prompt(questions)["columns"]
		columns = ",".join(str(x) for x in columns)
	else:
		columns = answers[0]

	if len(columns) != 0:
		print("\nKayıtlar getiriliyor... ")
		if where_clause is not None:
			query = ("select " + classify_table_primary_key + "," + columns + " from " + classify_table + " " + where_clause + " order by " + classify_table_primary_key)
		else:
			query = ("select " + classify_table_primary_key + "," + columns + " from " + classify_table + " order by " + classify_table_primary_key)
		cursor = sql.query_db(query)

		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Kayıtlar getirildi. Diziye aktarılıyor...")
		data = []
		total = cursor.rowcount

		if (output):
			pbar = tqdm(total=total, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} | {elapsed} ")
			pbar.set_description("    İşleniyor")

		for record in cursor:
			data.append(record)
			if (output):
				pbar.update(1)

		if (output):
			pbar.close()

		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Kayıtlar diziye aktarıldı.",flush=True)
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Kayıt Sayısı: {}".format(len(data)))

		if(output):
			questions = [
					inquirer.List('algorithm',
					message="Hangi veri madenciliği algoritması ile sınıflandırma yapmak istersiniz?",
					choices=[Fore.YELLOW + 'ID3' + Style.RESET_ALL + ' algoritmasını kullan.', Fore.YELLOW + 'CART' + Style.RESET_ALL + ' algoritmasını kullan.'],
					),
				]
			answers = inquirer.prompt(questions)["algorithm"]
		else:
			answers = answers[1]

		if "CART" in answers:
			if operation_type == "kdd":
				if control_cart():
					training_tree = get_tree_from_file(settings.cart_tree_file_path)
					algorithm = "cart"
				else:
					algorithm = "not-learned"
			if operation_type == "nsl":
				if control_nsl():
					training_tree = get_tree_from_file(settings.nsl_cart_tree_file_path)
					algorithm = "cart"
				else:
					algorithm = "not-learned"

		elif "ID3" in answers:
			if operation_type == "kdd":
				if control_id3():
					training_tree = get_tree_from_file(settings.id3_tree_file_path)
					algorithm = "id3"
				else:
					algorithm = "not-learned"
			if operation_type == "nsl":
				if control_nsl():
					training_tree = get_tree_from_file(settings.nsl_id3_tree_file_path)
					algorithm = "id3"
				else:
					algorithm = "not-learned"

		if algorithm != "not-learned":
			print("Sınıflandırma işlemi yapılıyor...")
			class_predicted = []

			if (output):
				pbar = tqdm(total=total, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} | {elapsed} ")
				pbar.set_description("    İşleniyor")
			columns = columns.split(",")
			columns.append(classify_table_primary_key)
			for line in data:
				predicted_class = prediction_of_line(training_tree, columns, line[1:], algorithm)
				class_predicted.append([line[0], predicted_class])
				if (output):
					pbar.update(1)
			if (output):
				pbar.close()
			print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Sınıflandırma işlemi tamamlandı.",flush=True)
			print("Veri tabanı güncelleniyor...",flush=True)
			update_records(class_predicted, [classify_table_primary_key, settings.prediction_column_name], classify_table, False)
			print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Veri tabanı güncellendi.",flush=True)
			if classify_table == settings.traffic_table_name:
				copy_unknown_attacks()
	else:
		print(Fore.RED + '--> ' + Style.RESET_ALL + 'Hiçbir özellik seçilmediğinden işlem iptal edildi.')

def copy_unknown_attacks():
	query = ('select * from ' + settings.traffic_table_name + ' where prediction = "' + settings.unknown_class_value + '"')
	cursor = sql.query_db(query)
	for record in cursor:
		line = list(record)
		values = ""

		query = ('select * from ' + settings.expert_data_table_name + ' where ' + settings.expert_data_primarykey_column + ' = ' + line[0])
		cursor = sql.query_db(query)
		total = 0
		for c in cursor:
			total = total + 1

		if total==0:
			for l in line:
				try:
					l = float(l)
					values = values + str(l) + ','
				except:
					values = values + '"' + l + '",'
			values = values[:-1]

			query = ("insert into " + settings.expert_data_table_name + "(" + "id,prediction," + settings.traffic_table_columns + ") values (" + values + ")")
			sql.query_db(query)
	sql.query_commit()


def execute_shell_command(command):
	subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)

def check_remote_file_exist(remote_user, remote_host, file_name):
	command = 'ssh ' + remote_user + '@' + remote_host + ' [ -f ' + file_name + ' ] && echo "Found" || echo "Not found"'
	output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)

	if "Not" in str(output):
		print(Fore.RED + "--> " + Style.RESET_ALL + file_name + " dosyası " + remote_host + " adresinde bulunamadı!\n")
		return False
	return True

def check_local_file_exist(file_name):
	command = '[ -f ' + file_name + ' ] && echo "Found" || echo "Not found"'
	output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)

	if "Not" in str(output):
		print(Fore.RED + "--> " + Style.RESET_ALL + file_name + " dosyası işletim sisteminde bulunamadı!\n")
		return False
	return True

def update_traffic_records():

	print("\n\n-----------------------------------------------------------")
	questions = [
					inquirer.List('option',
					message="Gerçek trafik kayıtları tablosuna yeni kayıtlar eklenecek. Devam etmek istiyor musunuz?",
					choices=[Fore.YELLOW + 'Evet' + Style.RESET_ALL, Fore.BLUE + "Hayır" + Style.RESET_ALL],
					),
				]
	answers = inquirer.prompt(questions)["option"]

	if "Evet" in answers:
		total = 6
		index = 1

		status = True

		if check_remote_file_exist(settings.listener_user, settings.listener_host, settings.listener_created_pcap_file_path) and check_remote_file_exist(settings.listener_user, settings.listener_host, settings.listener_bro_policy_file_path):
			print(Fore.YELLOW + "--> " + Style.RESET_ALL + "[ " + Fore.BLUE + str(index) + Style.RESET_ALL + " | " + Fore.BLUE + str(total) + Style.RESET_ALL + " ] PCAP dosyasından bağlantı bilgileri çıkarılıyor...")
			execute_shell_command('ssh ' + settings.listener_user + '@' + settings.listener_host + '  "bro -r ' + settings.listener_created_pcap_file_path +' ' + settings.listener_bro_policy_file_path + ' > ' + settings.listener_conn_list_file_path + '"')
			print(Fore.YELLOW + "    --> " + Style.RESET_ALL + "İşlem tamamlandı.\n")
		else:
			status = False

		if status:
			if check_remote_file_exist(settings.listener_user, settings.listener_host, settings.listener_conn_list_file_path):
				index = index + 1
				print(Fore.YELLOW + "--> " + Style.RESET_ALL + "[ " + Fore.BLUE + str(index) + Style.RESET_ALL + " | " + Fore.BLUE + str(total) + Style.RESET_ALL + " ] Bağlantı bilgileri bağlanma sayısına göre sıralanıyor...")
				execute_shell_command('ssh ' + settings.listener_user + '@' + settings.listener_host + ' "sort -n ' + settings.listener_conn_list_file_path + ' > ' + settings.listener_conn_sorted_list_file_path + '"')
				execute_shell_command('ssh ' + settings.listener_user + '@' + settings.listener_host + '  "rm ' + settings.listener_conn_list_file_path + '"')
				print(Fore.YELLOW + "    --> " + Style.RESET_ALL + "İşlem tamamlandı.\n")
			else:
				status = False

		if status:
			if check_remote_file_exist(settings.listener_user, settings.listener_host, settings.listener_conn_sorted_list_file_path) and check_remote_file_exist(settings.listener_user, settings.listener_host, settings.listener_traffic_analyzer_binary_file_path):
				index = index + 1
				print(Fore.YELLOW + "--> " + Style.RESET_ALL + "[ " + Fore.BLUE + str(index) + Style.RESET_ALL + " | " + Fore.BLUE + str(total) + Style.RESET_ALL + " ] Bağlantı bilgileri kullanılarak istatistiksel veriler hesaplanıyor...")
				execute_shell_command('ssh ' + settings.listener_user + '@' + settings.listener_host + ' "' + settings.listener_traffic_analyzer_binary_file_path + ' ' + settings.listener_conn_sorted_list_file_path + '"')
				execute_shell_command('ssh ' + settings.listener_user + '@' + settings.listener_host + ' "rm ' + settings.listener_conn_sorted_list_file_path + '"')
				print(Fore.YELLOW + "    --> " + Style.RESET_ALL + "İşlem tamamlandı.\n")
			else:
				status = False

		if status:
			if check_remote_file_exist(settings.listener_user, settings.listener_host, settings.listener_traffic_analyzer_output_file_path):
				index = index + 1
				print(Fore.YELLOW + "--> " + Style.RESET_ALL + "[ " + Fore.BLUE + str(index) + Style.RESET_ALL + " | " + Fore.BLUE + str(total) + Style.RESET_ALL + " ] Hesaplanan veriler alınıyor...")
				execute_shell_command('scp ' + settings.listener_user + '@' + settings.listener_host + ':' + settings.listener_traffic_analyzer_output_file_path + ' ' + settings.dm_traffic_infos_file_path)
				execute_shell_command('ssh ' + settings.listener_user + '@' + settings.listener_host + ' "rm ' + settings.listener_traffic_analyzer_output_file_path + '"')
				print(Fore.YELLOW + "    --> " + Style.RESET_ALL + "İşlem tamamlandı.\n")
			else:
				status = False

		if status:
			if check_local_file_exist(settings.dm_traffic_infos_file_path):
				index = index + 1
				print(Fore.YELLOW + "--> " + Style.RESET_ALL + "[ " + Fore.BLUE + str(index) + Style.RESET_ALL + " | " + Fore.BLUE + str(total) + Style.RESET_ALL + " ] Alınan veriler üzerindeki gereksiz kayıtlar temizleniyor...")
				execute_shell_command('cut -d" " -f2- ' + settings.dm_traffic_infos_file_path + ' > ' + settings.dm_traffic_infos_edited_file_path)
				execute_shell_command('rm ' + settings.dm_traffic_infos_file_path)
				print(Fore.YELLOW + "    --> " + Style.RESET_ALL + "İşlem tamamlandı.\n")
			else:
				status = False

		if status:
			if check_local_file_exist(settings.dm_traffic_infos_edited_file_path):
				index = index + 1
				print(Fore.YELLOW + "--> " + Style.RESET_ALL + "[ " + Fore.BLUE + str(index) + Style.RESET_ALL + " | " + Fore.BLUE + str(total) + Style.RESET_ALL + " ] Veri tabanı güncelleniyor...")

				with open(settings.dm_traffic_infos_edited_file_path) as traffics:
					total = 0
					for traffic in traffics:
						total = total + 1
				
				with open(settings.dm_traffic_infos_edited_file_path) as traffics:	
					pbar = tqdm(total=total, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} | {elapsed} ")
					pbar.set_description("    İşleniyor")
					for traffic in traffics:
						traffic_infos = traffic.split(" ")
						values = ""
						for info in traffic_infos:
							try:
								info = float(info)
								values = values + str(info) + ',' 
							except:
								values = values + '"' + info + '",' 
						values = values[:-1]
						query = ("insert into " + settings.traffic_table_name + "(" + settings.traffic_table_columns + ") values (" + values + ")")
						sql.query_db(query)
						pbar.update(1)
					pbar.close()
					sql.query_commit()
				execute_shell_command('rm ' + settings.dm_traffic_infos_edited_file_path)
				
				query = ("analyze table " + settings.traffic_table_name)
				sql.query_db(query)
				sql.query_commit()
				
				print(Fore.YELLOW + "    --> " + Style.RESET_ALL + "İşlem tamamlandı.")

def disable_print():
	sys.stdout = open(os.devnull, 'w')

def enable_print():
	sys.stdout = sys.__stdout__


def classify_traffic_records(operation_type, answers = []):
	where_clause = 'where prediction = "not_analyzed"'
	record_status = change_data_format(settings.traffic_table_name, settings.configs_operations[1], where_clause)
	if record_status != False:
		where_clause = 'where prediction = "not_analyzed" OR prediction = "' + settings.unknown_class_value + '"'
		discretization(settings.traffic_table_name, settings.configs_operations[1], where_clause)
		classify(operation_type, settings.traffic_table_name, settings.traffic_table_primary_key, settings.mining_columns, where_clause, answers)
	else:
		print(Fore.RED + "--> " + Style.RESET_ALL + "Tüm kayıtlar hali hazırda sınıflandırılmış.")

def realtime_classify_traffic_records(operation_type):
	print("\n\n-----------------------------------------------------------")
	questions = [
					inquirer.List('option',
					message="Siber Tehdit İstihbarat Verisi için yeni trafik kayıtları eklenecek. Devam etmek istiyor musunuz?",
					choices=[Fore.YELLOW + 'Evet' + Style.RESET_ALL, Fore.BLUE + "Hayır" + Style.RESET_ALL],
					),
				]
	answers = inquirer.prompt(questions)["option"]

	mining_columns = settings.mining_columns

	if "Evet" in answers:
		columns = mining_columns.split(",")

		questions = [
				inquirer.List('column_option',
				message="Veri madenciliği işlemlerinde kullanılacak özellikleri belirlemek ister misiniz?",
				choices=[Fore.YELLOW + 'Evet' + Style.RESET_ALL + ', özellik seç.', Fore.RED + 'Hayır' + Style.RESET_ALL + ', tümünü kullan.', Fore.BLUE + "İptal" + Style.RESET_ALL + ", ana menüye dön!"],
				),
			]
		answers = inquirer.prompt(questions)["column_option"]

		if "İptal" in answers:
			clear_screen()
			return

		if "Evet" in answers:
			questions = [
							inquirer.Checkbox('columns',
							message="Lütfen özellik seçiniz. Hiçbir özellik seçilmezse işlem iptal edilecektir",
							choices=mining_columns.split(","),
							),
						]
			columns = inquirer.prompt(questions)["columns"]
		columns = ",".join(str(x) for x in columns)
		if len(columns) != 0:

			questions = [
				inquirer.List('algorithm',
				message="Hangi veri madenciliği algoritması ile sınıflandırma yapmak istersiniz?",
				choices=[Fore.YELLOW + 'ID3' + Style.RESET_ALL + ' algoritmasını kullan.', Fore.YELLOW + 'CART' + Style.RESET_ALL + ' algoritmasını kullan.'],
				),
			]
			algorithm_answer = inquirer.prompt(questions)["algorithm"]

			try:
				while (True):
					print(Fore.BLUE + "\n    --> Dinleniyor..." + Style.RESET_ALL + "\n")
					try:
						execute_shell_command('ssh ' + settings.listener_user + '@' + settings.listener_host + '  "timeout 10 tcpdump -w ' + settings.listener_realtime_pcap_file_path + ' -i any"')
					except:
						pass
					execute_shell_command('ssh ' + settings.listener_user + '@' + settings.listener_host + '  "bro -r ' + settings.listener_realtime_pcap_file_path +' ' + settings.listener_bro_policy_file_path + ' > ' + settings.listener_conn_list_file_path + '"')
					execute_shell_command('ssh ' + settings.listener_user + '@' + settings.listener_host + '  "rm ' + settings.listener_realtime_pcap_file_path + '"')
					execute_shell_command('ssh ' + settings.listener_user + '@' + settings.listener_host + ' "sort -n ' + settings.listener_conn_list_file_path + ' > ' + settings.listener_conn_sorted_list_file_path + '"')
					execute_shell_command('ssh ' + settings.listener_user + '@' + settings.listener_host + '  "rm ' + settings.listener_conn_list_file_path + '"')
					execute_shell_command('ssh ' + settings.listener_user + '@' + settings.listener_host + ' "' + settings.listener_traffic_analyzer_binary_file_path + ' ' + settings.listener_conn_sorted_list_file_path + '"')
					execute_shell_command('ssh ' + settings.listener_user + '@' + settings.listener_host + ' "rm ' + settings.listener_conn_sorted_list_file_path + '"')
					execute_shell_command('scp ' + settings.listener_user + '@' + settings.listener_host + ':' + settings.listener_traffic_analyzer_output_file_path + ' ' + settings.dm_traffic_infos_file_path)
					execute_shell_command('ssh ' + settings.listener_user + '@' + settings.listener_host + ' "rm ' + settings.listener_traffic_analyzer_output_file_path + '"')
					execute_shell_command('cut -d" " -f2- ' + settings.dm_traffic_infos_file_path + ' > ' + settings.dm_traffic_infos_edited_file_path)
					execute_shell_command('rm ' + settings.dm_traffic_infos_file_path)

					with open(settings.dm_traffic_infos_edited_file_path) as traffics:
						total = 0
						for traffic in traffics:
							total = total + 1
					
					print(Fore.YELLOW + "    --> " + Style.RESET_ALL + str(total) + " paket yakalandı.")

					with open(settings.dm_traffic_infos_edited_file_path) as traffics:	
						for traffic in traffics:
							traffic_infos = traffic.split(" ")
							values = ""
							for info in traffic_infos:
								try:
									info = float(info)
									values = values + str(info) + ',' 
								except:
									values = values + '"' + info + '",' 
							values = values[:-1]
							query = ("insert into " + settings.traffic_table_name + "(" + settings.traffic_table_columns + ") values (" + values + ")")
							sql.query_db(query)
						sql.query_commit()
					execute_shell_command('rm ' + settings.dm_traffic_infos_edited_file_path)
					print(Fore.YELLOW + "    --> " + Style.RESET_ALL + str(total) + " paket veri tabanına kaydedildi.")

					disable_print()
					classify_traffic_records(operation_type, [columns,algorithm_answer])
					enable_print()
					print(Fore.YELLOW + "    --> " + Style.RESET_ALL + str(total) + " paket önişlemden geçirildi ve sınıflandırıldı.")

					query = ("select " + settings.prediction_column_name + "," + settings.traffic_table_cti_columns + " from " + settings.traffic_table_name + " order by " + settings.traffic_table_primary_key + " DESC LIMIT " + str(total))
					cursor = sql.query_db(query)

					for record in cursor:
						if (record[0] != "normal"):
							print(Fore.RED + "    --> ALARM:" + Style.RESET_ALL + " Atak türü: " + str(record[0]) + " (" + str(record[1]) + ":" + str(int(record[2])) + " -> " + str(record[3]) + ":" + str(int(record[4])) + ")")
						else:
							print(Fore.GREEN + "    --> NORMAL:" + Style.RESET_ALL + " (" + str(record[1]) + ":" + str(int(record[2])) + " -> " + str(record[3]) + ":" + str(int(record[4])) + ")")
			except KeyboardInterrupt:
				pass
		else:
			print(Fore.RED + '--> ' + Style.RESET_ALL + 'Hiçbir özellik seçilmediğinden işlem iptal edildi.')

def delete_tree_file(algorithm):
	try:
		if algorithm == "id3":
			os.remove(settings.id3_tree_file_path + "_diagram.png")
		elif algorithm == "cart":
			os.remove(settings.cart_tree_file_path + "_diagram.png")
	except:
		pass


def portsweep_accuracy():
	mining_columns = settings.mining_columns
	columns = mining_columns.split(",")
	columns = ",".join(str(x) for x in columns)
	algorithm_answer = "CART"
	operation_type = "nsl"
	traffic_table_name = settings.nsl_update_traffic_table_name
	attack = "portsweep"

	print("\n\n-----------------------------------------------------------")
	questions = [
					inquirer.List('option',
					message="'portsweep' saldırısının hangi sürümünü test etmek istiyorsunuz?",
					choices=['İlk ' + Fore.YELLOW + '3.000 port' + Style.RESET_ALL + ' ve ' + Fore.YELLOW + 'normal' + Style.RESET_ALL + ' hızda saldırı', 
					'İlk ' + Fore.YELLOW + '3.000 port' + Style.RESET_ALL + ' ve ' + Fore.YELLOW + 'agrasif' + Style.RESET_ALL + ' hızda saldırı',
					'İlk ' + Fore.YELLOW + '10.000 port' + Style.RESET_ALL + ' ve ' + Fore.YELLOW + 'en yüksek' + Style.RESET_ALL + ' hızda saldırı',
					Fore.YELLOW + 'Tüm portlar' + Style.RESET_ALL + ' ve ' + Fore.YELLOW + 'normal' + Style.RESET_ALL + ' hızda saldırı']
					),
				]
	answers = inquirer.prompt(questions)["option"]

	pcap_add = ""
	if "3.000" in answers:
		if "normal" in answers:
			pcap_add = "3"
		if "agrasif" in answers:
			pcap_add = "4"

	if "10.000" in answers:
		pcap_add = "10000"

	if "Tüm" in answers:
		pcap_add = "65"

	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Dinleme modülünden 'portsweep' saldırısı için gerçek trafik kayıtları getiriliyor...")
	execute_shell_command('ssh root@192.168.63.3 "bro -r /home/ubuntulistener/guncellik_testi/' + attack +'/created_traffic' + pcap_add + '.pcap ' + settings.listener_bro_policy_file_path + ' > /home/ubuntulistener/guncellik_testi/' + attack +'/conn_list"')
	execute_shell_command('ssh root@192.168.63.3 "sort -n /home/ubuntulistener/guncellik_testi/' + attack +'/conn_list > /home/ubuntulistener/guncellik_testi/' + attack +'/conn_list_sort"')
	execute_shell_command('ssh root@192.168.63.3 "/home/ubuntulistener/trafAld.out /home/ubuntulistener/guncellik_testi/' + attack +'/conn_list_sort"')
	execute_shell_command('scp root@192.168.63.3:/root/trafAld.list /home/ubuntudm/deneme/liste.txt')
	execute_shell_command('cut -d" " -f2- /home/ubuntudm/deneme/liste.txt > /home/ubuntudm/deneme/liste_edited.txt')
	print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + "İşlem tamamlandı.")
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Veri tabanı tablosu temizleniyor...")
	query = ("delete from " + traffic_table_name )
	sql.query_db(query)
	sql.query_commit()
	print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + "İşlem tamamlandı.")
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Yeni kayıtlar veri tabanına kaydediliyor...")
	with open('/home/ubuntudm/deneme/liste_edited.txt') as traffics:	
		for traffic in traffics:
			traffic_infos = traffic.split(" ")
			values = ""
			for info in traffic_infos:
				try:
					info = float(info)
					values = values + str(info) + ',' 
				except:
					values = values + '"' + info + '",' 
			values = values[:-1]
			query = ("insert into " + traffic_table_name + "(" + settings.traffic_table_columns + ") values (" + values + ")")
			sql.query_db(query)
		sql.query_commit()
	print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + "İşlem tamamlandı.")
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Önişleme ve dönüştürme adımları uygulanıyor...")
	disable_print()
	answers = [columns,algorithm_answer]
	where_clause = 'where prediction = "not_analyzed"'
	record_status = change_data_format(traffic_table_name, settings.configs_operations[1], where_clause)
	if record_status != False:
		where_clause = 'where prediction = "not_analyzed" OR prediction = "' + settings.unknown_class_value + '"'
		discretization(traffic_table_name, settings.configs_operations[1], where_clause)
		enable_print()
		print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + "İşlem tamamlandı.")
		print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Kayıtlar sınıflandırılıyor...")
		disable_print()
		classify(operation_type, traffic_table_name, settings.traffic_table_primary_key, settings.mining_columns, where_clause, answers)
	enable_print()
	print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + "İşlem tamamlandı.")
	
	query = ("select " + settings.prediction_column_name + "," + settings.traffic_table_cti_columns + " from " + traffic_table_name + " order by " + settings.traffic_table_primary_key)
	cursor = sql.query_db(query)

	count_true = 0
	count_total = 0
	others = []
	others_total = []
	for record in cursor:
		prediction = record[0]
		if prediction == attack:
			count_true = count_true + 1
		else:
			if prediction not in others:
				others.append(prediction)
				others_total.append(0)
			others_total[others.index(prediction)] = others_total[others.index(prediction)] + 1
		count_total = count_total + 1

	p = (count_true/count_total)*100
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Doğruluk: %" + '%.2f' % p)
	print(Fore.YELLOW + "--> " + Style.RESET_ALL + "Tahminler: " + "( Toplam kayıt: " + str(count_total) + ")")
	print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + "portsweep: " + str(count_true) + " adet ( %" + '%.2f' % p + " )")
	for prediction in others:
		t = others_total[others.index(prediction)]
		p = (t/count_total)*100
		print(Fore.YELLOW + "--> --> " + Style.RESET_ALL + prediction + ": " + str(t) + " adet ( %" + '%.2f' % p + " )")


initialize()

