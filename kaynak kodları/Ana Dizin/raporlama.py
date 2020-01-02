import eventlet
eventlet.monkey_patch()
from flask import Flask, render_template, request, send_file
from flask_socketio import SocketIO, emit
import time
import araclar.db_sql as db_op
from queue import Queue
from threading import Thread
import random
import ayarlar.ayarlar as settings
import socket
import os
import fcntl
import struct
import datetime
import stix2 as stiv

#message_producer("print", "ekrana bas 1"})
#message_producer("print", "-----------"})
#emit('page', {'message': "continue"})

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading", ping_timeout=3600)
thread = None
q = Queue()

def message_producer(event, message):
	client = request.sid
	socketio.emit(event, {'message': message}, room=client)
	eventlet.sleep(0)

########################################################################################################################################################################################
# Page Routing

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/db')
def db():
	try:
		query = "SHOW columns FROM " + settings.web_database_show_table_name
		cursor = db_op.query_db(query)
		columns = []
		for infos in cursor:
			columns.append(infos[0])
	except:
		columns = ["Hata: İlgili tablodan sütun bilgisi getirilemedi!"]
	return render_template('db.html', columns=columns)

@app.route('/reports')
def stis():
	return render_template('reports.html', columns=["STİ Dosya Adı"])

@app.route('/stifiles')
def stifiles():
	file_list = os.listdir("./raporlar")
	count = len(file_list)
	file_list.sort(reverse=True)

	stis = str(file_list)
	
	sti_files=[]
	for s in file_list:
		l = []
		l.append(s)
		sti_file = []
		sti_file.append(s)
		sti_files.append(sti_file)

	sti_file_list = str(sti_files)
	result = '{ "draw": ' + str(1) + ', "recordsTotal": ' + str(count) + ', "recordsFiltered": ' + str(count) + ', "data": ' + sti_file_list +'}'
	result = result.replace("'", '"')
	return result

@app.route('/records')
def records():
	draw = int(request.args.get('draw'))
	page = int(request.args.get('start'))
	length = int(request.args.get('length'))
	return get_database_records(draw, page, length, settings.web_database_show_table_name)

@app.route('/traffic')
def traffic_home():
	try:
		query = "SHOW columns FROM " + settings.traffic_table_name
		cursor = db_op.query_db(query)
		columns = []
		for infos in cursor:
			columns.append(infos[0])
	except:
		columns = ["Hata: İlgili tablodan sütun bilgisi getirilemedi!"]
	return render_template('traffic.html', columns=columns)

@app.route('/traffics')
def traffics():
	draw = int(request.args.get('draw'))
	page = int(request.args.get('start'))
	length = int(request.args.get('length'))
	return get_database_records(draw, page, length, settings.traffic_table_name)

def get_attack_pattern(attack_patterns, prediction):
	for ap in attack_patterns:
		if ap.name == prediction:
			return ap
	return None

def unique_lists_from_multidimensional_array(multidimensional):
	seen = set()
	newlist = []
	for item in multidimensional:
		t = tuple(item)
		if t not in seen:
			newlist.append(item)
			seen.add(t)
	return newlist

@app.route('/generate_report')
def generate_report():
	istihbarat_no = 0
	cyber_threads_infos = []
	attack_patterns = []
	sti_list = []

	records = get_all_records(settings.traffic_table_name, settings.prediction_column_name)
	unique_records = unique_lists_from_multidimensional_array(records)

	identity = stiv.Identity(name="Veri Madenciligi Temelli Siber Tehdit Istihbarati Tez Calismasi Onerilen Sistemin Uygulamasi - Suleyman Muhammed ARIKAN", identity_class="individual")
	sti_list.append(identity)

	predictions = [record[1] for record in unique_records]
	unique_predictions = set(predictions)

	for attack in unique_predictions:
		attact_pattern = stiv.AttackPattern(name=attack, created_by_ref=identity.id)
		attack_patterns.append(attact_pattern)
		sti_list.append(attact_pattern)

	cyber_threads_infos.append("intelligence_id-prediction-source_host-source_port-destination_port-protocol" + "\n")
	for record in records:
		if str(record[1]) != settings.unknown_class_value:
			cyber_thread_info = str(record[1]) + "-" + str(record[5]) + "-" + str(int(float(record[3]))) + "-" + str(int(float(record[4]))) + "-" + settings.protocols_list[int(record[8]) - 1] + "\n"
			cyber_threads_infos.append(cyber_thread_info)

			istihbarat_no = istihbarat_no + 1
			indicator_name = str(istihbarat_no) + " numaralı istihbarat - " + str(record[5])
			indicator_label = ["malicious-activity"]
			indicator_pattern = "[network-traffic:src_ref.type = 'ipv4-addr' AND "
			indicator_pattern = indicator_pattern + "network-traffic:src_ref.value = '" + str(record[5]) + "' AND "
			indicator_pattern = indicator_pattern + "network-traffic:src_port = " + str(int(float(record[3]))) + " AND "
			indicator_pattern = indicator_pattern + "network-traffic:dst_port = " + str(int(float(record[4]))) + " AND "
			indicator_pattern = indicator_pattern + "network-traffic:protocols[*] = '" + settings.protocols_list[int(record[8]) - 1] + "']"

			indicator = stiv.Indicator(name=indicator_name, labels=indicator_label, pattern=indicator_pattern, created_by_ref=identity.id)
			sti_list.append(indicator)

			attack_pattern = get_attack_pattern(attack_patterns, str(record[1]))
			relationship = stiv.Relationship(relationship_type='indicates', source_ref=indicator.id, target_ref=attack_pattern.id)
			sti_list.append(relationship)

	stiv_bundle = stiv.Bundle(sti_list)

	file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") 

	fo = open("raporlar/" + file_name + ".txt", "x")
	istihbarat_no = 0
	for cti in cyber_threads_infos:
		if (istihbarat_no != 0):
			cti = str(istihbarat_no) + "-" + cti
		fo.write(cti)
		istihbarat_no = istihbarat_no + 1
	fo.close()
	
	fo_stiv = open("raporlar/" + file_name + ".json", "x")
	fo_stiv.write(stiv_bundle.serialize())
	fo_stiv.close()
	return file_name

@app.route('/generated_report/<string:file_name>', methods=['GET'])
def generated_report(file_name):
	return send_file('raporlar/' + file_name, file_name)

@app.route('/dm')
def dm_home():
	return render_template('dm_shell.html')

@app.route('/dm/ip')
def ip():
	return get_url_address()

@app.route('/dm/tree/<string:tr>')
def tree(tr):
	if ((tr == "Id3") or (tr == "Cart") or (tr == "nsl_id3") or (tr == "nsl_cart")):
		status, image = get_tree_image(tr)

		if status:
			return send_file(image, cache_timeout=-1), 200
	return '', 404

@app.route('/expert/<string:id>/<string:cls>', methods=['GET'])
def expert(id,cls):
	query = 'UPDATE ' + settings.expert_data_table_name + ' SET ' + settings.expert_prediction_column_name + '="' + cls + '" where ' + settings.expert_data_primarykey_column + '=' + id
	cursor = db_op.query_db(query)
	print(query)
	db_op.query_commit()
	return "ok"

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
########################################################################################################################################################################################
# Socket Routing

@socketio.on('homepage')
def homepage():
	message_producer("page", "continue")
	show_database_connection_time()
	#message_producer("page", "interrupt")

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def connect_database():
	connectionStartTime = time.time()
	db_op.connect_db()
	return time.time() - connectionStartTime

def get_database_records(draw, start, length, table_name):
	try:
		query = "SELECT * FROM " + table_name + " limit %s, %s;" % (start, length)
		cursor = db_op.query_db(query)

		data_array = []
		for c in cursor:
			data_array.append(list(c))

		query_count = 'SELECT TABLE_ROWS FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = "' + table_name + '";'
		cursor_count = db_op.query_db(query_count)
		count = cursor.fetchone()[0]

		result = '{ "draw": ' + str(draw) + ', "recordsTotal": ' + str(count) + ', "recordsFiltered": ' + str(count) + ', "data": ' + str(data_array) +'}'
		result = result.replace("'", '"')

		return result
	except Exception as e:
		result = '{ "draw": 0, "recordsTotal": 0, "recordsFiltered": 0, "data": []}'
		return result

def get_all_records(table_name, prediction_column):
	query = 'SELECT * FROM ' + table_name + ' where ' + prediction_column + ' <> "' + settings.classes_values[0] + '" order by ' + prediction_column + ';'
	cursor = db_op.query_db(query)

	data_array = []
	for c in cursor:
		data_array.append(list(c))
	return data_array

def show_database_connection_time():
	message_producer("print", "-----------------------------------------------------------")
	message_producer("print", "--> %.3f saniyede veri tabanına bağlantı sağlandı." % durationOfConnection)
	message_producer("print", "-----------------------------------------------------------")

durationOfConnection = connect_database()

def get_url_address():
	return "http://" + settings.dm_host + ":" + str(settings.dm_host_datamining_shell_port)

def get_tree_image(tr):
	image_path = 'utils/tree/sma_agac_' + tr.lower() + '_tree_diagram.png'
	algorithm_status = ""
	query = ("select " + settings.configs_table_value_column + " from " + settings.configs_table_name + " where " + settings.configs_table_parameter_column + " = \"{}\"".format(settings.configs_algorithm_status_parameter))
	cursor = db_op.query_db(query)
	if cursor.rowcount == 0:
		set_config(parameter, settings.configs_sample_status_values[0])
		algorithm_status = settings.configs_sample_status_values[0]
	else:
		algorithm_status = cursor.fetchone()[0]

	if ((algorithm_status == tr) or (algorithm_status == "All")):
		return (os.path.isfile('./' + image_path), image_path)
	else:
		return (False, image_path)

if __name__ == '__main__':
	os.system("./gotty -c " + settings.shell_username + ":" + settings.shell_password + " -w python3 sts.py --max-connection 1 &")
	socketio.run(app, host='0.0.0.0', port=settings.dm_host_web_port, debug=False)
