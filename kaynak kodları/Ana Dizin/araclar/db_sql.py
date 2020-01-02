import mysql.connector
from mysql.connector import errorcode
from tqdm import tqdm
import os
import sys
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
import ayarlar.ayarlar as settings

cursor = None
cnx = None

def connect_db():
	global cnx
	try:
		global cnx
		cnx = mysql.connector.connect(user=settings.db_user, password=settings.db_password, host=settings.db_host, database=settings.db_name)
		global cursor
		cursor = cnx.cursor(buffered=True)
		return True, ""
	except mysql.connector.Error as err:
		cnx = None
		cursor = None
		return False, str(err)

def query_db(query):
	if cursor != None and cnx != None:
		cursor.execute("FLUSH HOSTS")
		cursor.execute("FLUSH TABLES")
		cursor.execute("FLUSH STATUS")
		cursor.execute(query)
		return cursor

def estimated_total_table_record(table_name):
	query_db('analyze table ' + table_name)
	data = query_db('SELECT TABLE_ROWS FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = \'' + str(table_name) + '\'')
	result = data.fetchone()
	return int(result[0])

def query_db_with_blob(query,cat, blob):
	if cursor != None and cnx != None:
		cursor.execute(query, (cat,blob,))
		return cursor

def query_commit():
	if cursor != None and cnx != None:
		cnx.commit()

def close_db():
	if cursor != None and cnx != None:
		cursor.close()
		cnx.close()
