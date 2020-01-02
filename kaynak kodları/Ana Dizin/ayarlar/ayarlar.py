db_host = "192.168.63.6"
db_user = "pythonuser"
db_password = "pass1234"
db_name = "ids_db"

dm_host = "192.168.63.7"
dm_host_web_port = 8081
dm_host_datamining_shell_port = 8080

raw_data_table_name = "table_kdd99_all"
raw_data_class_column = "class"
raw_data_primarykey_column = "id"
outliers_columns = ("duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
			"su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
			"srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
			"dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
			"dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate")

sample_data_table_name = "table_kdd99_sample"
sample_data_primarykey_column = "id"

expert_data_table_name = "table_expert"
expert_data_primarykey_column = "id"
expert_prediction_column_name = "prediction"

kdd10_data_table_name = "table_kdd99_10percent"
nsl_train_data_table_name = "table_nsl_train"
nsl_test_data_table_name = "table_nsl_test"


test_data_table_name = "table_kdd99_test"
test_data_primarykey_column = "id"

mining_columns = "duration,protocol_type,service,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,num_failed_logins,logged_in,num_compromised,root_shell,su_attempted,num_root,num_file_creations,num_shells,num_access_files,num_outbound_cmds,is_host_login,is_guest_login,count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate"

mining_class = "class"
prediction_column_name = "prediction"

nsl_update_traffic_table_name = "table_nsl_update"

class_feature_relation_columns = "duration,protocol_type,service,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,num_failed_logins,logged_in,num_compromised,root_shell,su_attempted,num_root,num_file_creations,num_shells,num_access_files,num_outbound_cmds,is_host_login,is_guest_login,count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate"
discretization_columns = "duration,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,num_failed_logins,logged_in,num_compromised,root_shell,su_attempted,num_root,num_file_creations,num_shells,num_access_files,num_outbound_cmds,is_host_login,is_guest_login,count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate"

discretization_max_split = 10

discretization_table_name = "table_discretization"
discretization_table_parameter_column = "parameter"
discretization_table_value_column = "value"

configs_table_name = "table_configs"
configs_table_parameter_column = "parameter"
configs_table_value_column = "value"
configs_table_value_column_default = "None"

configs_sample_status_parameter = "sample_status"
configs_sample_status_values = ["None", "Created", "Changing", "Deleted", "Changed"]

configs_test_status_parameter = "test_status"
configs_test_status_values = ["None", "Created", "Changing", "Deleted", "Changed"]

configs_data_format_status_parameter = "data_format_status"
configs_data_format_status_values = ["None", "Raw", "Transformed"]

configs_discretization_status_parameter = "discretization_status"
configs_discretization_status_values = ["None", "Raw", "Transformed"]

configs_algorithm_status_parameter = "algorithms_status"
configs_algorithm_status_values = ["None", "Id3", "Cart","All"]

configs_nsl_algorithm_status_parameter = "nsl_algorithms_status"
configs_nsl_algorithm_status_values = ["None" ,"All"]

configs_operations = ["sample", "test", "traffic"]

unknown_class_value = "bilinmeyen_saldiri"
unknown_class_name = "Bilinmeyen Saldırı"
classes_values = [ "normal","buffer_overflow","loadmodule","perl","neptune","smurf","guess_passwd","pod","teardrop","portsweep","ipsweep","land","ftp_write","back","imap","satan","phf",
			"nmap","multihop","warezmaster","warezclient","spy","rootkit"]
classes_names = ["Normal Trafik","Buffer Overflow Saldırısı","Loadmodule Saldırısı","Perl Saldırısı","Neptune Saldırısı","Smurf Saldırısı","Guess Passwd Saldırısı","Pod Saldırısı",
			"Teardrop Saldırısı","Portsweep Saldırısı","Ipsweep Saldırısı","Land Saldırısı","Ftp Write Saldırısı","Back Saldırısı","Imap Saldırısı","Satan Saldırısı",
			"Phf Saldırısı","Nmap Saldırısı","Multihop Saldırısı","Warezmaster Saldırısı","Warezclient Saldırısı","Spy Saldırısı","Rootkit Saldırısı"]

class_01_name = "Normal trafik"
class01_value = "normal"
class_02_name = "Bilinen ataklar"
class02_value = "buffer_overflow"

columns_data_transformation = "protocol_type,service,flag"

database_update_index_name = "update_index"

id3_tree_file_path = 'araclar/agaclar/sma_agac_id3_tree.sma'
cart_tree_file_path = 'araclar/agaclar/sma_agac_cart_tree.sma'
nsl_id3_tree_file_path = 'araclar/agaclar/sma_agac_nsl_id3_tree.sma'
nsl_cart_tree_file_path = 'araclar/agaclar/sma_agac_nsl_cart_tree.sma'

web_database_show_table_name = expert_data_table_name

traffic_table_name = "table_real_traffic"
traffic_table_primary_key = "id"
traffic_table_columns = "first_connection_time,source_port,destination_port,source_host,destination_host,duration,protocol_type,service,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,num_failed_logins,logged_in,num_compromised,root_shell,su_attempted,num_root,num_file_creations,num_shells,num_access_files,num_outbound_cmds,is_host_login,is_guest_login,count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate"
traffic_table_cti_columns = "source_host, source_port,destination_host,destination_port"
listener_user = "root"
listener_host = "192.168.63.3"
listener_created_pcap_file_path = "/home/ubuntulistener/traffic/internet_traffics.pcap"
listener_realtime_pcap_file_path = "/home/ubuntulistener/traffic/realtime_traffic.pcap"
listener_bro_policy_file_path = "/home/ubuntulistener/policies/darpa2gurekddcup.bro"
listener_conn_list_file_path = "/home/ubuntulistener/traffic/conn.list"
listener_conn_sorted_list_file_path = "/home/ubuntulistener/traffic/conn_sort.list"
listener_traffic_analyzer_binary_file_path = "/home/ubuntulistener/trafAld.out"
listener_traffic_analyzer_output_file_path = "/root/trafAld.list"
dm_traffic_infos_file_path = "/home/ubuntudm/traffic_infos"
dm_traffic_infos_edited_file_path = "/home/ubuntudm/traffic_infos_edited"

shell_username = "suleyman"
shell_password = "arikan"

protocols = "icmp,tcp,udp"
protocols_list = protocols.split(",")

services_list = ["other", "private", "tcpmux", "echo", "discard", "systat", "daytime", "netstat", "qotd", "msp", "chargen", "ftp-data", "ftp", "fsp", "ssh", "telnet", "smtp", "time", "rlp", "nameserver", "whois", "tacacs", "re-mail-ck", "domain", "mtp", "tacacs-ds", "bootps", "bootpc", "tftp", "gopher", "rje", "finger", "http", "link", "kerberos", "supdup", "hostnames", "iso-tsap", "acr-nema", "csnet-ns", "rtelnet", "pop2", "pop3", "sunrpc", "auth", "sftp", "uucp-path", "nntp", "ntp", "pwdgen", "loc-srv", "netbios-ns", "netbios-dgm", "netbios-ssn", "imap2", "snmp", "snmp-trap", "cmip-man", "cmip-agent", "mailq", "xdmcp", "nextstep", "bgp", "prospero", "irc", "smux", "at-rtmp", "at-nbp", "at-echo", "at-zis", "qmtp", "z3950", "ipx", "imap3", "pawserv", "zserv", "fatserv", "rpc2portmap", "codaauth2", "clearcase", "ulistserv", "ldap", "imsp", "svrloc", "https", "snpp", "microsoft-ds", "kpasswd", "urd", "saft", "isakmp", "rtsp", "nqs", "npmp-local", "npmp-gui", "hmmp-ind", "asf-rmcp", "qmqp", "ipp", "exec", "biff", "login", "who", "shell", "syslog", "printer", "talk", "ntalk", "route", "timed", "tempo", "courier", "conference", "netnews", "netwall", "gdomap", "uucp", "klogin", "kshell", "dhcpv6-client", "dhcpv6-server", "afpovertcp", "idfp", "remotefs", "nntps", "submission", "ldaps", "tinc", "silc", "kerberos-adm", "webster", "rsync", "ftps-data", "ftps", "telnets", "imaps", "ircs", "pop3s", "socks", "proofd", "rootd", "openvpn", "rmiregistry", "kazaa", "nessus", "lotusnote", "ms-sql-s", "ms-sql-m", "ingreslock", "prospero-np", "datametrics", "sa-msg-port", "kermit", "groupwise", "l2f", "radius", "radius-acct", "msnp", "unix-status", "log-server", "remoteping", "cisco-sccp", "search", "pipe-server", "nfs", "gnunet", "rtcm-sc104", "gsigatekeeper", "gris", "cvspserver", "venus", "venus-se", "codasrv", "codasrv-se", "mon", "dict", "f5-globalsite", "gsiftp", "gpsd", "gds-db", "icpv2", "iscsi-target", "mysql", "nut", "distcc", "daap", "svn", "suucp", "sysrqd", "sieve", "epmd", "remctl", "f5-iquery", "ipsec-nat-t", "iax", "mtn", "radmin-port", "rfe", "mmcc", "sip", "sip-tls", "aol", "xmpp-client", "xmpp-server", "cfengine", "mdns", "postgresql", "freeciv", "amqps", "amqp", "ggz", "x11", "x11-1", "x11-2", "x11-3", "x11-4", "x11-5", "x11-6", "x11-7", "gnutella-svc", "gnutella-rtr", "sge-qmaster", "sge-execd", "mysql-proxy", "afs3-fileserver", "afs3-callback", "afs3-prserver", "afs3-vlserver", "afs3-kaserver", "afs3-volser", "afs3-errors", "afs3-bos", "afs3-update", "afs3-rmtsys", "font-service", "http-alt", "bacula-dir", "bacula-fd", "bacula-sd", "xmms2", "nbd", "zabbix-agent", "zabbix-trapper", "amanda", "dicom", "hkp", "bprd", "bpdbm", "bpjava-msvc", "vnetd", "bpcd", "vopied", "db-lsp", "dcap", "gsidcap", "wnn6", "kerberos4", "kerberos-master", "passwd-server", "krb-prop", "krbupdate", "swat", "kpop", "knetd", "zephyr-srv", "zephyr-clt", "zephyr-hm", "eklogin", "kx", "iprop", "supfilesrv", "supfiledbg", "linuxconf", "poppassd", "moira-db", "moira-update", "moira-ureg", "spamd", "omirr", "customs", "skkserv", "predict", "rmtcfg", "wipld", "xtel", "xtelw", "support", "cfinger", "frox", "ninstall", "zebrasrv", "zebra", "ripd", "ripngd", "ospfd", "bgpd", "ospf6d", "ospfapi", "isisd", "afbackup", "afmbackup", "xtell", "fax", "hylafax", "distmp3", "munin", "enbd-cstatd", "enbd-sstatd", "pcrd", "noclog", "hostmon", "rplay", "nrpe", "nsca", "mrtd", "bgpsim", "canna", "syslog-tls", "sane-port", "ircd", "zope-ftp", "tproxy", "omniorb", "clc-build-daemon", "xinetd", "mandelspawn", "git", "zope", "webmin", "kamanda", "amandaidx", "amidxtape", "smsqp", "xpilot", "sgi-cmsd", "sgi-crsd", "sgi-gcd", "sgi-cad", "isdnlog", "vboxd", "binkp", "asp", "csync2", "dircproxy", "tfido", "fido"]

#services = "http,smtp,domain_u,auth,finger,telnet,eco_i,ftp,ntp_u,ecr_i,other,urp_i,private,pop_3,ftp_data,netstat,daytime,ssh,echo,time,name,whois,domain,mtp,gopher,remote_job,rje,ctf,supdup,link,systat,discard,X11,shell,login,imap4,nntp,uucp,pm_dump,IRC,Z39_50,netbios_dgm,ldap,sunrpc,courier,exec,bgp,csnet_ns,http_443,klogin,printer,netbios_ssn,pop_2,nnsp,efs,hostnames,uucp_path,sql_net,vmnet,iso_tsap,netbios_ns,kshell,urh_i,http_2784,harvest,aol,tftp_u,http_8001,tim_i,red_i"
#unknown_services = "eco_i,ecr_i,urp_i,remote_job,ctf,pm_dump,nnsp,vmnet,urh_i,harvest,tim_i,red_i"
#change_services = "domain_u, ntp_u,pop_3,name,imap4,Z39_50,netbios_dgm,csnet_ns,ftp_data,http_443,netbios_ssn,netbios_ns,pop_2,efs,uucp_path,sql_net,iso_tsap,http_2784:www-dev/2784,tftp_u,http_8001:vcom-tunnel/8001"

flags = "OTH,REJ,RSTO,RSTOS0,RSTR,RSTRH,S0,S1,S2,S3,SF,SH,SHR"
flags_list = flags.split(",")

def transform_to_dummy_data_process(column_name, value, line):
	if column_name == "first_connection_time":
		return datetime.datetime.fromtimestamp(int(value)).strftime('%Y-%m-%d %H:%M:%S')
	if column_name == "protocol_type":
		return protocols_list.index(value) + 1
	elif column_name == "service":
		try:
			return services_list.index(value)
		except:
			return 0 # if you dont find it in list, send index of "other"
	elif column_name == "flag":
		try:
			return flags_list.index(value) + 1
		except:
			return 0 #unknown flag
	return -1
