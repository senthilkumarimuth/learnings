import glob
root_dir =r"C:\\Users\\senthil.marimuthu\\Documents\\tvs\\vehicle_health_prediction\\datasets\\iqube_health"
# root_dir needs a trailing slash (i.e. /root/dir/)
for filename in glob.iglob(root_dir + '**\*.parquet', recursive=True):
     print(filename)