import pandas as pd
import wfdb
path = "term-preterm-ehg-database-1.0.1"

record = wfdb.rdrecord(path + '/tpehgdb/tpehg546')
# wfdb.plot_wfdb(record=record)
print(record.__dict__['sig_name'])


# Turn Record into dataframe and CSV
def to_df(filename):
    record = wfdb.rdrecord(filename)
    labels = record.__dict__['sig_name']
    p_signal = record.__dict__['p_signal']
    df = pd.DataFrame(p_signal, columns=labels)
    return df


def to_csv(record, output_name):
    labels = record.__dict__['sig_name']
    p_signal = record.__dict__['p_signal']
    df = pd.DataFrame(p_signal, columns=labels)
    df.to_csv(output_name)


# Get Data Records
def read_records(file_path):
    with open(file_path, 'r') as file:
        records = file.read().splitlines()
    return records


file_path = 'term-preterm-ehg-database-1.0.1/RECORDS'

record_names = read_records(file_path)


def get_records(list_of_records, prefix):
    return_me = []
    for record in list_of_records:
        return_me.append(wfdb.rdrecord(prefix + '/' + record))
    return return_me


records = get_records(record_names, path)


# Turn into CSV
# Preterm into CSV
count = 0
for record in records:
    to_csv(record, "CSV2/" + record_names[count].split("/")[1] + ".csv")
    count += 1
