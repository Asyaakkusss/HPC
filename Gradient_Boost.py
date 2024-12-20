import xgboost as xgb
import pandas as pd
import os
print(xgb.__version__)
import numpy as np

def read_smr_file(file_path):
    records = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith(' Re') or line.startswith('-'):  # Skip the header line
                continue
            parts = line.split('|')
            record_name = parts[0].strip()
            premature = parts[4].strip() == 't'
            early = parts[5].strip() == 't'
            records[record_name] = premature
    return records

def is_premature(record_name, records):
    return records.get(record_name, False)

# Example usage
file_path = '/Users/jdanninger/Documents/GitHub/HPC/term-preterm-ehg-database-1.0.1/tpehgdb.smr'
records = read_smr_file(file_path)
record_name = 'tpehg1007'
premature = is_premature(record_name, records)
print(f'{record_name} is premature: {premature}')
# Load all the data into an array
csv2_path = "/Users/jdanninger/Documents/GitHub/HPC/CSV2"



def load_csvs_to_dict(csv2_path):
    dataframes = {}
    for filename in os.listdir(csv2_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(csv2_path, filename)
            df = pd.read_csv(file_path)
            # only keep columns "1" "2" and "3"

            # df = df[[ "1", "2", "3"]]


            # for cols ["1", "2", "3"] create new cols ["1 fft", "2 fft", "3 fft"] which are col 1 but with fft applied
            # df = df["1"]

            for col in ["1", "2", "3"]:
                df[col + " fft"] = np.fft.fft(df[col])
                # the FFT is a complex number, so we need to split it into real and imaginary parts
                df[col + " fft real"] = df[col + " fft"].apply(lambda x: x.real)
                # drop anything iwth imaginary part
                df = df.drop(columns=[col + " fft"])




            key = os.path.splitext(filename)[0]  # Use filename without extension as key
            dataframes[key] = df
    return dataframes

# Example usage
csv2_path = '/Users/jdanninger/Documents/GitHub/HPC/CSV2'
dataframes_dict = load_csvs_to_dict(csv2_path)

# Print the keys of the dictionary to verify
keys = dataframes_dict.keys()
print(keys)
print(dataframes_dict["tpehg567"].head())

# get outcomes which is a dictionary of every key with the value of whether or not the baby was premature
def get_outcomes(dataframes_dict, records):
    outcomes = {}
    for key in dataframes_dict.keys():
        premature = is_premature(key, records)
        outcomes[key] = premature
    return outcomes
outcomes = get_outcomes(dataframes_dict, records)
print(outcomes)







import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Combine dataframes into a single DataFrame
def combine_dataframes(dataframes_dict, outcomes):
    combined_df = pd.DataFrame()
    labels = []
    for key, df in dataframes_dict.items():
        if key in outcomes:
            df['record_name'] = key
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            labels.extend([outcomes[key]] * len(df))
    return combined_df, labels

# Prepare the data
combined_df, labels = combine_dataframes(dataframes_dict, outcomes)

# Split the data into features and target
X = combined_df.drop(columns=['record_name'])
y = labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the gradient boosting model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

from sklearn.metrics import roc_auc_score

# Make predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
auc = roc_auc_score(y_test, y_pred_proba)
print(f'AUC: {auc}')


