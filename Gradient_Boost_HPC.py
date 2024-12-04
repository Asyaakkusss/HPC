import xgboost as xgb
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print(xgb.__version__)


# Check if a record is premature
def is_premature(record_name):
    if record_name[0] == 't':
        return True
    return False


# Load all the data into an array
csv2_path = "CSV2"


def load_csvs_to_dict(csv2_path):
    dataframes = {}
    for filename in os.listdir(csv2_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(csv2_path, filename)
            df = pd.read_csv(file_path)
            # only keep the columns ["1", "2", "3"]
            df = df[["1"]]
            key = os.path.splitext(filename)[0]  # Use filename without extension as key
            dataframes[key] = df
    return dataframes


# Example usage
csv2_path = 'CSV2'
dataframes_dict = load_csvs_to_dict(csv2_path)

# Print the keys of the dictionary to verify
keys = dataframes_dict.keys()
print(keys)
print(dataframes_dict["tpehg567"].head())


# get outcomes which is a dictionary of every key with the value of whether or not the baby was premature
def get_outcomes(dataframes_dict):
    outcomes = {}
    for key in dataframes_dict.keys():
        premature = is_premature(key)
        outcomes[key] = premature
    return outcomes


outcomes = get_outcomes(dataframes_dict)
print(outcomes)


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the gradient boosting model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
