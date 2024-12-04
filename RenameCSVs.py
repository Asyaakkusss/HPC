import pandas as pd
import os


# Renames CSV files according to mapping of preterm outcome
def rename_files_from_csv(csv_file):
    df = pd.read_csv(csv_file)

    for index, row in df.iterrows():
        old_filename = row[0] + ".csv"
        new_filename = row[1] + old_filename

        try:
            os.rename("CSV2/" + old_filename, "CSV2/" + new_filename)
            print(f"Renamed {old_filename} to {new_filename}")
        except FileNotFoundError:
            print(f"File not found: {old_filename}")
        except OSError as e:
            print(f"Error renaming file: {e}")


csv_file = "tpehgdbMap.csv"
rename_files_from_csv(csv_file)
