#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import pandas as pd
import numpy as np
import os


# %%


# Change SGU xlsx month here
SGU_Raw = "G:/Works/!!Intern/DATA SGU SI/SGU Data/JR5_DATA RAW/JR5M01Y2021_DATA RAW.xlsx"
SGU_sheet_name = "RX3502_JR5 "


# %%


SGU_raw = pd.read_excel(SGU_Raw, sheet_name=SGU_sheet_name)
string_columns = ['NG', 'DP', 'DB', 'BP', 'BP2', 'Converted BP', 'ST', 'NOTK', 'NOIR', 'S', 'NP', 'PKIS', 'HMIS', 'J', 'KET', 'B']


# %%


df01 = pd.read_excel(SGU_Raw, sheet_name=SGU_sheet_name, 
                     dtype={col: str for col in string_columns})


# %%


# remove whitespace 
df01.columns = df01.columns.str.strip()

# replace space with underscore
df01 = df01.rename(columns=lambda x: x.replace(" ", "_"))

# uppercasing column headers
df01.columns = df01.columns.str.upper()


# %%


def check_column_lengths(df01, columns, required_lengths):
    for col, req_len in zip(columns, required_lengths):
        df01[col] = df01[col].astype(str).str.zfill(req_len)
        df01[col] = df01[col].str[:req_len]
    
    return df01

columns = ['NG', 'DP', 'DB', 'BP', 'BP2', 'CONVERTED_BP', 'ST', 'NOTK', 'NOIR', 'S', 'NP', 'PKIS', 'HMIS', 'J', 'KET', 'B']
required_lengths = [2, 2, 3, 3, 3, 3, 1, 4, 2, 1, 3, 2, 2, 1, 4, 2]

df01 = check_column_lengths(df01, columns, required_lengths)


# %%


columns_to_convert = ['NG', 'DP', 'DB', 'BP', 'BP2', 'CONVERTED_BP', 'ST', 'NOTK', 'NOIR', 'S', 'NP', 'PKIS', 'HMIS', 'J', 'KET', 'B']
df01['ID_38'] = df01.loc[:, columns_to_convert].astype(str).apply(''.join, axis=1)
df01['ID_38'] = df01['ID_38'].astype(str)


# %%


df_sorted = df01.sort_values(by='ID_38')
unique_A = df_sorted['ID_38'].drop_duplicates()
pd.set_option('display.max_rows', None)

counts = df_sorted['ID_38'].value_counts().sort_index()
result = pd.DataFrame({'ID_38': counts.index, 'Count': counts.values})
print(result.to_string(index=False))


# %%


df01.dtypes


# %%


# Change STB xlsx month here
STB_Raw = "G:/Works/!!Intern/DATA SGU SI/STB Data/JR42021BULANAN_FINAL_CSV/dsB012021STB.xlsx"
STB_sheet_name = "Data"


# %%


STB_raw = pd.read_excel(STB_Raw, sheet_name=STB_sheet_name)
string_columns = ['NG', 'DP', 'DB', 'BP', 'BP2', 'CONVERTEDBP', 'ST', 'NOTK', 'NOIR', 'S', 'NP', 'PKIS', 'HMIS', 'J', 'KET', 'B']


# %%


df02 = pd.read_excel(STB_Raw, sheet_name=STB_sheet_name,
                   dtype={col: str for col in string_columns})


# %%


# remove whitespace 
df02.columns = df02.columns.str.strip()

# replace space with underscore
df02 = df02.rename(columns=lambda x: x.replace(" ", "_"))

# uppercasing column headers
df02.columns = df02.columns.str.upper()


# %%


def check_column_lengths(df02, columns, required_lengths):
    for col, req_len in zip(columns, required_lengths):
        df02[col] = df02[col].astype(str).str.zfill(req_len)
        df02[col] = df02[col].str[:req_len]
    
    return df02

columns = ['NG', 'DP', 'DB', 'BP', 'BP2', 'CONVERTEDBP', 'ST', 'NOTK', 'NOIR', 'S', 'NP', 'PKIS', 'HMIS', 'J', 'KET', 'B']
required_lengths = [2, 2, 3, 3, 3, 3, 1, 4, 2, 1, 3, 2, 2, 1, 4, 2]

df02 = check_column_lengths(df02, columns, required_lengths)


# %%


df_sorted = df02.sort_values(by='NP')
unique_A = df_sorted['NP'].drop_duplicates()
pd.set_option('display.max_rows', None)

counts = df_sorted['NP'].value_counts().sort_index()
result = pd.DataFrame({'NP': counts.index, 'Count': counts.values})
print(result.to_string(index=False))


# %%


columns_to_convert = ['NG', 'DP', 'DB', 'BP', 'BP2', 'CONVERTEDBP', 'ST', 'NOTK', 'NOIR', 'S', 'NP', 'PKIS', 'HMIS', 'J', 'KET', 'B']
df02['ID_38'] = df02.loc[:, columns_to_convert].astype(str).apply(''.join, axis=1)
df02['ID_38'] = df02['ID_38'].astype(str)


# %%


df_sorted = df02.sort_values(by='ID_38')
unique_A = df_sorted['ID_38'].drop_duplicates()
pd.set_option('display.max_rows', None)

counts = df_sorted['ID_38'].value_counts().sort_index()
result = pd.DataFrame({'ID_38': counts.index, 'Count': counts.values})
print(result.to_string(index=False))


# %%


df01['ID_38'] = df01['ID_38'].astype(str)
df02['ID_38'] = df02['ID_38'].astype(str)


# %%


df01['NAMA'] = df01['NAMA'].str.upper()
df02['NAMA'] = df02['NAMA'].str.upper()


# %%


df01 = df01.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df02 = df02.applymap(lambda x: x.strip() if isinstance(x, str) else x)


# %%


for column in df01.columns:
    print(column)


# %%


for column in df02.columns:
    print(column)


# %%


df_sorted = df01.sort_values(by='NAMA')
unique_A = df_sorted['NAMA'].drop_duplicates()
pd.set_option('display.max_rows', None)

counts = df_sorted['NAMA'].value_counts().sort_index()
result = pd.DataFrame({'NAMA': counts.index, 'Count': counts.values})
print(result.to_string(index=False))


# %%


df_sorted = df02.sort_values(by='NAMA')
unique_A = df_sorted['NAMA'].drop_duplicates()
pd.set_option('display.max_rows', None)

counts = df_sorted['NAMA'].value_counts().sort_index()
result = pd.DataFrame({'NAMA': counts.index, 'Count': counts.values})
print(result.to_string(index=False))


# %%


df_sorted = df02.sort_values(by='PEMBERAT_FINAL')
unique_A = df_sorted['PEMBERAT_FINAL'].drop_duplicates()
pd.set_option('display.max_rows', None)

counts = df_sorted['PEMBERAT_FINAL'].value_counts().sort_index()
result = pd.DataFrame({'PEMBERAT_FINAL': counts.index, 'Count': counts.values})
print(result.to_string(index=False))


# %%


# Iterate over each row in 'df01'
for index, row in df01.iterrows():
    # Get the values from columns in 'df01'
    idnum = row['ID_38']
    no_id = row['NAMA']

    # Find the corresponding row(s) in 'df02' based on the matching values in 'NAMA' column
    replacement_rows = df02.loc[df02['ID_38'] == idnum]

    # Check if any matching row(s) were found in 'df02' based on 'NAMA' column
    if not replacement_rows.empty:
        # Get the values from columns in 'df02'
        pemberat_final = replacement_rows.iloc[0]['PEMBERAT_FINAL']

        # Update the values in 'df01' with the matching values from 'df02'
        df01.at[index, 'PEMBERAT_FINAL'] = pemberat_final

        print(f"Matching ID found for row {index}")
    else:
        # Find the corresponding row(s) in 'df02' based on the matching values in 'NO ID' column
        replacement_rows = df02.loc[df02['NAMA'] == no_id]

        # Check if any matching row(s) were found in 'df02' based on 'NO ID' column
        if not replacement_rows.empty:
            # Get the values from columns in 'df02'
            pemberat_final = replacement_rows.iloc[0]['PEMBERAT_FINAL']

            # Update the values in 'df01' with the matching values from 'df02'
            df01.at[index, 'PEMBERAT_FINAL'] = pemberat_final

            print(f"Matching ID found based on 'NAMA' for row {index}")
        else:
            print(f"No matching ID found for row {index}")


# %%


duplicates = df01[df01.duplicated('ID_38')]

if duplicates.empty:
    print("No duplicates found.")
else:
    print("Duplicates found:")
    print(duplicates)


# %%


output_folder = "G:/Works/!!Intern/DATA SGU SI/SGU Data/SGU Output/Bulanan/"
filename = os.path.basename(SGU_Raw)
file_naming = filename[:16]
output_file = f"{file_naming}_FC.csv"
df01.to_csv(os.path.join(output_folder, output_file), index=False)


# %%




