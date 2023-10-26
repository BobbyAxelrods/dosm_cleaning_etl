#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import numpy as np
import os
import glob
import time 
import traceback 

import sys 
import warnings
import numpy as np
import pyreadstat
import shutil
import openpyxl
from sqlalchemy import create_engine, text
import psycopg2

pd.set_option('display.max_columns', None)


# In[2]:


# PLEASE RUN THIS QUERY TO SIMULATE UPDATE IN TABLE BEFORE RUNNING SCRIPT 
# INSERT INTO reference_data."USER_INPUT" ("year") VALUES(2021);


# In[3]:


#Query to get user input from UI in db input 
# Schema name and condition values
db_params = {
    'host': '### INSERT IP HERE',
    'database': '### INSERT',
    'user': '### INSERT',
    'password': '### INSERT'
}

schema_name = 'reference_data'
table_name = 'USER_INPUT'
# Connect to the database
conn = psycopg2.connect(**db_params)
cursor = conn.cursor()

# cursor.execute((f'SELECT * from {schema_name}."{table_name}" ORDER BY timestamp_column ')
cursor.execute(f'''
    SELECT *
    FROM (
        SELECT *, ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS row_num
        FROM {schema_name}."{table_name}"
    ) AS numbered
''')
rows= cursor.fetchall()

columns = [x[0] for x in cursor.description]
df = pd.DataFrame(rows,columns=columns)
# Close the cursor and the connection
cursor.close()
conn.close()
               
#add function to truncate table each time new data coming in 


# In[4]:


#Get the latest user input data numbered by max row 
max_row_num = df['row_num'].max()
max_row = df[df['row_num'] == max_row_num]
#MODULE 0 : QUERY SQL TABLE BASED ON USER INPUT VALUE FROM FRONT END 
# QUARTER = max_row['quarter'].values[0]
YEAR = max_row['year'].values[0]


# In[5]:


print(f"We are cleaning STB_YEAR_{YEAR} Survey Data")


# In[6]:


#Query table that contain quarter and years selectfion 

import psycopg2

def get_tables_with_quarter_year(cursor, schema_name, year):
    # Get a list of all table names in the specified schema
    cursor.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema_name}'")
    table_names = [row[0] for row in cursor.fetchall()]

    # List to store the tables that meet the condition
    result_tables = []

    for table_name in table_names:
        # Check if the table has the specified columns 'quarter' and 'year'
        cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_schema = '{schema_name}' AND table_name = '{table_name}'")
        column_names = [row[0] for row in cursor.fetchall()]

        if 'QUARTER' in column_names and 'YEARS' in column_names:
            # Execute a query to check if the table has any records with quarter=1 and year=2021
            cursor.execute(f'''SELECT 1 FROM {schema_name}."{table_name}" WHERE "YEARS" = {YEAR} LIMIT 1''')
            if cursor.fetchone() is not None:
                result_tables.append(table_name)

    return result_tables


# In[7]:


# engine_specific = create_engine('postgresql+psycopg2://admin:admin@10.251.49.51:5432/postgres',connect_args={'options':'-csearch_path={}'.format('production_indicator_viz')})


# Database connection parameters
db_params = {
    'host': '###INSERT IP HERE',
    'database': '### INSERT ',
    'user': '### INSERT',
    'password': '### INSERT'
}

# Schema name and condition values
schema_name = 'production_micro_final_stb_monthly'


# Connect to the database
conn = psycopg2.connect(**db_params)
cursor = conn.cursor()

# Get the tables that meet the condition
result_tables = get_tables_with_quarter_year(cursor, schema_name, YEAR)

# Close the cursor and the connection
cursor.close()
conn.close()

# Print the result
print(f"Tables with {YEAR}:")
print(result_tables)


# In[8]:


# Store data in dicitonary for wrangling 

schema_name = 'production_micro_final_stb_monthly'

# Connect to the database
conn = psycopg2.connect(**db_params)
cursor = conn.cursor()

#store all dataframe in after loop in a dictionary 
dataframe_dict = {}

#From result table , query table name and store the value in DF 
for table_name in result_tables:
    query = f'''SELECT * FROM {schema_name}."{table_name}"'''
    cursor.execute(query)
    data = cursor.fetchall()
    columns = [y[0] for y in cursor.description]
    df = pd.DataFrame(data,columns=columns)
    
    dataframe_dict[table_name] = df
    
    print(f'{table_name} has been stored in dictionary for quality check')
    
# Close the cursor and the connection
cursor.close()
conn.close()


# In[9]:


#Standardization 

# Create a new dictionary to store the updated DataFrames
updated_dataframe_dict = {}

# Loop through the tables in result_tables
for x in result_tables:
    # Update columns of the DataFrame and store it in the new dictionary
    updated_dataframe = dataframe_dict[x].copy()  # Create a copy of the original DataFrame
    updated_dataframe.columns = updated_dataframe.columns.str.upper().str.replace(r'[\W_]+', '')
    
    # Store the updated DataFrame in the new dictionary
    updated_dataframe_dict[x] = updated_dataframe
    
    # Print a message
    print(f'Columns for table {x} have been standardized to non-whitespace, no special characters, and all uppercase.')

# Now, updated_dataframe_dict contains the updated DataFrames



# OLD METHOD 

# for x in result_tables:
#     dataframe_dict[x].columns = dataframe_dict[x].columns.str.upper().str.replace(r'[\W_]+','')
#     print(f'Column for table {x} has been standardize to non-whitespace, no special charac & all uppercase ')


# In[10]:


column_size= []
rowtotal = 0
for table_name in result_tables:
    if table_name in dataframe_dict:
        table_shape = dataframe_dict[table_name].shape
        column_size.append(table_shape[1])
        rowtotal += table_shape[0] 
        print(f'Shape of {table_name}: {table_shape} where the ')
    else:
        print(f'{table_name} not found in the dataframe_dict.')
print(f'The total rows after merging all data from all months for year {YEAR} are : {rowtotal} rows')


# In[11]:


get_max_size = max(column_size)
print(f'From multi dataframe , the biggest column is {get_max_size} ')


# In[12]:


#merging vertically 3 dataframe into 1 dataframe 
concatenated_dataframe = pd.concat(updated_dataframe_dict.values(), ignore_index=True)
print(f'Multi final data for year {YEAR} have been merged into 1 data frame ')


# In[13]:


table_shape_new = concatenated_dataframe.shape
col_diff = table_shape_new[1] - get_max_size
print(f'Shape of merged dataframe: {table_shape_new} with column size of {table_shape_new[1]} and have {col_diff} differences between 3 dataframe  ')


# In[14]:


#we can make this dynamic, but require some times to make it dynamics
df1 = updated_dataframe_dict[result_tables[0]]
df2 = updated_dataframe_dict[result_tables[1]]
df3 = updated_dataframe_dict[result_tables[2]]
df4 = updated_dataframe_dict[result_tables[3]]
df5 = updated_dataframe_dict[result_tables[4]]
df6 = updated_dataframe_dict[result_tables[5]]
df7 = updated_dataframe_dict[result_tables[6]]
df8 = updated_dataframe_dict[result_tables[7]]
df9 = updated_dataframe_dict[result_tables[8]]
df10 = updated_dataframe_dict[result_tables[9]]
df11 = updated_dataframe_dict[result_tables[10]]
df12 = updated_dataframe_dict[result_tables[11]]


# In[15]:


all_columns = pd.concat([df1, df2, df3], axis=1).columns
unique_columns = all_columns[~all_columns.duplicated()]


# In[16]:


non_intersecting_columns = unique_columns[~unique_columns.isin(df1.columns) | ~unique_columns.isin(df2.columns) | ~unique_columns.isin(df3.columns)]
view = pd.DataFrame(non_intersecting_columns)
print("Columns that are not common/no intersection across multi final data : ", view)


# In[17]:


df_jr4 = concatenated_dataframe


# In[18]:


# MODULE 1 : BASIC PREPARATION


# In[19]:


#read whatever files types being ingested 
def read_anything(path):
    #Get all files avaialble in the path (path shall be only 1 file at a time to manage this )
    get_working_files = [x for x in os.listdir(path)]
#     get_working_files = x
    #make this as a function later 
    if len(get_working_files) == 0:
        print('My goodman, no files either csv nor excel found, please recheck in path the existence')
        return None
    
    #Excel found 
    time_start = time.time()
    get_types_available = os.path.splitext(get_working_files[0])[1]
    file_name = os.path.splitext(get_working_files[0])[0]
    if get_types_available.endswith('.xlsx'):
        
        time_start = time.time()
        print('We found excel files, hence we will read it and save to df_master, hold a moment ....')
        df_master = pd.read_excel(path+'/'+get_working_files[0],dtype=str).dropna(how='all')
        int_columns = []
        for col in df_master.columns:
            if df_master[col].notnull().all() and df_master[col].str.isdigit().all():
                int_columns.append(col)
        
        df_master[int_columns] = df_master[int_columns].astype(int)
        time_end = time.time()
        
        diff_time = time_end - time_start
        print(f'My performance reading {file_name} file took : {diff_time} seconds')
        return df_master
    
    elif get_types_available.endswith('.csv'):
        print('We found csv files, hence we will read it and save to df_master, hold a moment ....')
        time_start = time.time()  
        try:
            df_master = pd.read_csv(os.path.join(path, get_working_files[0]), skip_blank_lines=True,dtype=str).dropna(how='all')
        except UnicodeDecodeError:
            # If 'utf-8' fails, try 'ISO-8859-1' encoding
            df_master = pd.read_csv(os.path.join(path, get_working_files[0]), encoding='ISO-8859-1', skip_blank_lines=True,dtype=str).dropna(how='all')
        int_columns = []
        for col in df_master.columns:
            if df_master[col].notnull().all() and df_master[col].str.isdigit().all():
                int_columns.append(col)
        df_master[int_columns] = df_master[int_columns].astype(int)
        time_end = time.time()
        diff_time = time_end - time_start
        print(f'My performance reading {file_name} file took : {diff_time} seconds')
        return df_master


# In[20]:


start_time = time.time()
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',100)
warnings.filterwarnings('ignore')
start_time = time.time()
current_workingpath = os.getcwd()


# #### PATH DECLARATION 


# In[21]:


#### INPUT FOR HDFS

t01_path = os.path.join(current_workingpath,'INPUT_T01')
path_t02 = os.path.join(current_workingpath,'INPUT_T02')
temp_path = os.path.join(current_workingpath,'INPUT_MAP_RINSTRATA')
bppd_storage_path = os.path.join(current_workingpath,'INPUT_T03')
bppd_database = os.path.join(current_workingpath,'OUTPUT_QUALITYCHECK_8')
pop_fac_check = os.path.join(current_workingpath,'OUTPUT_QUALITYCHECK_7')
popfac_path = os.path.join(current_workingpath,'OUTPUT_QUALITYCHECK_7')
bin_path = os.path.join(current_workingpath,'BIN')

#### INPUT FOR NIFI SERVER ONLY 

# t01_path = '/home/hadoop/codes/prod_stb_monthly/INPUT_T01'
# path_t02 = '/home/hadoop/codes/prod_stb_monthly/INPUT_T02'
# temp_path = '/home/hadoop/codes/prod_stb_monthly/INPUT_RAWDATA_STB_JR4'
# jr4_raw_path = '/home/hadoop/codes/prod_stb_monthly/INPUT_RAWDATA_STB_JR2'
# jr2_raw_path = '/home/hadoop/codes/prod_stb_monthly/INPUT_T02'
# bppd_storage_path = '/home/hadoop/codes/prod_stb_monthly/INPUT_T03'
# bppd_database = '/home/hadoop/codes/prod_stb_monthly/OUTPUT_QUALITYCHECK_8'
# pop_fac_check = '/home/hadoop/codes/prod_stb_monthly/OUTPUT_QUALITYCHECK_7'
# popfac_path ='/home/hadoop/codes/prod_stb_monthly/OUTPUT_QUALITYCHECK_7'
# bin_path = '/home/hadoop/codes/prod_stb_monthly/BIN'


# In[22]:


#Check if digit required is not enough, to add leading 0 to ensure no id 38 is complete
def check_column_lengths(df01, columns, required_lengths):
    for col, req_len in zip(columns, required_lengths):
        df01[col] = df01[col].astype(str).str.zfill(req_len)
        df01[col] = df01[col].str[:req_len]
    print('Value in columns specified has been added with leading 0 to ensure 38 digit')
    return df01


# In[23]:


# Concatting value from selected column to generate NOID 38 

def generate_new_noid(df,columns,noid_col_name):
    df[noid_col_name] = df.loc[:,columns].astype(str).apply(''.join, axis=1)
    print(f'{noid_col_name} has been generated by merging values from specified columns')
    return df 


# In[24]:


#check all value in columns shall be 38 (or specified by user)

def check_digit_match(df,noid_col_name):
    digit_generated = df[noid_col_name].apply(lambda x: len(str(x)))
    counts_digit_unique = digit_generated.value_counts()
    if len(counts_digit_unique) > 1: 
        print(f"Recheck due to inconsistent digit in NOID {counts_digit_unique}")
    else: 
        counts_digit = digit_generated.unique()[0]
        print(f'{counts_digit} consistent digits has been generated , does this tally with client requirement? ')
        


# In[25]:


## Manipulate Here 
columns_sel = ['NG', 'DP', 'DB', 'BP', 'BP2', 'CONVERTEDBP', 'ST', 'NOTK', 'NOIR', 'S', 'NP', 'PKIS', 'HMIS', 'J', 'KET', 'B']
required_lengths = [2, 2, 3, 3, 3, 3, 1, 4, 2, 1, 3, 2, 2, 1, 4, 2]
df_jr4_new = check_column_lengths(df_jr4, columns_sel, required_lengths)
noid_col_name = 'NOID_38'
required_digits = 38


# In[26]:


df_jr4_new_2 = generate_new_noid(df_jr4_new,columns_sel,noid_col_name)


# In[27]:


check_digit_match(df_jr4_new_2,noid_col_name)


# ## MODULE 2 : GROUPING ACCORDING TO CLIENT REQUIREMENT

# 	- GROUP BY DALAM UMUR 5 TAHUN 
# 	- GROUP BY DALAM UMUR 10 TAHUN 
# 	- GROUP BY DALAM ETNIK SEMENANJUNG 
# 	- GROUP BY ETNIK SABAH (BUMIPUTRA SABAH)
# 	- GROUP BY ETNIK SARAWAK 
# 	- GROUP BY CIT_NONCIT 
# 	- GROUP BY RIN_STRATA 


# In[28]:


df_jr4 = df_jr4_new_2


# In[29]:


nan_indices = df_jr4.index[df_jr4['U'].isnull()]


# In[30]:


# #make columns upfronts detect dataypes and fill first with nan to ensure complete fillup
# df_jr4['G1'] = np.nan
# df_jr4['KU_5'] = np.nan
# df_jr4['G2'] = np.nan
# df_jr4['G3'] = np.nan
# df_jr4['G4'] = np.nan
# df_jr4['G5'] = np.nan
# df_jr4['CIT_NONCIT'] = np.nan
# df_jr4['RIN_STRATA'] = np.nan

# # df_jr4['U'] = pd.to_numeric(df_jr4['U'], errors='coerce')
# # df_jr4['KET'] = pd.to_numeric(df_jr4['KET'], errors='coerce')
# # df_jr4['KW'] = pd.to_numeric(df_jr4['KW'], errors='coerce')
# # df_jr4['ST'] = pd.to_numeric(df_jr4['ST'], errors='coerce')


# df_jr4['U'] =df_jr4['U'].astype(int)
# df_jr4['KET'] =df_jr4['KET'].astype(int)
# df_jr4['KW'] =df_jr4['KW'].astype(int)
# df_jr4['ST'] = df_jr4['ST'].astype(int) 

#make columns upfronts detect dataypes and fill first with nan to ensure complete fillup
df_jr4['G1'] = np.nan
df_jr4['KU_5'] = np.nan
df_jr4['G2'] = np.nan
df_jr4['G3'] = np.nan
df_jr4['G4'] = np.nan
df_jr4['G5'] = np.nan
df_jr4['CIT_NONCIT'] = np.nan
df_jr4['RIN_STRATA'] = np.nan

# df_jr4['U'] = pd.to_numeric(df_jr4['U'], errors='coerce')
# df_jr4['KET'] = pd.to_numeric(df_jr4['KET'], errors='coerce')
# df_jr4['KW'] = pd.to_numeric(df_jr4['KW'], errors='coerce')
# df_jr4['ST'] = pd.to_numeric(df_jr4['ST'], errors='coerce')

# Convert 'U' column to numeric, replacing non-finite values with NaN
df_jr4['U'] = pd.to_numeric(df_jr4['U'], errors='coerce')

# Replace NaN values with a default value (e.g., 0)
df_jr4['U'].fillna(0, inplace=True)


df_jr4['KET'] = pd.to_numeric(df_jr4['KET'], errors='coerce')
# Replace NaN values with a default value (e.g., 0)
df_jr4['KET'].fillna(0, inplace=True)


df_jr4['KW'] = pd.to_numeric(df_jr4['KW'], errors='coerce')
# Replace NaN values with a default value (e.g., 0)
df_jr4['KW'].fillna(0, inplace=True)


df_jr4['ST'] = pd.to_numeric(df_jr4['ST'], errors='coerce')
# Replace NaN values with a default value (e.g., 0)
df_jr4['ST'].fillna(0, inplace=True)

# Now convert the 'U' column to integers

df_jr4['U'] =df_jr4['U'].astype(int)

df_jr4['KET'] =df_jr4['KET'].astype(int)
df_jr4['KW'] =df_jr4['KW'].astype(int)
df_jr4['ST'] = df_jr4['ST'].astype(int) 

# #### G1 Checks KU_5


# In[31]:


conditions = [
    df_jr4['U'].between(0, 4, inclusive='both'),
    df_jr4['U'].between(5, 9, inclusive='both'),
    df_jr4['U'].between(10, 14, inclusive='both'),
    df_jr4['U'].between(15, 19, inclusive='both'),
    df_jr4['U'].between(20, 24, inclusive='both'),
    df_jr4['U'].between(25, 29, inclusive='both'),
    df_jr4['U'].between(30, 34, inclusive='both'),
    df_jr4['U'].between(35, 39, inclusive='both'),
    df_jr4['U'].between(40, 44, inclusive='both'),
    df_jr4['U'].between(45, 49, inclusive='both'),
    df_jr4['U'].between(50, 54, inclusive='both'),
    df_jr4['U'].between(55, 59, inclusive='both'),
    df_jr4['U'].between(60, 64, inclusive='both'),
    df_jr4['U'].between(65, 69, inclusive='both'),
    df_jr4['U'].between(70, 74, inclusive='both'),
    df_jr4['U'].between(75, 79, inclusive='both'),
    df_jr4['U'].between(80, 84, inclusive='both'),
    (df_jr4['U'] >= 85)
]


# Define values to fill in based on conditions
values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

# Use np.select to fill in values based on conditions
df_jr4['KU_5'] = np.select(conditions, values, default=df_jr4['KU_5'])

#Check for null values in new column created 

# df[['U','G1']]
df_jr4[['KU_5']].notnull().value_counts()

# add condition when theres null values, trigger alert 
if df_jr4['KU_5'].isnull().any().any():
    print("Column KU_5 contains null values,please recheck data ")
else:
    print("Column KU_5 does not contain null values, proceed with next group")
##****** TBC 


# #### G2 Checks KU_10


# In[32]:


condition = [
    
    df_jr4['U'].between(0, 9,inclusive = 'both'),
    df_jr4['U'].between(10, 19,inclusive = 'both'),
    df_jr4['U'].between(20, 29,inclusive = 'both'),
    df_jr4['U'].between(30, 39,inclusive = 'both'),
    df_jr4['U'].between(40, 49,inclusive = 'both'),
    df_jr4['U'].between(50, 59,inclusive = 'both'),
    df_jr4['U'].between(60, 69,inclusive = 'both'),
    df_jr4['U'].between(70, 79,inclusive = 'both'),
    df_jr4['U'].between(80, 89,inclusive = 'both'),
    df_jr4['U'].between(90, 99,inclusive = 'both'),
    df_jr4['U'].between(100, 109,inclusive = 'both'),
    df_jr4['U'].between(110, 119,inclusive = 'both'),
    df_jr4['U'].between(120, 129,inclusive = 'both')
]


values = [1,2,3,4,5,6,7,8,9,10,11,12,13]

df_jr4['G2'] = np.select(condition,values, default=df_jr4['G2'])

if df_jr4['G2'].isnull().any().any():
    print("Column G2 contains null values, recheck ")
else:
    print("Column G2 does not contain null values, proceed with next group")


# #### G3 Checks KET


# In[33]:


condition = [
    ((df_jr4['KET'] == 1100) | (df_jr4['KET'] == 3210)),
    df_jr4['KET'].isin([2111, 2112, 2113, 2114, 2115, 2116, 2121, 2122, 2123, 2124, 2125, 2126, 2131, 2132, 2133, 2134, 2135, 2136, 3110, 3120, 3130, 3140, 3150, 3160, 3170, 3180, 3190, 3200, 3220, 3230, 3240, 3250, 3260, 3998, 4110, 4120, 4130, 4140, 4150, 4160, 4170, 4180, 4190, 4200, 4210, 4220, 4230, 4240, 4250, 4260, 4270, 4280, 4290, 4300, 4310, 4320, 4330, 4340, 4350, 4360, 4998]),
    df_jr4['KET'].isin([5110, 5120, 5130, 5140, 5150, 5160, 5170, 5180, 5190, 5200, 5998]),
    df_jr4['KET'].isin([6110, 6120, 6130, 6140, 6150, 6160, 6170, 6180, 6998]),
    df_jr4['KET'].isin([7110, 7120,7130, 7140, 7150, 7160, 7170, 7180, 7190, 7200, 7210, 7220, 7230, 7998, 8110, 8120, 8130, 8140, 8150, 8160, 8998, 9110, 9120, 9130, 9140, 9150, 9998])
]
 
values = [1,2,3,4,5]
df_jr4['G3'] = np.select(condition, values, default=df_jr4['G3'])
if df_jr4['G3'].isnull().any().any():
    print("Column G3 contains null values, recheck ")
else:
    print("Column G3 does not contain null values, proceed with next group")
    


# #### G4 Checks KET_SAB


# In[34]:


condition = [
    ((df_jr4['KET'] == 1100) | (df_jr4['KET'] == 3210)),
    ((df_jr4['KET'] == 3150) | (df_jr4['KET'] == 3190)),
    ((df_jr4['KET'] == 3110)),
    ((df_jr4['KET'] == 3220)),
    df_jr4['KET'].isin([2111,2112,2113,2114,2115,2116, 2121,2122,2123,2124,2125,2126, 2131,2132,2133,2134,2135,2136, 3120, 3130, 3140, 3160, 3170, 3180, 3200, 3230, 3240, 3250, 3260, 3998, 4110, 4120, 4130, 4140, 4150, 4160, 4170, 4180, 4190, 4200, 4210, 4220, 4230, 4240, 4250, 4260, 4270, 4280, 4290, 4300, 4310, 4320, 4330, 4340, 4350, 4360, 4998]),
    df_jr4['KET'].isin([ 5110, 5120, 5130, 5140, 5150, 5160, 5170, 5180, 5190, 5200, 5998]),
    df_jr4['KET'].isin([6110, 6120, 6130, 6140, 6150, 6160, 6170, 6180, 6998, 7110, 7120,7130, 7140, 7150, 7160, 7170, 7180, 7190, 7200, 7210, 7220, 7230, 7998, 8110, 8120, 8130, 8140, 8150, 8160, 8998, 9110, 9120, 9130, 9140, 9150, 9998]),
]
 
values = [1,2,3,4,5,6,7]
df_jr4['G4'] = np.select(condition, values, default=df_jr4['G4'])
if df_jr4['G4'].isnull().any().any():
    print("Column G4 contains null values, recheck ")
else:
    print("Column G4 does not contain null values, proceed with next group")
    


# #### G5 Checks : KUMPULAN ETNIK SARAWAK  


# In[35]:


condition = [
    ((df_jr4['KET'] == 1100) | (df_jr4['KET'] == 3210)),
    ((df_jr4['KET'] == 4140)),
    ((df_jr4['KET'] == 4110)),
    ((df_jr4['KET'] == 4260)),
    df_jr4['KET'].isin([2111,2112,2113,2114,2115,2116, 2121,2122,2123,2124,2125,2126, 2131,2132,2133,2134,2135,2136, 3110, 3120, 3130, 3140, 3150, 3160, 3170, 3180, 3190, 3200, 3220, 3230, 3240, 3250, 3260, 3998, 4120, 4130, 4150, 4160, 4170, 4180, 4190, 4200, 4210, 4220, 4230, 4240, 4250, 4270, 4280, 4290, 4300, 4310, 4320, 4330, 4340, 4350, 4360, 4998]),
    df_jr4['KET'].isin([5110, 5120, 5130, 5140, 5150, 5160, 5170, 5180, 5190, 5200, 5998]),
    df_jr4['KET'].isin([6110, 6120, 6130, 6140, 6150, 6160, 6170, 6180, 6998, 7110, 7120,7130, 7140, 7150, 7160, 7170, 7180, 7190, 7200, 7210, 7220, 7230, 7998, 8110, 8120, 8130, 8140, 8150, 8160, 8998, 9110, 9120, 9130, 9140, 9150, 9998]),
]
 
values = [1,2,3,4,5,6,7]
df_jr4['G5'] = np.select(condition, values, default=df_jr4['G5'])
if df_jr4['G5'].isnull().any().any():
    print("Column G5 contains null values, recheck ")
else:
    print("Column G5 does not contain null values, proceed with next group")
    


# #### G6 Checks : 6 : KUMPULAN CIT_NONCIT


# In[36]:


condition = [
    df_jr4['KW'] == 458,
    df_jr4['KW'] != 458

]

values = [1,2]

df_jr4['CIT_NONCIT'] = np.select(condition,values, default=df_jr4['CIT_NONCIT'])
mask = df_jr4[['KW','CIT_NONCIT']].isnull().any(axis=1)
row_null = df_jr4.loc[mask]

x=row_null[['KW','CIT_NONCIT']]

if not x .empty:
    print('Recheck this portion since it contain null values')
else: 
    print("Column G6 does not contain null values, proceed with next group")


# #### G7 Checks :  KUMPULAN RIN_STRATA


# In[37]:


condition = [
    ((df_jr4['ST'] ==2 ) | (df_jr4['ST'] ==1 )),
    ((df_jr4['ST'].between(3,9,inclusive = 'both')) | (df_jr4['ST'] ==0 ))
]
values = [1,2]

df_jr4['RIN_STRATA'] = np.select(condition,values, default=df_jr4['RIN_STRATA'])

mask = df_jr4[['ST','RIN_STRATA']].isnull().any(axis=1)
row_null = df_jr4.loc[mask]

x=row_null[['ST','RIN_STRATA']]

if not x .empty:
    print('Recheck this portion since it contain null values')
else: 
    print("Column RIN_STRATA does not contain null values, proceed with next group")


# In[38]:


end_time = time.time()
end_time_1 = start_time - end_time

print(f'Code took {end_time_1} seconds to finish phase 1')


# In[39]:


df_jr4_a = df_jr4


# ## MODULE 3 : MARKING PRIMARY_FIRST FOR VARIABLES XM


# In[40]:


#remove column name with spaces to standardize 
# columns_with_spaces = [col for col in df.columns if ' ' in col]
columns_with_spaces = [col for col in df_jr4.columns if ' ' in col]
new_columns = [col.replace(' ', '') for col in df_jr4.columns]
df_jr4.columns = new_columns



# In[41]:


XM = ['B', 'NG', 'DP', 'DB', 'BP', 'BP2', 'CONVERTEDBP', 'ST', 'NOTK', 'NOIR', 'S', 'NP']

for col in XM:
    if not df_jr4[col].dtype == 'object':
        df_jr4[col] = df_jr4[col].astype(str)
        


# In[42]:


df_jr4['PF_INPUT'] = df_jr4.apply(lambda row: ''.join(row[XM]), axis=1)

#Indentify first duplicate value in column PF_MAIN
mask_duplicate = df_jr4.duplicated(subset=['PF_INPUT'], keep='first')

#Mark all value in PF_MAIN with value 2 as a starting values 
df_jr4['PF_OUTPUT'] = 2

# The line of code conditions = [~mask_duplicate, mask_duplicate] creates a list of two boolean arrays
# : ~mask_duplicate and mask_duplicate. 
#     The ~ operator is the bitwise NOT operator, which in this context is used to invert 
#     the boolean values in the mask_duplicate array.

# Set values based on duplicate mask
values = [1, 2]
conditions = [~mask_duplicate, mask_duplicate]
df_jr4['PF_OUTPUT'] = np.select(conditions, values, default=df_jr4['PF_OUTPUT'])

# df_jr4[['PF_INPUT','PF_OUTPUT']].value_counts().sort_values('PF_INPUT')
# print(df_jr4.groupby(['PF_INPUT','PF_OUTPUT']).size().reset_index(name='count').sort_values('PF_OUTPUT'))


# In[43]:


end_time = time.time()
end_time_2 = start_time - end_time

print(f'Code took {end_time_2} seconds to finish phase 2 generating primary first XM ')


# ##  MODULE 3 : ADJUSTED WEIGHT 


# In[44]:


df_bir = df_jr4[['NG','RIN_STRATA','PF_OUTPUT','PF_INPUT']]
filtered_df = df_bir[df_bir['PF_OUTPUT'] == 1]
df_bir1 = filtered_df.drop('PF_OUTPUT', axis=1)


df_bir2 = df_bir[df_bir['PF_OUTPUT'] == 2]


pivoted_pf = pd.pivot_table(df_bir1, values='PF_INPUT', index='NG', columns='RIN_STRATA', aggfunc='count')
pivoted_dc = pd.pivot_table(df_bir2, values='PF_INPUT', index='NG', columns='RIN_STRATA', aggfunc='count')


# In[45]:


# df2 = pivoted_df.reset_index()
df_pf = pivoted_pf.reset_index()

#Dalam Bandar 
df_pf_01 = df_pf[['NG',1.0]]
#Luar Bandar 
df_pf_02 = df_pf[['NG',2.0]]

df_pf_01 = df_pf_01.reset_index()
df_pf_02 = df_pf_02.reset_index()

df_pf_01.set_index('index', inplace=True)
df_pf_02.set_index('index', inplace=True)

df_pf_01.rename(columns={'NG':'KOD_NEGERI',1.0:'BIL_ISI_RUMAH_RESPON_SELESAI'}, inplace=True)
df_pf_02.rename(columns={'NG':'KOD_NEGERI',2.0:'BIL_ISI_RUMAH_RESPON_SELESAI'}, inplace=True)


# In[46]:


# ##  MODULE 4 : BMP ADJUSTED WEIGHT DATA VALIDATION 


# In[47]:


#read T01
df_jr5 = df_jr4



df_t01 = read_anything(t01_path)


# In[48]:


df_t01.drop(columns=['BIL_ISI_RUMAH_RESPON_SELESAI','ADJUSTED_WEIGHT'], inplace=True)
df_t01.reset_index(drop=True, inplace=True)

#Splitting template table into 2 dataframe with 01 dalam bandar and luar bandar 02 
df_t01_01 = df_t01[df_t01['RIN_STRATA']==1]
df_t01_02 = df_t01[df_t01['RIN_STRATA']==2]

df_pf_01['KOD_NEGERI'] = df_pf_01['KOD_NEGERI'].str.replace('.', '').astype(int)
df_pf_02['KOD_NEGERI'] = df_pf_02['KOD_NEGERI'].str.replace('.', '').astype(int)
df_t01_01['KOD_NEGERI'] = pd.to_numeric(df_t01_01['KOD_NEGERI'])
df_t01_02['KOD_NEGERI'] = pd.to_numeric(df_t01_02['KOD_NEGERI'])

t01_merged = pd.merge(df_t01_01,df_pf_01, on='KOD_NEGERI')
t02_merged = pd.merge(df_t01_02,df_pf_02, on='KOD_NEGERI')
aw_df = pd.concat([t01_merged,t02_merged], ignore_index=True)


# In[49]:


months_t01 = df_t01['BULAN'].iloc[0]
years_t01 = df_t01['TAHUN'].iloc[0]
quarter_t01 = df_t01['QUARTER'].iloc[0]


# In[50]:


import pandas as pd

def fillna_and_convert_to_float(df, column_list):
    """
    Fill NaN values with 0 and convert specified columns to the float data type.

    Parameters:
        df (pd.DataFrame): The DataFrame to process.
        column_list (list): A list of column names to fill NaN and convert to float.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    df_copy = df.copy()  # Create a copy of the DataFrame to avoid modifying the original DataFrame

    # Fill NaN with 0 for specified columns
    for col in column_list:
        df_copy[col].fillna(0, inplace=True)

    # Convert specified columns to float data type
    df_copy[column_list] = df_copy[column_list].astype(float)

    return df_copy


# In[ ]:





# In[51]:


#Ensure no Nan in here dfue to arimethic requirement 

column_list = ['ADJUSTED_WEIGHT_BMP','BIL_ISI_RUMAH','BIL_ISI_RUMAH_RESPON_SELESAI']
aw_df = fillna_and_convert_to_float(aw_df, column_list)


# In[52]:


# loop through the 'BIL_ISI_RUMAH_RESPON_SELESAI' column and replace NaN values with 0
for i in range(len(aw_df)):
    if np.isnan(aw_df.loc[i, 'BIL_ISI_RUMAH_RESPON_SELESAI']):
        aw_df.loc[i, 'BIL_ISI_RUMAH_RESPON_SELESAI'] = 0
        
        
aw_df['BIL_ISI_RUMAH_RESPON_SELESAI'] = aw_df['BIL_ISI_RUMAH_RESPON_SELESAI'].astype(int)
aw_df['ADJUSTED_WEIGHT'] =aw_df['BIL_ISI_RUMAH'] / aw_df['BIL_ISI_RUMAH_RESPON_SELESAI'] 
aw_df['ADJUSTED_WEIGHT'] = round(aw_df['ADJUSTED_WEIGHT'], 2)
aw_df['SEMAKAN_AW_BMP'] = (aw_df['ADJUSTED_WEIGHT'] - aw_df['ADJUSTED_WEIGHT_BMP'])
aw_df['SEMAKAN_AW_BMP'] = round(aw_df['SEMAKAN_AW_BMP'], 2)
aw_df.rename(columns={'KOD_NEGERI':'NG'},inplace=True )





# In[53]:


def nan_clear(col_name): 
    for i in range(len(aw_df)):
        if np.isnan(aw_df.loc[i,col_name]):
            aw_df.loc[i, 'ADJUSTED_WEIGHT_BMP'] = 0



# In[54]:


nan_clear('ADJUSTED_WEIGHT_BMP')
nan_clear('ADJUSTED_WEIGHT')
nan_clear('SEMAKAN_AW_BMP')


# In[55]:


df_temp = read_anything(temp_path)


# In[56]:


df_aw2 = aw_df.merge(df_temp, how='left', on=('NG','RIN_STRATA'))


# In[57]:


df_aw3 = df_aw2[['NG','RIN_STRATA','ST','ADJUSTED_WEIGHT']]


# In[58]:


# aw_df.to_csv(bmp_awdf_compare_check+'/'+'awdf_bmp_compare_check.csv')

if any(aw_df['SEMAKAN_AW_BMP'] != 0):
    print(f'Error: SEMAKAN_AW_BMP failed because has non-zero value(s), please recheck BPPD value and jr4 new adjusted weight')
    

else: 
    print('Success: SEMAKAN_AW_BMP success since there are no differences between BPPD and DOSM adjusted weight')


# In[59]:


#Remap aw_df RIN STRATA to ST too ADW VALUE 

# MERGE AW RIN_STRATA + KOD_NEGERI + ADJUSTED_WEIGHT into a new DF = df_jr4_aw 

#Change KOD_NEGERI to NG 
try: 
    aw_df['NG'] = aw_df['KOD_NEGERI']
    aw_df.drop('KOD_NEGERI', axis=1,inplace=True)
    #Change df_jr5(RIN_STRATA) to remove floats if any to int 

except: 
    print(f'Light Warning !: Column KOD_NEGERI has been removed, ignore renaming column from KOD_NEGERI to NG')

df_jr5['RIN_STRATA'] = df_jr5['RIN_STRATA'].astype(int)
df_jr5['NG'] = df_jr5['NG'].str.replace('.', '').astype(int)
df_jr5['ST']= df_jr5['ST'].astype(int)



# In[60]:


df_jr4_aw = df_jr5.merge(df_aw3, how='left', on=('NG', 'RIN_STRATA','ST'))
print(f'Success : JR4 has been merged with latest checked adjusted weight from BPPD & DOSM values')


# In[61]:


df_aw3_a = df_aw3


# ##  MODULE 5 : JADUAL A_1


# In[62]:


#read 1


# In[63]:


pivoted_pf = pd.pivot_table(df_jr4_aw, values='ADJUSTED_WEIGHT',index=['CIT_NONCIT','NG','KU_5'], columns=['J','G3'], aggfunc='sum')
# pivoted_pf.index.get_level_values(0).value_counts()
# , removed due to unclear files provided 


# In[64]:


# Function Library 
def pivot_table(group, ng_code):
    pivoted_pf = pd.pivot_table(df_jr4_aw, values='ADJUSTED_WEIGHT', index=['CIT_NONCIT', 'NG', 'KU_5'], columns=['J',group], aggfunc='sum')
    filtered_pivoted_pf_sem = pivoted_pf.loc[pivoted_pf.index.get_level_values('CIT_NONCIT')==1]
    filtered_pivoted_pf = filtered_pivoted_pf_sem.loc[filtered_pivoted_pf_sem.index.get_level_values('NG').isin(ng_code)]

    # Return the output dataframe
    return filtered_pivoted_pf

def reset_frame(df):
    df = df.reset_index()
    df.columns = df.columns.map(lambda x: '_'.join(map(str, x)))
    df = df.rename(columns={'CIT_NONCIT_':'CIT_NONCIT','NG_':'NG','KU_5_':'KU_5'})
    return df

def pivot_table_bw():
    pivoted_pf = pd.pivot_table(df_jr4_aw, values='ADJUSTED_WEIGHT', index=['CIT_NONCIT','KU_5','NG'], columns=['J'], aggfunc='sum')
    filtered_pivoted_pf = pivoted_pf.loc[pivoted_pf.index.get_level_values('CIT_NONCIT')==2]
    # Return the output dataframe
    return filtered_pivoted_pf



# In[65]:


ng_code= [1,2,3,4,5,6,7,8,9,10,11,14,16]
group='G3'
df_sem = pivot_table(group,ng_code)

ng_code= [12,15]
group='G4'
df_sab = pivot_table(group,ng_code)

ng_code= [13]
group='G5'
df_sar = pivot_table(group,ng_code)


# In[66]:


#df_sem_1 ==  all column reset into 1 level and column name changed 
df_sem_1 = reset_frame(df_sem)
df_sab_1 = reset_frame(df_sab)
df_sar_1 = reset_frame(df_sar)


# In[67]:


df_sar_1.columns = df_sar_1.columns.str.replace('.0','')
df_sab_1.columns = df_sab_1.columns.str.replace('.0','')
df_sem_1.columns = df_sem_1.columns.str.replace('.0','')


# In[68]:


df_bw = pivot_table_bw()
df_bw = df_bw.reset_index()
df_bw = df_bw.rename(columns={1:'1_BW',2:'2_BW'})
df_bw = df_bw.rename(columns={'1': '1_BW', '2': '2_BW'})
# add clause to check whether column successfully renamed from 1 to 1_BW or not
df_bw = df_bw.drop('CIT_NONCIT',axis=1)
if '1_BW' in df_bw.columns and '2_BW' in df_bw.columns:
    print("Columns renamed successfully.")
else:
    print("Columns renaming failed.")


# In[69]:


df_sem_bw_merged = df_sem_1.merge(df_bw,how='left',on=['NG','KU_5'])
df_sab_bw_merged = df_sab_1.merge(df_bw,how='left',on=['NG','KU_5'])
df_sar_bw_merged = df_sar_1.merge(df_bw,how='left',on=['NG','KU_5'])


# In[70]:


print('Success: df_semenanjung, df_sabah & df_sarawak have been generated to produce trend documents')


# ### MODULE 6 : BPPD + JADUAL A1 PRODUCE TREND DOCUMENTS


# In[71]:


#read file t02 ()

df_to2 = read_anything(path_t02)


# In[72]:


#convert male to 1 & female to 2 
df_to2['J'] = np.nan

conditions = [
    ((df_to2['GENDER'] == 'MALE')),
    ((df_to2['GENDER'] == 'FEMALE')),
]

values = [1,2]

df_to2['J'] = np.select(conditions,values,default=df_to2['J'])

df_to2.reset_index
df_to2['J'] = df_to2['J'].astype(int)


# In[73]:


# Rename columns JADUAL A1
df_to2 = df_to2.rename(columns={
    'BPPD_MALAY': '1_SEM',
    'BPPD_OTHER_BUMI': '2_SEM',
    'BPPD_CHINESE': '3_SEM',
    'BPPD_INDIAN': '4_SEM',
    'BPPD_OTHERS': '5_SEM',
    'BPPD_BUKAN_WARGA': 'BW_SEM',
    
    'BPPD_SABAH_MELAYU': '1_SAB',
    'BPPD_SABAH_KADAZAN': '2_SAB',
    'BPPD_SABAH_BAJAU': '3_SAB',
    'BPPD_SABAH_MURUT': '4_SAB',
    'BPPD_SABAH_BUMIPUTERA LAIN': '5_SAB',
    'BPPD_SABAH_CINA': '6_SAB',
    'BPPD_SABAH_LAIN LAIN': '7_SAB',
    'BPPD_SABAH_BUKAN_WARGA': 'BW_SAB',
    
    'BPPD_SARAWAK_MELAYU': '1_SAR',
    'BPPD_SARAWAK_IBAN': '2_SAR',
    'BPPD_SARAWAK_BIDAYUH': '3_SAR',
    'BPPD_SARAWAK_MELANAU': '4_SAR',
    'BPPD_SARAWAK_BUMIPUTERA': '5_SAR',
    'BPPD_SARAWAK_CINA': '6_SAR',
    'BPPD_SARAWAK_LAIN_LAIN': '7_SAR',
    'BPPD_SARAWAK_BUKAN_WARGA': 'BW_SAR',
    'KOD_NEGERI': 'NG'
})


# #rename column in this format 1_X_B for Male & 2_X_B for Female 
# #make it as format of TO # GENDERNUMBER{1 OR 2} _#{NAMING FORMAT BPPD }}}
# #AFTER COLUMN GENDER, START TO FILL IN 1_"""""" OR 2_'''''' FRONT OF EACH COLUMNS 

# #DROP COLUMNS 
df_to3 = df_to2.drop(columns={'NEGERI', 'AGE_GROUP', 'GENDER'})


# # df_female = df.loc[df.index.get_level_values('GENDER') == 'MALE']


# In[74]:


#Split male female then rename column 
def split_gender(q,df):
    df = df_to3.loc[df_to2['J'].isin([q])]
    return df


# In[75]:


#FUNCTION TO 1. RENAME COLUMN NAME & 2.  THEN SPLIT IT INTO 3 UNIQUE DATAFRAME SEM , SAB & SAR 
#get the column index of KU_5 to identify column to fill in 
def rename_col(df,index_to_fill_after,fillwith):
    cols = list(df.columns)
    ku5_index = cols.index(index_to_fill_after)+1

    #insert new column after certain column name 
    index_to_edit = cols[ku5_index:]
    new_cols = cols[:ku5_index] + [f'{fillwith}_' + x for x in index_to_edit]
    df.columns = new_cols
    return df

    #Alternatives
    # new_cols = cols[:ku5_index+1]
    # for x in index_to_edit:
    #     new_cols.append('1_'+x)

    # new_cols = ['1_'+ col for col in base_index]
    # new_cols = base_index + new_cols


# In[76]:


q = 1
df_male = pd.DataFrame()
df = df_male
df_male = split_gender(q,df)

q = 2
df_female = pd.DataFrame()
df = df_female
df_female = split_gender(q,df)

df = df_male
index_to_fill_after = 'KU_5'
fillwith = 1

df_male = rename_col(df,index_to_fill_after,fillwith)

df = df_female
index_to_fill_after = 'KU_5'
fillwith = 2

df_female = rename_col(df,index_to_fill_after,fillwith)


# In[77]:


#split female to 3 category sem, sab & sar 
#then convert to csv for calculation 


# In[78]:


def get_col_list(df,col_start_name,col_end_name):
    #get all column name in df 
    cols = list(df.columns)
    #get index of the desired column name start & end 
    starting_index = cols.index(col_start_name)
    last_col = cols.index(col_end_name)+1
    
    #front 2 column must maintain in each filter 
    base_index = cols[:2]
    
    #get the range of desired columns
    desired_column = cols[starting_index:last_col]
    new_col = base_index + desired_column
    return new_col

def separator(df,ng_list, col_list):
    df = df.loc[df['NG'].isin(ng_list), col_list]
    print('Success: Trend separated')
    return df

def merge_m_f(df1,df2):
    df = df1.merge(df2,how='left',on=('NG','KU_5'))
    print('Success: Trend merged')
    return df

def save(df,path,name):
    df.to_excel(path+'/'+name+'.xlsx',index=False)
    
    print('Success : Files were being saved in xlsx format for quality checking. It will be moved to other container once this process finished ')


# In[79]:


#sem master
df = df_male
col_start_name = '1_1_SEM'
col_end_name = '1_BW_SEM'
df_sem_col_list = get_col_list(df,col_start_name,col_end_name)

col_list = df_sem_col_list
ng_list = 1,2,3,4,5,6,7,8,9,10,11,14,16
df =df_male
df_sem_bppd_male = separator(df,ng_list, col_list)
df_sem_bppd_male

#female & sem
df = df_female
col_start_name = '2_1_SEM'
col_end_name = '2_BW_SEM'
df_sem_col_list = get_col_list(df,col_start_name,col_end_name)

col_list = df_sem_col_list
ng_list = 1,2,3,4,5,6,7,8,9,10,11,14,16
df =df_female
df_sem_bppd_female = separator(df,ng_list, col_list)
df_sem_bppd_female

df1= df_sem_bppd_male
df2= df_sem_bppd_female
df_sem_bppd_master = merge_m_f(df1,df2)

df = df_sem_bppd_master
path = bppd_database #temp_path
name = 'df_sem_bppd_master'
save(df,path,name)


# In[80]:


#sab master
df = df_male
col_start_name = '1_1_SAB'
col_end_name = '1_BW_SAB'
df_sab_col_list = get_col_list(df,col_start_name,col_end_name)

col_list = df_sab_col_list
ng_list = 12,15
df =df_male
df_sab_bppd_male = separator(df,ng_list, col_list)
df_sab_bppd_male

#female & sem
df = df_female
col_start_name = '2_1_SAB'
col_end_name = '2_BW_SAB'
df_sab_col_list = get_col_list(df,col_start_name,col_end_name)

col_list = df_sab_col_list
ng_list = 12,15
df =df_female
df_sab_bppd_female = separator(df,ng_list, col_list)
df_sab_bppd_female

df1= df_sab_bppd_male
df2= df_sab_bppd_female
df_sab_bppd_master = merge_m_f(df1,df2)

df = df_sab_bppd_master
path = bppd_database #temp_path
name = 'df_sab_bppd_master'
save(df,path,name)


# In[81]:


#sar master
df = df_male
col_start_name = '1_1_SAR'
col_end_name = '1_BW_SAR'
df_sar_col_list = get_col_list(df,col_start_name,col_end_name)

col_list = df_sar_col_list
ng_list =13,13
df =df_male
df_sar_bppd_male = separator(df,ng_list, col_list)
df_sar_bppd_male

#female & sar
df = df_female
col_start_name = '2_1_SAR'
col_end_name = '2_BW_SAR'
df_sar_col_list = get_col_list(df,col_start_name,col_end_name)

col_list = df_sar_col_list
ng_list = 13,13
df =df_female
df_sar_bppd_female = separator(df,ng_list, col_list)
df_sar_bppd_female

df1= df_sar_bppd_male
df2= df_sar_bppd_female
df_sar_bppd_master = merge_m_f(df1,df2)

df = df_sar_bppd_master
path = bppd_database #temp_path
name = 'df_sar_bppd_master'
save(df,path,name)


# In[ ]:





# In[82]:


###AFTER SAVINGS 


# In[83]:


#jaduala1 dataframe 
df_sem_a1_master = df_sem_bw_merged
df_sab_a1_master = df_sab_bw_merged
df_sar_a1_master = df_sar_bw_merged


# In[84]:


#merging dataframe 
df1 = df_sem_bppd_master
df2 = df_sem_a1_master

df_sem_master = merge_m_f(df1,df2)



# In[85]:


#merging dataframe 
df1 = df_sab_bppd_master
df2 = df_sab_a1_master

df_sab_master = merge_m_f(df1,df2)


# In[86]:


#merging dataframe 
df1 = df_sar_bppd_master
df2 = df_sar_a1_master

df_sar_master = merge_m_f(df1,df2)


# In[87]:


def rearrange_columns(df,structure):
    # Define a regular expression pattern to match the integer in the column names
    pattern = r'(\d+)'

    # Extract the integer from each column name using the regular expression
    int_cols = [(int(re.findall(pattern, col)[0]), col) for col in df.columns if re.findall(pattern, col)]

    # Sort the column names based on the extracted integer
    int_cols.sort()

    # Rearrange the column names by pairing the '_SEM' columns with their corresponding non-SEM columns
    new_cols = []
    for i in range(0, len(int_cols), 2):
        sem_col = f'{int_cols[i][1]}_{structure}'
        new_cols.append(int_cols[i][1])
        new_cols.append(sem_col)
    new_cols.remove('KU_5_'+structure) if 'KU_5_'+structure in new_cols else print("'KU_5_'+structure' not found in new_cols")


    # Add any remaining columns that were not paired to the end of the list
    if len(new_cols) < len(df.columns):
        for col in df.columns:
            if col not in new_cols:
                new_cols.append(col)
    if 'CIT_NONCIT' in new_cols:
        new_cols.remove('CIT_NONCIT')
    else:
        print("'CIT_NONCIT' not found in new_cols")

    # Reorder the columns in the dataframe
#     df = df[new_cols]
# ** changed due to structure non sense
    filter_nonsense = [x for x in new_cols if x in df.columns]
    df = df[filter_nonsense]
    print(f'{structure} re-arranged as per client requirements')
            
    return df


# In[88]:


df = df_sem_master
structure = 'SEM'
df_sem_master_trend = rearrange_columns(df,structure)

df = df_sab_master
structure = 'SAB'
df_sab_master_trend = rearrange_columns(df,structure)

df = df_sar_master
structure = 'SAR'
df_sar_master_trend = rearrange_columns(df,structure)


# In[89]:


#OK


# In[90]:


df_sem_master


# In[91]:


def separator2(df,ng_list):
    df = df.loc[df['NG'].isin(ng_list)]
    #remove nan , to allow arimethic operation and change to int or float 
    for x in df.columns:
        df[x] = df[x].astype(float)
    
    #run function to change 
    print(f'NG:{ng_list} separated into other dataframe')
    return df


# In[92]:


#sem ng==1

df = df_sem_master_trend
ng_list = 1,1
df_sem_master_trend_1 = separator2(df,ng_list)

#sem ng==2

df = df_sem_master_trend
ng_list = 2,2
df_sem_master_trend_2 = separator2(df,ng_list)

#sem ng==3

df = df_sem_master_trend
ng_list = 3,3
df_sem_master_trend_3 = separator2(df,ng_list)

#sem ng==4

df = df_sem_master_trend
ng_list = 4,4
df_sem_master_trend_4 = separator2(df,ng_list)

#sem ng==5

df = df_sem_master_trend
ng_list = 5,5
df_sem_master_trend_5 = separator2(df,ng_list)

#sem ng==6

df = df_sem_master_trend
ng_list = 6,6
df_sem_master_trend_6 = separator2(df,ng_list)

#sem ng==7

df = df_sem_master_trend
ng_list = 7,7
df_sem_master_trend_7 = separator2(df,ng_list)

#sem ng==8

df = df_sem_master_trend
ng_list = 8,8
df_sem_master_trend_8 = separator2(df,ng_list)

#sem ng==9

df = df_sem_master_trend
ng_list = 9,9
df_sem_master_trend_9 = separator2(df,ng_list)

#sem ng==10

df = df_sem_master_trend
ng_list = 10,10
df_sem_master_trend_10 = separator2(df,ng_list)

#sem ng==11

df = df_sem_master_trend
ng_list = 11,11
df_sem_master_trend_11 = separator2(df,ng_list)

#sabah ng==12
df = df_sab_master_trend
ng_list = 12,12
df_sab_master_trend_12 = separator2(df,ng_list)

#sarawak ng==13
df = df_sar_master_trend
ng_list = 13,13
df_sar_master_trend_13 = separator2(df,ng_list)

#sem ng==14

df = df_sem_master_trend
ng_list = 14,14
df_sem_master_trend_14 = separator2(df,ng_list)

#sabah ng==15

df = df_sab_master_trend
ng_list = 15,15
df_sab_master_trend_15 = separator2(df,ng_list)

#sem ng==16

df = df_sem_master_trend
ng_list = 16,16
df_sem_master_trend_16 = separator2(df,ng_list)


# In[93]:


#in order to generate pop fac , we need to change nan to 0 to allow arimethic operation and change to float at least


# In[94]:


#function to generate DIV column for all columns 
def generate_div(cols,region):
    a = [col for col in cols if col.endswith(region)]
    b = [re.findall(r'\d{1,3}', col)[0] for col in a]
    c = cols.filter(regex='^(' + '|'.join(b) + ')')
    for i, row in c.iterrows():
        for col1, value1 in row.items():
            if col1.endswith(region):
                col2 = col1[:-4]
                value2 = row[col2]
                new_col_name = col2 + '_DIV'
                if value2 != 0:
                    c.loc[i, new_col_name] = value1 / value2
      
    
    return c 


# In[95]:


cols = df_sem_master_trend_1
region = 'SEM'
df_sem_div_1 = generate_div(cols, region)

cols = df_sem_master_trend_2
region = 'SEM'
df_sem_div_2 = generate_div(cols, region)

cols = df_sem_master_trend_3
region = 'SEM'
df_sem_div_3 = generate_div(cols, region)

cols = df_sem_master_trend_4
region = 'SEM'
df_sem_div_4 = generate_div(cols, region)

cols = df_sem_master_trend_5
region = 'SEM'
df_sem_div_5 = generate_div(cols, region)

cols = df_sem_master_trend_6
region = 'SEM'
df_sem_div_6 = generate_div(cols, region)

cols = df_sem_master_trend_7
region = 'SEM'
df_sem_div_7 = generate_div(cols, region)

cols = df_sem_master_trend_8
region = 'SEM'
df_sem_div_8 = generate_div(cols, region)

cols = df_sem_master_trend_9
region = 'SEM'
df_sem_div_9 = generate_div(cols, region)

cols = df_sem_master_trend_10
region = 'SEM'
df_sem_div_10 = generate_div(cols, region)

cols = df_sem_master_trend_11
region = 'SEM'
df_sem_div_11 = generate_div(cols, region)

cols = df_sab_master_trend_12
region = 'SAB'
df_sab_div_12 = generate_div(cols, region)

cols = df_sar_master_trend_13
region = 'SAR'
df_sar_div_13 = generate_div(cols, region)

cols = df_sem_master_trend_14
region = 'SEM'
df_sem_div_14 = generate_div(cols, region)

cols = df_sab_master_trend_15
region = 'SAB'
df_sab_div_15 = generate_div(cols, region)

cols = df_sem_master_trend_16
region = 'SEM'
df_sem_div_16 = generate_div(cols, region)

print(f'Phase Division Completed Successfully')


# In[96]:


import pandas as pd
import re

def concatenate_div_columns(df):
    d = [col for col in df if col.endswith('DIV')]
    f = df.loc[:,d]
    g = f.round(2)

    col1 = pd.DataFrame([])
    col2 = pd.DataFrame([])
    col3 = pd.DataFrame([])
    for column in g.columns:
        if column.startswith('1') and not re.match(r'.*BW_DIV.*', column):
            col1 = pd.concat([col1, g[column].reset_index(drop=True)], axis=0)
            col1 = col1.fillna(0)
        elif column.startswith('2') and not re.match(r'.*BW_DIV.*', column):
            col2 = pd.concat([col2, g[column].reset_index(drop=True)], axis=0)
            col2 = col2.fillna(0)
        elif re.match(r'.*BW_DIV.*', column):
            col3 = pd.concat([col3, g[column].reset_index(drop=True)], axis=0)
            col3 = col3.fillna(0)

    concatenated_df = pd.concat([col1.reset_index(drop=True), col2.reset_index(drop=True), col3.reset_index(drop=True)], axis=1)
    concatenated_df.columns = ['Male', 'Female', 'Non Citizen']

    return concatenated_df


# In[97]:


df = df_sem_div_1
df_popfac_1 = concatenate_div_columns(df)

df = df_sem_div_2
df_popfac_2 = concatenate_div_columns(df)

df = df_sem_div_3
df_popfac_3 = concatenate_div_columns(df)

df = df_sem_div_4
df_popfac_4 = concatenate_div_columns(df)

df = df_sem_div_5
df_popfac_5 = concatenate_div_columns(df)

df = df_sem_div_6
df_popfac_6 = concatenate_div_columns(df)

df = df_sem_div_7
df_popfac_7 = concatenate_div_columns(df)

df = df_sem_div_8
df_popfac_8 = concatenate_div_columns(df)

df = df_sem_div_9
df_popfac_9 = concatenate_div_columns(df)

df = df_sem_div_10
df_popfac_10 = concatenate_div_columns(df)

df = df_sem_div_11
df_popfac_11 = concatenate_div_columns(df)

df = df_sab_div_12
df_popfac_12 = concatenate_div_columns(df)

df = df_sar_div_13
df_popfac_13 = concatenate_div_columns(df)

df = df_sem_div_14
df_popfac_14 = concatenate_div_columns(df)

df = df_sab_div_15
df_popfac_15 = concatenate_div_columns(df)

df = df_sem_div_16
df_popfac_16 = concatenate_div_columns(df)

print(f'Popfac calculated successfully for all negeri ')


# In[98]:


for i in range(1,17):
    df_name = f'df_popfac_{i}'
    df = globals()[df_name]
    df.to_excel(pop_fac_check+f'/{df_name}.xlsx',index=False)


# In[99]:


# # aw_df.to_csv(aw_df_check+'/'+'ADW_CHECK.csv')
# aw_df 
# for i in range(1,17):
#     df_name = f'df_popfac_{i}'
#     df = globals()[df_name]
#     df.to_csv(qualitycheck_merged_path+'/'+f'{df_name}.csv',index=False)


# ### MODULE POPFAC VALUE CONVERTER


# In[100]:


#Store temporary popfac value in dictionary 

import os
import glob
import pandas as pd
import numpy as np

file_list = []

for x in range (1,17):
    file_name = os.path.join(popfac_path,f'df_popfac_{x}*')
    #make this as an array of path , to make it loopable 
    files = glob.glob(file_name)
    if files:
        latest_files = max(files, key=os.path.getctime)
# os.path.getctime returns the time of the last metadata change to a file, which includes changes to the file's permissions, ownership, or timestamps.
# os.path.getmtime returns the time of the last modification to the file's content, which includes any changes to the actual data in the file.
# So which one you should use depends on what you consider to be the relevant change to the file.
# In most cases, os.path.getmtime is the more appropriate choice, as it reflects changes to the actual data in the file. However, if you are interested in changes to the file's metadata, such as changes to its permissions or ownership, then os.path.getctime would be more appropriate.
# In the context of reading the latest file in a directory, you would generally want to use os.path.getmtime, as you are likely interested in the latest version of the file's content.
        file_list.append(latest_files)
    
file_dict = {}

for i, file_path in enumerate(file_list):
    df_name = f'df_{i+1}'
    file_dict[df_name] = pd.read_excel(file_path) 



# In[101]:


#12 ,13 & 15 have 7 etnik number 
#the rest have 5 etnik numbers 
#ETNIK SABAH 12 & 15 
#ETNIK SARAWAK 13 
#ETINIK SEMENANJUNG 1 TO 11


def popfac_converter(ng_val,df_num,gender_num,entik_number,etnik_label,popfac_name): 
    df = file_dict[df_num]
    df['NG'] = ng_val
    df = df[[columnd,'NG']]
    df['J'] = gender_num
    df['KU_5'] = 0
    df['CIT_NONCIT'] = 1
    df[etnik_label] = 0



    for i in range(entik_number):
        start_idx = i * 18
        end_idx = start_idx + 18
        df.loc[start_idx:end_idx-1, etnik_label] = i+1

    num_rows = df.shape[0]

    seq = np.tile(np.arange(1, 19), (num_rows//18 + 1))[:num_rows]
    df['KU_5'] = seq

    df[popfac_name] = df[columnd]
    df = df.drop(columnd,axis=1)
    return df


def popfac_converter_noncit(ng_val,df_num,gender_num,etnik_label,popfac_name): 
    df = file_dict[df_num]
    df['NG'] = ng_val

    df = df[[columnd,'NG']]
    df['J'] = 0
    df['KU_5'] = 0
    df['CIT_NONCIT'] = 2
    df[etnik_label] = 0
    
    for i in range(3):
        start_idx = i*18
        end_idx =start_idx + 18
        df.loc[start_idx:end_idx-1, 'J'] = i+1

    for i in range(5):
        start_idx = i * 18
        end_idx = start_idx + 18
        df.loc[start_idx:end_idx, etnik_label] = i+1

    num_rows = df.shape[0]

    seq = np.tile(np.arange(1, 19), (num_rows//18 + 1))[:num_rows]
    df['KU_5'] = seq

    df[popfac_name] = df[columnd]
    df = df.drop(columnd,axis=1)
    df = df.dropna(subset=[popfac_name])
    return df

def combiner(df1,df2,df3):
    df = pd.concat([df1,df2,df3], axis=0, ignore_index=True)
    df
    print(f'All dataframe combined successfully')
    return df     
#     duplicates = df_concat.columns[df_concat.columns.duplicated()]
# The axis parameter in pd.concat() specifies the axis along which the data frames will be concatenated.
# When axis=0, the data frames will be concatenated vertically, i.e., rows will be appended one after another, which means that the resulting data frame will have more rows.
# When axis=1, the data frames will be concatenated horizontally, i.e., columns will be appended side by side, which means that the resulting data frame will have more columns.
    


# In[102]:


df_popfac_sem = pd.DataFrame()
df_popfac_sar = pd.DataFrame()
df_popfac_sab = pd.DataFrame()


# In[103]:


ng_val = 16
df_num = 'df_16'
columnd = 'Female'
gender_num = 2
entik_number = 5
etnik_label = 'G3'
popfac_name = 'POPFAC_SEM' 
df2 = popfac_converter(ng_val,df_num,gender_num,entik_number,etnik_label,popfac_name)


columnd = 'Male'
gender_num = 1
df1 = popfac_converter(ng_val,df_num,gender_num,entik_number,etnik_label,popfac_name)

columnd = 'Non Citizen'
df3 = popfac_converter_noncit(ng_val,df_num,gender_num,etnik_label,popfac_name)


df = combiner(df1,df2,df3)
df_popfac_sem = pd.concat([df,df_popfac_sem],axis=0,ignore_index=True)


# In[104]:


#Etnik SABAH 15 

ng_val = 15
df_num = 'df_15'
columnd = 'Female'
gender_num = 2
entik_number = 7
etnik_label = 'G4'
popfac_name = 'POPFAC_SAB' 
df2 = popfac_converter(ng_val,df_num,gender_num,entik_number,etnik_label,popfac_name)

columnd = 'Male'
gender_num = 1
df1 = popfac_converter(ng_val,df_num,gender_num,entik_number,etnik_label,popfac_name)

columnd = 'Non Citizen'
df3 = popfac_converter_noncit(ng_val,df_num,gender_num,etnik_label,popfac_name)

df = combiner(df1,df2,df3)
df_popfac_sab = pd.concat([df,df_popfac_sab],axis=0,ignore_index=True)


# In[105]:


#Etnik SABAH  12 

ng_val = 12
df_num = 'df_12'
columnd = 'Female'
gender_num = 2
entik_number = 7
etnik_label = 'G4'
popfac_name = 'POPFAC_SAB' 
df2 = popfac_converter(ng_val,df_num,gender_num,entik_number,etnik_label,popfac_name)

columnd = 'Male'
gender_num = 1
df1 = popfac_converter(ng_val,df_num,gender_num,entik_number,etnik_label,popfac_name)

columnd = 'Non Citizen'
df3 = popfac_converter_noncit(ng_val,df_num,gender_num,etnik_label,popfac_name)

df= combiner(df1,df2,df3)
df_popfac_sab = pd.concat([df,df_popfac_sab],axis=0,ignore_index=True)


# In[106]:


#Etnik SARAWAK 13

ng_val = 13
df_num = 'df_13'
columnd = 'Female'
gender_num = 2
entik_number = 7
etnik_label = 'G5'
popfac_name = 'POPFAC_SAR' 
df2 = popfac_converter(ng_val,df_num,gender_num,entik_number,etnik_label,popfac_name)

columnd = 'Male'
gender_num = 1
df1 = popfac_converter(ng_val,df_num,gender_num,entik_number,etnik_label,popfac_name)

columnd = 'Non Citizen'
df3 = popfac_converter_noncit(ng_val,df_num,gender_num,etnik_label,popfac_name)

df = combiner(df1,df2,df3)
df_popfac_sar = pd.concat([df,df_popfac_sar],axis=0,ignore_index=True)


# In[107]:


ng_val = 14
df_num = 'df_14'
columnd = 'Female'
gender_num = 2
entik_number = 5
etnik_label = 'G3'
popfac_name = 'POPFAC_SEM' 
df2 = popfac_converter(ng_val,df_num,gender_num,entik_number,etnik_label,popfac_name)

columnd = 'Male'
gender_num = 1
df1 = popfac_converter(ng_val,df_num,gender_num,entik_number,etnik_label,popfac_name)

columnd = 'Non Citizen'
df3 = popfac_converter_noncit(ng_val,df_num,gender_num,etnik_label,popfac_name)


df = combiner(df1,df2,df3)
df_popfac_sem = pd.concat([df,df_popfac_sem],axis=0,ignore_index=True)


# In[108]:


for x in range(1,12):
    ng_val = x
    df_num = f'df_{x}'
    columnd = 'Female'
    gender_num = 2
    entik_number = 5
    etnik_label = 'G3'
    popfac_name = 'POPFAC_SEM' 
    df2 = popfac_converter(ng_val,df_num,gender_num,entik_number,etnik_label,popfac_name)


    columnd = 'Male'
    gender_num = 1
    df1 = popfac_converter(ng_val,df_num,gender_num,entik_number,etnik_label,popfac_name)

    columnd = 'Non Citizen'
    df3 = popfac_converter_noncit(ng_val,df_num,gender_num,etnik_label,popfac_name)

    df = combiner(df1,df2,df3)
    df_popfac_sem = pd.concat([df,df_popfac_sem],axis=0,ignore_index=True)


# In[109]:


df1 = df_jr4_a


# In[110]:


x = ['G3','NG','CIT_NONCIT','KU_5','J']
df_popfac_sem[x]= df_popfac_sem[x].astype(int)
df_popfac_sem[x].dtypes


# In[111]:


x = ['G4','NG','CIT_NONCIT','KU_5','J']
df_popfac_sab[x] = df_popfac_sab[x].astype(int)
df_popfac_sab[x].dtypes


# In[112]:


x = ['G5','NG','CIT_NONCIT','KU_5','J']
df_popfac_sar[x] = df_popfac_sar[x].astype(int)
df_popfac_sar[x].dtypes


# In[113]:


x = ['NG','CIT_NONCIT','KU_5','J']
for col in x:
    df1[col] = pd.to_numeric(df1[col], errors='coerce')
df1[x] = df1[x].astype(int)
df1[x].dtypes


# In[114]:


# df1 = df_jr4_a
df_final_1 = df1.merge(df_popfac_sem, how ='outer' , on=['G3','NG','CIT_NONCIT','KU_5','J'])
df_final_2 = df_final_1.merge(df_popfac_sab, how ='outer', on=['G4','NG','CIT_NONCIT','KU_5','J'])
df_final_3 = df_final_2.merge(df_popfac_sar, how='outer' ,on=['G5','NG','CIT_NONCIT','KU_5','J'])


# In[115]:


df_final_3


# In[116]:


#delete row yang NOID == nan 
noid = df_final_3.columns[0]
df_final_3.dropna(subset=[noid], inplace=True)

#merge popfac sem sab & sar if nan value 
df_final_3['POPFAC'] = df_final_3[['POPFAC_SEM', 'POPFAC_SAB', 'POPFAC_SAR']].apply(lambda x: '|'.join(x.dropna().astype(str)), axis=1)
df_final_3.drop(['POPFAC_SEM', 'POPFAC_SAB', 'POPFAC_SAR'], axis=1, inplace=True)



# In[117]:


df_final_3


# In[118]:


#read files 
df_adw = df_aw3_a 


# In[119]:


df_merged_popfac_adw = df_final_3.merge(df_adw, how='outer', on=['ST','NG','RIN_STRATA'])
df_merged_popfac_adw.dropna(subset=[noid],inplace=True)


# In[120]:


df_merged_popfac_adw['POPFAC'] = df_merged_popfac_adw['POPFAC'].replace('',np.nan).fillna(1)


# In[121]:


df_merged_popfac_adw['POPFAC']= df_merged_popfac_adw['POPFAC'].astype(float)


# In[122]:


df_merged_popfac_adw['PEMBERAT_1'] = df_merged_popfac_adw['ADJUSTED_WEIGHT']*df_merged_popfac_adw['POPFAC']


# In[123]:


df_jr4_new = df_merged_popfac_adw


# In[124]:


df_jr4_pivotted = df_jr4_new.pivot_table(index='NG',columns='CIT_NONCIT',values='PEMBERAT_1', aggfunc='sum')


# In[125]:


df_jr4_pivotted_temp = df_jr4_pivotted.reset_index(drop=False)
df_jr4_pivotted_temp_2 = df_jr4_pivotted_temp.reset_index(drop=True)
df1x = df_jr4_pivotted_temp_2.loc[:,('NG',1.0)]
df1y = df_jr4_pivotted_temp_2.loc[:,('NG',2.0)]
df1x['CIT_NONCIT'] = 1
df1y['CIT_NONCIT'] = 2
df1x = df1x.rename(columns={1.0:'ANGGARAN_SAMPEL_STB'})
df1y = df1y.rename(columns={2.0:'ANGGARAN_SAMPEL_STB'})

df1x= df1x.reset_index().rename_axis(None,axis=1)
df1y = df1y.reset_index().rename_axis(None,axis=1)
df11 = df1x.loc[:,('NG','ANGGARAN_SAMPEL_STB','CIT_NONCIT')]
df22 = df1y.loc[:,('NG','ANGGARAN_SAMPEL_STB','CIT_NONCIT')]
df_jr4_pivoted = pd.concat([df11,df22])
# df1x = df1x.reset_index(drop=True).rename_axis(None, axis=1)


# In[126]:


# Read excel T03 (Bppd) 
latest_bppd_files = read_anything(bppd_storage_path)


# In[127]:


latest_bppd_files_2 = df_jr4_pivoted.merge(latest_bppd_files,how='outer',on=['NG','CIT_NONCIT'])


# In[128]:


#merge into df_bppd_user filled 


final_bppd = latest_bppd_files_2.loc[:,('NG','CIT_NONCIT','ANGGARAN_SAMPEL_STB','ANGGARAN_PENDUDUK(BPPD)')]


# In[129]:


#merge with 
df_wpop = df_jr4_new.merge(final_bppd,how='outer',on=['NG','CIT_NONCIT'])


# In[130]:


df_wpop['ID_POP'] = df_wpop['NG'].astype(str) + df_wpop['CIT_NONCIT'].astype(str)


# In[131]:


df_wpop['ID_POP'] = df_wpop['ID_POP'].astype(str).str.replace(r'\.0$', '')


# In[132]:


# Convert the columns to string type
df_wpop['ANGGARAN_PENDUDUK(BPPD)'] = df_wpop['ANGGARAN_PENDUDUK(BPPD)'].astype(str)
df_wpop['ANGGARAN_SAMPEL_STB'] = df_wpop['ANGGARAN_SAMPEL_STB'].astype(str)

# Removing commas from 'ANGGARAN_PENDUDUK(BPPD)' column
df_wpop['ANGGARAN_PENDUDUK(BPPD)'] = df_wpop['ANGGARAN_PENDUDUK(BPPD)'].str.replace(',', '')

# Removing commas from 'ANGGARAN_SAMPEL_STB' column
df_wpop['ANGGARAN_SAMPEL_STB'] = df_wpop['ANGGARAN_SAMPEL_STB'].str.replace(',', '')

# Converting the columns to integers
df_wpop['ANGGARAN_PENDUDUK(BPPD)'] = df_wpop['ANGGARAN_PENDUDUK(BPPD)'].astype(float)


# In[133]:


df_wpop['ANGGARAN_SAMPEL_STB'] = df_wpop['ANGGARAN_SAMPEL_STB'].astype(float)


# Jana Pemberat Final 
# 
# 
# 1. from pop_temp_files , merge with df_jr4_new  based on idpop , to bring weight pop inside 
# 2. generate column pemberat final (weight pop) x pemberat first 


# In[ ]:





# In[134]:


df_wpop['WEIGHT_POP'] = (df_wpop['ANGGARAN_PENDUDUK(BPPD)'])/(df_wpop['ANGGARAN_SAMPEL_STB'])


# In[135]:


#weight pop = anggaran penduduk bppd / anggaran sampel (STB)


# In[136]:


df_wpop['PEMBERAT_FINAL'] = (df_wpop['WEIGHT_POP'] * df_wpop['PEMBERAT_1'])


# In[137]:


years_t01 = years_t01.astype(str)
quarter = quarter_t01.astype(str)


# ### Ingest to DATABASE DIRECTLY 


# In[138]:


schema='production_micro_fc_stb_annually'


# In[139]:


engine = create_engine('postgresql+psycopg2://admin:admin@10.251.49.51:5432/postgres')
connection = engine.connect()
print(connection)


# In[140]:


end_time = time.time() # get the end time 
time_running = end_time - start_time  # Calculate the time difference
minutes = time_running / 60  # Convert time_running to minutes
print(f'it took {minutes} minutes to run the whole process')


# In[141]:


# #Handle duplicates row if run twice 

# WITH duplicates_cte AS (
#     SELECT column1, column2, ..., 
#            ROW_NUMBER() OVER(PARTITION BY column1, column2, ... ORDER BY column1) AS row_num
#     FROM your_table
# )
# DELETE FROM your_table
# WHERE (column1, column2, ...) IN (
#     SELECT column1, column2, ...
#     FROM duplicates_cte
#     WHERE row_num > 1
# );

#Select distinct 
#drop duplicate once query 


# In[142]:


def sanitize_column_name(name):
    # Remove all non-alphanumeric characters except underscores
    name = re.sub(r'\W+', '', name)
    
    # Remove leading digits if present
    name = re.sub(r'^\d+', '', name)
    
    # Ensure the name doesn't start with an underscore
    if name.startswith('_'):
        name = name[1:]
    
    return name


# In[143]:


df_wpop.columns = df_wpop.columns.map(sanitize_column_name)


# In[144]:


df_wpop.to_sql('JR4A'+'Y'+years_t01+'_FC',con=engine,schema=schema,index=False,if_exists='replace')


# ### Remove files in bppd_database path to avoid clash process in future


# In[145]:


#to remove data directly
def clear_garbage(path):
    file_avaialble = [x for x in os.listdir(path)]

    try:
        for x in file_avaialble:
            os.remove(path+'/'+x)
            y = str(x).upper()
            print(f'{y} excess files from processing has been relocated, contact vendor if you require the files for quality check ')
    except Exception as e:
        print(f'Error relocating the files: {path} - {e}')


# In[146]:


#move data to clear 
def mover(path, destination_folder):
    files_available = [x for x in os.listdir(path)]

    try:
        for file_name in files_available:
            source_file = os.path.join(path, file_name)
            destination_file = os.path.join(destination_folder, file_name)
            shutil.move(source_file, destination_file)
            y = str(file_name).upper()
            print(f'{y} excess files from processing has been relocated to {destination_folder}. Contact the vendor if you require the files for quality check.')
    except Exception as e:
        print(f'Error relocating the files: {path} - {e}')


# In[147]:


path = bppd_database
destination_folder = bin_path
mover(path, destination_folder)

path = popfac_path
destination_folder = bin_path
mover(path, destination_folder)

path = t01_path
destination_folder = bin_path
mover(path, destination_folder)



path = bppd_storage_path
destination_folder = bin_path
mover(path, destination_folder)

path = temp_path
destination_folder = bin_path
mover(path, destination_folder)



path = path_t02
destination_folder = bin_path
mover(path, destination_folder)


# In[ ]:




