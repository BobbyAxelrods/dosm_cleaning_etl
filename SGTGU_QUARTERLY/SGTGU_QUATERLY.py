#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import os
import glob
import time
import traceback 
import sys 
import warnings
from sqlalchemy import create_engine, text
import psycopg2
import shutil
warnings.filterwarnings('ignore')
start_time = time.time() # get start time 
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)



#read whatever files types being ingested 
def read_anything(path,num):
    #Get all files avaialble in the path (path shall be only 1 file at a time to manage this )
    get_working_files = [x for x in os.listdir(path)]
#     get_working_files = x
    #make this as a function later 
    if len(get_working_files) == 0:
        print('My goodman, no files either csv nor excel found, please recheck in path the existence')
        return None
    
    #Excel found 
    time_start = time.time()
#     get_types_available = os.path.splitext(get_working_files[0])[1]
#     file_name = os.path.splitext(get_working_files[0])[0]
    for file_name in get_working_files:
        get_types_available = os.path.splitext(file_name)[1]
        if get_types_available.endswith('.xlsx'):
            
            time_start = time.time()
            print('We found excel files, hence we will read it and save to df_master, hold a moment ....')
            df_master = pd.read_excel(path+'/'+file_name,dtype=str,header=num).dropna(how='all')

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
                df_master = pd.read_csv(os.path.join(path,file_name), skip_blank_lines=True,dtype=str,header=num).dropna(how='all')
            except UnicodeDecodeError:
                # If 'utf-8' fails, try 'ISO-8859-1' encoding
                df_master = pd.read_csv(os.path.join(path,file_name), encoding='ISO-8859-1', skip_blank_lines=True,dtype=str,header=num).dropna(how='all')
            int_columns = []
            for col in df_master.columns:
                if df_master[col].notnull().all() and df_master[col].str.isdigit().all():
                    int_columns.append(col)
            df_master[int_columns] = df_master[int_columns].astype(int)
            time_end = time.time()
            diff_time = time_end - time_start
            print(f'My performance reading {file_name} file took : {diff_time} seconds')
            return df_master
        
    print('No suitable files (csv or excel) found in the specified path. Continuing to search...')
    return df_master



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




df = df.fillna(0).astype('int')




#Get the latest user input data numbered by max row 
max_row_num = df['row_num'].max()
max_row = df[df['row_num'] == max_row_num]
#MODULE 0 : QUERY SQL TABLE BASED ON USER INPUT VALUE FROM FRONT END 
QUARTER = max_row['quarter'].values[0]
YEAR = max_row['year'].values[0]



print(f'We are cleaning SGTGU Quarter {QUARTER} & Year {YEAR} Survey Data ')



### PATH DECLARATION (ADJUST FOR HDFS ADN WINDOWS )
#get t01 latest files path
current_path = os.getcwd()
files_input_path_t01 = os.path.join(current_path, 'FILES INPUT', 'T01')
raw_data_t01 = glob.glob(os.path.join(files_input_path_t01,  '*.xlsx'))
get_t01_latest = max(raw_data_t01, key=os.path.getmtime)

# files_input_path_raw = os.path.join(current_path, 'FILES INPUT', 'RAW_DATA')
# raw_data_raw = glob.glob(os.path.join(files_input_path_raw,  '*.xlsx'))
# get_raw_latest = max(raw_data_raw, key=os.path.getmtime)
bin_path =  os.path.join(current_path, 'BIN')

files_input_path_raw = os.path.join(current_path, 'FILES INPUT', 'RAW_DATA')


#set output path 
files_output_path_result = os.path.join(current_path, 'FILES OUTPUT')
files_output_path_result_final = os.path.join(current_path, 'FILES OUTPUT')



#Read all files as per list t01 and structure into proper dataframe and exclude all unecessary row 
#Files that need to be read is in sheet number 2 and start reading it from row 6 to avoid messy structure

#t01 
# df_t01 = pd.read_excel(get_t01_latest,dtype=str,header=5,sheet_name='HEADER_MAIN')
df_t01 = pd.read_excel(get_t01_latest,dtype=str,header=5,sheet_name='HEADER_MAIN')

#Read all files as per list t02 and structure into proper dataframe and exclude all unecessary row 
#t02
# df_t02 = pd.read_excel(get_t01_latest,dtype=str,sheet_name='MSIC2008_MAIN')
df_t02 = pd.read_excel(get_t01_latest,dtype=str,sheet_name='MSIC2008_MAIN')


# In[9]:


df_t01['NEW_ORDER BY VAR'] = df_t01['NEW_ORDER BY VAR'].astype(int)
df_t01_sort = df_t01.sort_values('NEW_ORDER BY VAR')


# In[ ]:





# In[10]:


#Read all files as per list raw and structure into proper dataframe and exclude all unecessary row
#user to confirm file structure in here. and ile format 

df_raw = read_anything(files_input_path_raw,2)




print('Reading files success. We are organizing the header according to the files uploaded')




#compare header between t01 against df_raw  

df_raw_col_list = df_raw.columns.to_list()
df_old_name_list = df_t01_sort['ORI_EJOB HEADER'].to_list()
df_new_name_list = df_t01_sort['NEW_DATASET HEADER'].to_list()




df_raw_rename = df_raw.rename(columns=dict(zip(df_old_name_list,df_new_name_list)))
df_raw_rename_reorg =df_raw_rename.reindex(columns=df_new_name_list)




df_raw_rename_reorg_check = df_raw_rename_reorg.columns.to_list()
result= []
j = zip(df_raw_rename_reorg_check, df_new_name_list)
for x, y in j: 
    result.append(x != y)
    
if any(result):
    print("Columns are not renamed and reorganized properly.Please recheck reference file uploaded ")
else:
    print("Columns are renamed and reorganized successfully according to header references provided.")



print('We are examining MSIC 2008 to fill SECTION_SMPL & SECTION}')



#get the number of column so to rearrange back column from t02 back to raw main dataframe 
section_column_number = df_raw_rename_reorg.columns.get_loc('SECTION')
section_smpl_column_number = df_raw_rename_reorg.columns.get_loc('SECTION_SMPL')
msic2008_column_number = df_raw_rename_reorg.columns.get_loc('MSIC2008')
msic2008_smpl_column_number = df_raw_rename_reorg.columns.get_loc('MSIC2008_SMPL')








## Add portion to create SECTION_SMPL & SECTION




df_t02_copy_1 = df_t02.copy()
df_t02_copy_2 = df_t02.copy()



#Create for SECTION_SMPL & MSIC2008_SMPL for join with df raw and fill up SECTION SMPL column 



df_t02_copy_1 = df_t02_copy_1.rename(columns={'without \'0\'':'MSIC2008_SMPL'})
df_t02_smpl = df_t02_copy_1[['MSIC2008_SMPL','SECTION']]
df_t02_smpl = df_t02_smpl.rename(columns={'SECTION':'SECTION_SMPL'})



output_smpl = df_t02_smpl.columns.to_list
print(f'success in generating specific column {output_smpl} for t02' )



#Create for SECTION & MSIC2008 for join with df raw and fill up SECTION SMPL column 




df_t02_copy_2 = df_t02_copy_2.rename(columns={'without \'0\'':'MSIC2008'})
df_t02_non = df_t02_copy_2[['MSIC2008','SECTION']]




output_non = df_t02_non.columns.to_list
print(f'success in generating specific column {df_t02_non} for t02' )


# In[25]:


#delete SECTION & SECTION_SMPL in df_raw_rename_reorg

df_raw_rename_reorg_dropped_section_smpl = df_raw_rename_reorg.drop(['SECTION','SECTION_SMPL'], axis=1)


# In[26]:


df_final_raw_data_v1 = df_t02_non.merge(df_raw_rename_reorg_dropped_section_smpl, how ='right', on='MSIC2008')


# In[27]:


df_t02_smpl['MSIC2008_SMPL'] = df_t02_smpl['MSIC2008_SMPL'].astype(int)


# In[28]:


df_final_raw_data_v2 = df_t02_smpl.merge(df_final_raw_data_v1, how ='right', on='MSIC2008_SMPL')


# In[29]:


columns_to_move = ['SECTION_SMPL','SECTION','MSIC2008','MSIC2008_SMPL']
new_indexes = [section_smpl_column_number,section_column_number,msic2008_smpl_column_number,msic2008_column_number,]

columns = df_final_raw_data_v2.columns.to_list()

#remove the column in list 
for column in columns_to_move:
    columns.remove(column)
    
for column,index in zip(columns_to_move,new_indexes):
    columns.insert(index,column)

df_final_raw_data = df_final_raw_data_v2.reindex(columns=columns)


# In[30]:


df_final_raw_data.to_csv('testing.csv')


# In[31]:


df_temp = df_final_raw_data[['NEWSSID','REGISTERED_NAME','TRADING_NAME']]
df = df_final_raw_data.copy()
year = df.loc[1,'YEAR']
quarter = df.loc[1,'QUARTER']


# #### MODULE 4 FILL UP STRATA_EMPL COLUMNS ACCORDING TO FILTER OF STRATA_EMPL & SECTION_SMPL 
# 
# IF SECTION_SMPL == C AND F1310 == RANGE(0,5) -> FILL COLUMN STRATA_EMPL= 4
# IF SECTION_SMPL == C AND F1310 == RANGE(5,75) -> FILL COLUMN STRATA_EMPL= 3
# IF SECTION_SMPL == C AND F1310 == RANGE(75,201) -> FILL COLUMN STRATA_EMPL= 2
# IF SECTION_SMPL == C AND F1310 >= 201 -> FILL COLUMN STRATA_EMPL= 1
# 
# IF SECTION_SMPL != C AND F1310 == RANGE(0,5) -> FILL COLUMN STRATA_EMPL= 4
# IF SECTION_SMPL != C AND F1310 == RANGE(5,30) -> FILL COLUMN STRATA_EMPL= 3
# IF SECTION_SMPL != C AND F1310 == RANGE(30,76) -> FILL COLUMN STRATA_EMPL= 2
# IF SECTION_SMPL != C AND F1310 >= 76 -> FILL COLUMN STRATA_EMPL= 1
# 

# In[32]:


#Get dynamic column ending with F1310 
colx = [col for col in df_final_raw_data.columns if 'F1310' in col]


# In[33]:


#update strata_empl value based on filtering of F1310 and SECTION_SMPL

condition_a_1 = (df_final_raw_data['SECTION_SMPL'] == 'C') & (df_final_raw_data[colx[0]] < 5)
condition_a_2 = (df_final_raw_data['SECTION_SMPL'] == 'C') & (df_final_raw_data[colx[0]].isin(range(5,75)))
condition_a_3 = (df_final_raw_data['SECTION_SMPL'] == 'C') & (df_final_raw_data[colx[0]].isin(range(75,201)))
condition_a_4 = (df_final_raw_data['SECTION_SMPL'] == 'C') & (df_final_raw_data[colx[0]] >= 201)

condition_list = [
    condition_a_1,
    condition_a_2,
    condition_a_3,
    condition_a_4
]
default_value = np.nan
choices = [4,3,2,1]
df_final_raw_data['STRATA_EMPL'] = np.select(condition_list,choices, default=default_value)


# In[34]:


#update strata_empl value based on filtering of F1310 and SECTION_SMPL

condition_b_1 = ((df_final_raw_data['SECTION_SMPL'] != 'C') & (df_final_raw_data[colx[0]] < 5)) #& pd.notnull(df_final_raw_data[colx[0]
condition_b_2 = ((df_final_raw_data['SECTION_SMPL'] != 'C') & (df_final_raw_data[colx[0]].isin(range(5,30))))
condition_b_3 = ((df_final_raw_data['SECTION_SMPL'] != 'C') & (df_final_raw_data[colx[0]].isin(range(30,76))))
condition_b_4 = ((df_final_raw_data['SECTION_SMPL'] != 'C') & (df_final_raw_data[colx[0]] >= 76))

condition_list = [
    condition_b_1,
    condition_b_2,
    condition_b_3,
    condition_b_4
]
default_value = np.nan
choices = [4,3,2,1]
df_final_raw_data['STRATA_EMPL'] = np.select(condition_list,choices, default=default_value)


# In[ ]:





# #### MODULE 5 : 
# ADDING LEADING ZERO TO FRONT SUBSTATE CODE TO COMPLETE 2 DIGITS 

# In[35]:


# change all column to 
df_final_raw_data['SUBSTATE_CODE'] = df_final_raw_data['SUBSTATE_CODE'].apply(lambda x: str(int(x)) if pd.notnull(x) and not isinstance(x, str) else '')

# for i, x in enumerate(df_final_raw_data['SUBSTATE_CODE']):
#     if pd.notnull(x):
#         if not isinstance(x, str):
#             df_final_raw_data.loc[i, 'SUBSTATE_CODE'] = str(int(x))
#         else:
#             df_final_raw_data.loc[i, 'SUBSTATE_CODE'] = x
#     else:
#         df_final_raw_data.loc[i, 'SUBSTATE_CODE'] = ''
        
# isinstance(object, type)


# In[36]:


checktypes = df_final_raw_data['SUBSTATE_CODE'].dtype
print(f'Column Substate_code changed to dtypes = {checktypes} for the basis of adding leading 0')


# In[37]:


before = df_final_raw_data['SUBSTATE_CODE'].iloc[1]
print(f'Check does this meet requirement before changing value this is the unique value list: {before}')


# In[38]:


for i, x in enumerate(df_final_raw_data['SUBSTATE_CODE']):
    if x not in ['0','']:
        if len(x) <=2:
            df_final_raw_data.loc[i,'SUBSTATE_CODE']= x.zfill(2)


# In[39]:


after = df_final_raw_data['SUBSTATE_CODE'].iloc[1]
print(f'Check does this meet requirement after changing value this is the unique value list: {after}')


# In[ ]:





# ADDING LEADING ZERO TO FRONT NEWSSID TO COMPLETE 12 DIGITS 

# In[40]:


# # 2 Change flaot to str 
# for i, x in enumerate(df_final_raw_data['NEWSSID']):
#     if pd.notnull(x) and x not in ['0','']:
#         if not isinstance(x,str):
#             df_final_raw_data.loc[i,'NEWSSID'] = str(int(x))
#         else:
#             df_final_raw_data.loc[i,'NEWSSID'] = x
#     else:
#          df_final_raw_data.loc[i, 'SUBSTATE_CODE'] = ''   

df_final_raw_data['NEWSSID'] = df_final_raw_data['NEWSSID'].apply(lambda x: str(int(x)) if pd.notnull(x) and not isinstance(x, str) else '')


# In[41]:


checktypes = df_final_raw_data['NEWSSID'].dtype
print(f'Column NEWSSID changed to dtypes = {checktypes} for the basis of adding leading 0')


# In[42]:


before = df_final_raw_data['NEWSSID'].iloc[1]
print(f'Check for NEWSSID does this meet requirement before changing value this is the unique value list: {before}')


# In[43]:


for i, x in enumerate(df_final_raw_data['NEWSSID']):
    if x not in ['','0']:
        if len(x) <= 12 :
            df_final_raw_data.loc[i,'NEWSSID'] = x.zfill(12)


# In[44]:


after = df_final_raw_data['NEWSSID'].iloc[1]

print(f'Check for NEWSSID does this meet requirement before changing value this is the unique value list: {after}')


# In[ ]:





# ADDING LEADING ZERO TO FRONT MSIC2008_SMPL TO COMPLETE 5 DIGITS 

# In[45]:


df_final_raw_data['MSIC2008_SMPL'] = df_final_raw_data['MSIC2008_SMPL'].apply(lambda x: str(int(x)) if pd.notnull(x) and not isinstance(x, str) else '')
checktypes = df_final_raw_data['MSIC2008_SMPL'].dtype
print(f'Column MSIC2008_SMPL changed to dtypes = {checktypes} for the basis of adding leading 0')


# In[46]:


before = df_final_raw_data['MSIC2008_SMPL'].iloc[1]
print(f'Check for MSIC2008_SMPL does this meet requirement before changing value this is the unique value list: {before}')


# In[47]:


for i, x in enumerate(df_final_raw_data['MSIC2008_SMPL']):
    if x not in ['','0']:
        if len(x) <= 5 :
            df_final_raw_data.loc[i,'MSIC2008_SMPL'] = x.zfill(12)


# In[48]:


after = df_final_raw_data['MSIC2008_SMPL'].iloc[1]

print(f'Check for MSIC2008_SMPL does this meet requirement before changing value this is the unique value list: {after}')


# In[ ]:





# ADDING LEADING ZERO TO FRONT MSIC2008 TO COMPLETE 5 DIGITS 

# In[49]:


df_final_raw_data['MSIC2008'] = df_final_raw_data['MSIC2008'].apply(lambda x: str(int(x)) if pd.notnull(x) and not isinstance(x, str) else '')
checktypes = df_final_raw_data['MSIC2008'].dtype
print(f'Column MSIC2008 changed to dtypes = {checktypes} for the basis of adding leading 0')


# In[50]:


before = df_final_raw_data['MSIC2008'].iloc[1]
print(f'Check for MSIC2008 does this meet requirement before changing value this is the unique value list: {before}')


# In[51]:


for i, x in enumerate(df_final_raw_data['MSIC2008']):
    if x not in ['','0']:
        if len(x) <= 5 :
            df_final_raw_data.loc[i,'MSIC2008'] = x.zfill(12)


# In[52]:


after = df_final_raw_data['MSIC2008'].iloc[1]

print(f'Check for one of the value from MSIC2008, does this meet requirement before changing value this is the unique value list: {after}')


# In[ ]:





# ADDING LEADING ZERO TO FRONT STATE_CODE TO COMPLETE 5 DIGITS 

# In[53]:


df_final_raw_data['STATE_CODE'] = df_final_raw_data['STATE_CODE'].apply(lambda x: str(int(x)) if pd.notnull(x) and not isinstance(x, str) else '')
checktypes = df_final_raw_data['STATE_CODE'].dtype
print(f'Column STATE_CODE changed to dtypes = {checktypes} for the basis of adding leading 0')


# In[54]:


before = df_final_raw_data['STATE_CODE'].iloc[1]
print(f'Check for STATE_CODE does this meet requirement before changing value this is the unique value list: {before}')


# In[55]:


for i, x in enumerate(df_final_raw_data['STATE_CODE']):
    if x not in ['','0']:
        if len(x) <= 5 :
            df_final_raw_data.loc[i,'STATE_CODE'] = x.zfill(2)


# In[56]:


after = df_final_raw_data['STATE_CODE'].iloc[1]

print(f'Check for one of the value from STATE_CODE, does this meet requirement before changing value this is the unique value list: {after}')


# In[ ]:





# ### Module 7 : Summation all category (1-9) for summation of total salaries & wages (Q=L+M+N+O+P)

# In[57]:


#Get the starting index of column F0101 in col 
coly = [col for col in df_final_raw_data.columns if 'F0101' in col]
first_index = df_final_raw_data.columns.get_loc(coly[0])


# In[58]:


vab_list_mod_8 =  [ 'L', 'M', 'N', 'O', 'P','Q']
#Select column name to manipulate from master dataframe 
selection_col_2 = []
for x in df_final_raw_data.columns[first_index:]:
    if x[:1] in vab_list_mod_8:
        selection_col_2.append(x)


# In[59]:


#assign selection for module 7 into 

df_mod7_1=df_final_raw_data[selection_col_2]


# In[60]:


#wrangle the data 
#1. List all column from filtered to be appended in list_j
list_j = []
for x in df_mod7_1.columns:
    y = x[:6]
    list_j.append(y)
#2. Get the unique list 
list_j_unique = list(set(list_j))  

#3. Get the unique sorted alphabetically 
sort_alph = ['L','M','N','O','P','Q']
#Sort the list from unique list into separated set of list to make it easier during loop
# sample output a = ['L23204', 'M23204', 'N23204', 'O23204', 'P23204', 'Q23204']
list_j_sort = sorted(list_j_unique,key=lambda x:(len(x),x[0]))


print(f'{list_j_sort} are the unique partial front string to be re arranged before filtering from main dataframe process')

#make a dictionary from sortation filter so we can filter based on variables 

group_dict = {}
for col_name in list_j_sort:
    end = col_name[-2:]
#if column name existing int group_dictionary, then append in created list
    if end in group_dict:
        group_dict[end].append(col_name)
        
#if dont have , then create a new one. 
    else: 
        group_dict[end] = [col_name]

dynamic_keys = sorted(group_dict.keys())

print(f'{dynamic_keys} are unique partial string sorted by quarters ascending ')


# In[61]:


# #Strategy 
# Loop through column_no_q and column_w_q to filter columns based on the conditions.
# Change the data type of columns in df[sum_this] and df[paste_here] to numeric and fill any missing values with 0.
# Calculate the sum of each row in df[sum_this] and assign the result to sum_val.
# Reshape sum_val to a column vector and assign it to df_output[paste_here].
# Change the data type of columns in df_output[sum_this] and df_output[paste_here] to numeric and fill any missing values with 0.
# Calculate the sum before the arithmetic expression for df_output[paste_here] and df_output[sum_this] using sum(axis=0) and store them in result_mod7_before and input_mod7_before, respectively.
# Calculate the sum after the amendment for df_output[paste_here] and df_output[sum_this] using sum(axis=0) and store them in result_mod7_after and input_mod7_after, respectively.
# Compare result_mod7_after[0] with input_mod7_after to check if the results match and print the corresponding message.


# In[62]:


# a function to filter data by partial string match and sum based on conditional 
def process_dataframe(df,group_num,grouping_1,df_output):
    sum_this = []
    paste_here = []
    listx = group_dict[group_num]
    exclude_q = [x for x in listx if 'Q' not in x]
    q_only = [x for x in  listx if 'Q' in x]
    # get the list of column available in df to separate 
    column_no_q = [x for x in df if any(q in x for q in exclude_q)]
    column_w_q = [x for x in df if any(q in x for q in q_only)]
    
    #Get the list of column that is in list 
    #The sample of dictionary as follows : - 

    # {'04': ['L23204', 'M23204', 'N23204', 'O23204', 'P23204', 'Q23204'],
    #  '06': ['L23206', 'M23206', 'N23206', 'O23206', 'P23206', 'Q23206'],
    #  '05': ['L23205', 'M23205', 'N23205', 'O23205', 'P23205', 'Q23205']}

    # empty frame to insert value for each function run temporary to paste value in df main 

    #loop to filter from df
    for x in column_no_q:
        for y in group_dict[group_num]:
            if x.endswith(grouping_1) and y in x:
                sum_this.append(x)

    for x in column_w_q:
        for y in group_dict[group_num]:
            if x.endswith(grouping_1) and y in x:
                paste_here.append(x)
                
    #change datatype from string to int and sum value from list of sumthis to get the overall value  
    df[sum_this] = df[sum_this].apply(pd.to_numeric, errors='coerce')
    df[sum_this] = df[sum_this].fillna(0).astype(int)
    df[paste_here] = df[paste_here].apply(pd.to_numeric, errors='coerce')
    df[paste_here] = df[paste_here].fillna(0).astype(int)
    df_output[sum_this] = df_output[sum_this].apply(pd.to_numeric, errors='coerce')
    df_output[sum_this] = df_output[sum_this].fillna(0).astype(int)
    df_output[paste_here] = df_output[paste_here].apply(pd.to_numeric, errors='coerce')
    df_output[paste_here] = df_output[paste_here].fillna(0).astype(int)

    #get the aggreagate of all column in sum_this
    sum_val = df[sum_this].sum(axis=1)
    #paste the aggregate in column QXXX and ending with group_num in df_final_raw as final value 
    df_output[paste_here] = sum_val.values.reshape(-1, 1)

    #get the column Q in 01 and sum the value column first from main dataframe 

    #change the dtyps for section affected only from df main so we can sum this 

    #check & test 
    #Get the sum before arimethic expression for df_main as initial value 
    result_mod7_before = df_output[paste_here].sum(axis=0)
    input_mod7_before = df_output[sum_this].sum(axis=0)
    input_mod7_before = input_mod7_before.sum()
    
    #get the sum of column after in df_mod7 and after ammendment made in 
    result_mod7_after = df_output[paste_here].sum(axis=0)
    input_mod7_after = df_output[sum_this].sum(axis=0)
    input_mod7_after = input_mod7_after.sum()

    if result_mod7_after[0] == input_mod7_after:
        print(f'{paste_here} successfully aggregated & match with source value which initially {result_mod7_before[0]} turned to {result_mod7_after[0]}')
    else:
        print(f'{paste_here} warning, result and source do not match, recheck data')


# In[63]:


#group num is based on dynamic_keys = ['04', '05', '06'] position which is dunamically based on quarter.
# sum the value based on pulled dataframe and paste the value into df_main into as ouput 
grouping_1 = ('1','2','3','4','5','6','7','8','9','0')
group_num = dynamic_keys[0]
df = df_mod7_1
df_output = df_final_raw_data

for x in grouping_1:
    process_dataframe(df,group_num,x,df_output)


# In[64]:


#group num is based on dynamic_keys = ['04', '05', '06'] position which is dunamically based on quarter.
# sum the value based on pulled dataframe and paste the value into df_main into as ouput 
grouping_1 = ('1','2','3','4','5','6','7','8','9','0')
group_num = dynamic_keys[1]
df = df_mod7_1
df_output = df_final_raw_data

for x in grouping_1:
    process_dataframe(df,group_num,x,df_output)


# In[65]:


#group num is based on dynamic_keys = ['04', '05', '06'] position which is dunamically based on quarter.
# sum the value based on pulled dataframe and paste the value into df_main into as ouput 
grouping_1 = ('1','2','3','4','5','6','7','8','9','0')
group_num = dynamic_keys[2]
df = df_mod7_1
df_output = df_final_raw_data

for x in grouping_1:
    process_dataframe(df,group_num,x,df_output)


# In[ ]:





# ### Module 8 : Summation all category (1-9) for summation of total separation (X=D+E+F)

# In[66]:


vab_list_mod_9 =  ['D','E','F','X']
#Select column name to manipulate from master dataframe 
selection_col_3 = []
for x in df_final_raw_data.columns[first_index:]:
    if x[:1] in vab_list_mod_9:
        selection_col_3.append(x)


# In[67]:


#filter the column out from main_df 
df_vabs_mod9= df_final_raw_data[selection_col_3]

#wrangle the data 
#1. List all column from filtered to be appended in list_j
list_j = []
for x in df_vabs_mod9.columns:
    y = x[:6]
    list_j.append(y)
#2. Get the unique list 
list_j_unique = list(set(list_j))  

#3. Get the unique sorted alphabetically 
sort_alph = ['D','E','F','X']
#Sort the list from unique list into separated set of list to make it easier during loop
# sample output a = ['L23204', 'M23204', 'N23204', 'O23204', 'P23204', 'Q23204']
list_j_sort = sorted(list_j_unique,key=lambda x:(len(x),x[0]))


print(f'{list_j_sort} are the unique partial front string to be re arranged before filtering from main dataframe process')

#make a dictionary from sortation filter so we can filter based on variables 

group_dict = {}
for col_name in list_j_sort:
    end = col_name[-2:]
#if column name existing int group_dictionary, then append in created list
    if end in group_dict:
        group_dict[end].append(col_name)
        
#if dont have , then create a new one. 
    else: 
        group_dict[end] = [col_name]

dynamic_keys = sorted(group_dict.keys())

print(f'{dynamic_keys} are unique partial string sorted by quarters ascending ')


# In[68]:


# a function to filter data by partial string match and sum based on conditional 
def process_dataframe_2(df,group_num,grouping_1,df_output):
    sum_this = []
    paste_here = []
    listx = group_dict[group_num]
    exclude_q = [x for x in listx if 'X' not in x]
    q_only = [x for x in  listx if 'X' in x]
    # get the list of column available in df to separate 
    column_no_q = [x for x in df if any(q in x for q in exclude_q)]
    column_w_q = [x for x in df if any(q in x for q in q_only)]
    
    #Get the list of column that is in list 
    #The sample of dictionary as follows : - 

    # {'04': ['L23204', 'M23204', 'N23204', 'O23204', 'P23204', 'Q23204'],
    #  '06': ['L23206', 'M23206', 'N23206', 'O23206', 'P23206', 'Q23206'],
    #  '05': ['L23205', 'M23205', 'N23205', 'O23205', 'P23205', 'Q23205']}

    # empty frame to insert value for each function run temporary to paste value in df main 

    #loop to filter from df
    for x in column_no_q:
        for y in group_dict[group_num]:
            if x.endswith(grouping_1) and y in x:
                sum_this.append(x)

    for x in column_w_q:
        for y in group_dict[group_num]:
            if x.endswith(grouping_1) and y in x:
                paste_here.append(x)
                
    #change datatype from string to int and sum value from list of sumthis to get the overall value  
    df[sum_this] = df[sum_this].apply(pd.to_numeric, errors='coerce')
    df[sum_this] = df[sum_this].fillna(0).astype(int)
    df[paste_here] = df[paste_here].apply(pd.to_numeric, errors='coerce')
    df[paste_here] = df[paste_here].fillna(0).astype(int)
    df_output[sum_this] = df_output[sum_this].apply(pd.to_numeric, errors='coerce')
    df_output[sum_this] = df_output[sum_this].fillna(0).astype(int)
    df_output[paste_here] = df_output[paste_here].apply(pd.to_numeric, errors='coerce')
    df_output[paste_here] = df_output[paste_here].fillna(0).astype(int)

    #get the aggreagate of all column in sum_this
    sum_val = df[sum_this].sum(axis=1)
    
    #paste the aggregate in column QXXX and ending with group_num in df_final_raw as final value 
    df_output[paste_here] = sum_val.values.reshape(-1, 1)
    
#     sum_val_df = pd.DataFrame({col: sum_val for col in paste_here})

#     df_output[paste_here] = sum_val_df

    #get the column Q in 01 and sum the value column first from main dataframe 

    #change the dtyps for section affected only from df main so we can sum this 

    #check & test 
    #Get the sum before arimethic expression for df_main as initial value 
    result_mod7_before = df_output[paste_here].sum(axis=0)
    input_mod7_before = df_output[sum_this].sum(axis=0)
    input_mod7_before = input_mod7_before.sum()
    
    #get the sum of column after in df_mod7 and after ammendment made in 
    result_mod7_after = df_output[paste_here].sum(axis=0)
    input_mod7_after = df_output[sum_this].sum(axis=0)
    input_mod7_after = input_mod7_after.sum()

    if result_mod7_after[0] == input_mod7_after:
        print(f'{paste_here} successfully aggregated & match with source value which initially {result_mod7_before[0]} turned to {result_mod7_after[0]}')
    else:
        print(f'{paste_here} warning, result and source do not match, recheck data')


# In[69]:


#group num is based on dynamic_keys = ['04', '05', '06'] position which is dunamically based on quarter.
# sum the value based on pulled dataframe and paste the value into df_main into as ouput 
grouping_1 = ('1','2','3','4','5','6','7','8','9','0')
group_num = dynamic_keys[0]
df = df_vabs_mod9
df_output = df_final_raw_data

for x in grouping_1:
    process_dataframe_2(df,group_num,x,df_output)


# In[70]:


#group num is based on dynamic_keys = ['04', '05', '06'] position which is dunamically based on quarter.
# sum the value based on pulled dataframe and paste the value into df_main into as ouput 
grouping_1 = ('1','2','3','4','5','6','7','8','9','0')
group_num = dynamic_keys[1]
df = df_vabs_mod9
df_output = df_final_raw_data

for x in grouping_1:
    process_dataframe_2(df,group_num,x,df_output)


# In[71]:


#group num is based on dynamic_keys = ['04', '05', '06'] position which is dunamically based on quarter.
# sum the value based on pulled dataframe and paste the value into df_main into as ouput 
grouping_1 = ('1','2','3','4','5','6','7','8','9','0')
group_num = dynamic_keys[2]
df = df_vabs_mod9
df_output = df_final_raw_data

for x in grouping_1:
    process_dataframe_2(df,group_num,x,df_output)


# In[ ]:





# ### Module 9. Summation all category (1-9) for variable A, B, C, D, E, F, G, H, I, J, L, M, N, O, P, R & X 

# In[72]:


#Plan 
# Refer based on upload files header 
# identify starting and ending of soalan baru bertambah 
# Filter out column for variable in list 

# Variables involved A, B, C, D, E, F, G, H, I, J, L, M, N, O, P, R & X
# Range Criteria (SUM of 1-9) = 10 


# from main df -> extract to variable that involve only (df_vabs)
# from df_vabs run the condition required and replace sum value into column 10 
# from df_vabs extract only column 10 into df_sum
# replace the value for all rows that have the same column name through looping, dont merge 




# In[73]:


#Strategy New 
# Filter affected column based on alphabet inclusive of ammended column X & Q which in module 7 & 8 
# Filter partial string that is unique 

#1. Filter list of column based on variables first 
vab_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'O', 'P','Q', 'R ','X']
selection_col = ['NEWSSID']
for x in df_final_raw_data.columns[first_index:]:

    if x[:1] in vab_list:
        selection_col.append(x)

# print(f'This are columns selected {selection_col} for aggreagation process')



df_vabs = df_final_raw_data[selection_col]

#1. List all column from filtered to be appended in list_j
list_j = []
for x in df_vabs.columns:
    y = x[:6]
    list_j.append(y)
    
#2. Get the unique list 
list_j_unique = list(set(list_j))  

#3. Get the unique sorted alphabetically 
sort_alph = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'O', 'P','Q', 'R ','X']
#Sort the list from unique list into separated set of list to make it easier during loop
# sample output a = ['L23204', 'M23204', 'N23204', 'O23204', 'P23204', 'Q23204']
list_j_sort = sorted(list_j_unique,key=lambda x:(len(x),x[0]))


print(f'{list_j_sort} are the unique partial front string to be re arranged before filtering from main dataframe process')

#make a dictionary from sortation filter so we can filter based on variables 

group_dict = {}
for col_name in list_j_sort:
    end = col_name[-2:]
#if column name existing int group_dictionary, then append in created list
    if end in group_dict:
        group_dict[end].append(col_name)
        
#if dont have , then create a new one. 
    else: 
        group_dict[end] = [col_name]

dynamic_keys = sorted(group_dict.keys())

print(f'{dynamic_keys} are unique partial string sorted by quarters ascending ')


# In[74]:


#From unique dynamic keys ['04', '05', '06', 'SI'] are unique partial string sorted by quarters ascending 
#dynamic_keys
#From ['A23204', 'A23206', 'A23205', 'B23205', 'B23206', 'B23204', 'C23205', 'C23206', 'C23204', 'D23206', 'D23205', 'D23204', 'E23206', 'E23204', 'E23205', 'F23204', 'F23205', 'F23206', 'G23205', 'G23206', 'G23204', 'H23204', 'H23205', 'H23206', 'I23206', 'I23204', 'I23205', 'J23205', 'J23204', 'J23206', 'L23206', 'L23204', 'L23205', 'M23205', 'M23206', 'M23204', 'NEWSSI', 'N23204', 'N23205', 'N23206', 'O23206', 'O23204', 'O23205', 'P23204', 'P23205', 'P23206', 'Q23206', 'Q23204', 'Q23205', 'X23205', 'X23206', 'X23204'] are the unique partial front string to be re arranged before filtering from main dataframe process
#group_dict

#Loop based on groupdict[0 to max index]
# filter column ending with 01 to 09 = col_sum_0109
# filter column ending with 10 = col_paste_here_10 
# Axxx04_xxxx01 to Axxx04_xxxx09 .sum(axis=1) = sum_val 
# sum_val



# In[75]:


# Function
def aggregate_data(df, df_output, group_num):
    z = [x for x in df if any(q in x for q in y)]
    ending_to_sum = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
    ending_sum = ('0')

    listx = group_dict[group_num]
    total_items = len(listx)

    #from listx
    listx = group_dict[group_num]
    total_items = len(listx)
    # now we want to loop according to sequence A (01-09).sum()
    for n in range(total_items):
        listx = group_dict[group_num][n]
        for x in listx:
            ending_to_sum_col = []
            ending_sum_col = []
            # tosum_filter_list = [value for value in z if any(value.startswith(prefix) for prefix in listx) and value.endswith(ending_to_sum)]
            tosum_filter_list = [value for value in z if value.startswith(listx) and value.endswith(ending_to_sum)]
            sum_column = [value for value in z if value.startswith(listx) and value.endswith(ending_sum)]

            # append in column temporary for we sum it in main_df 
            ending_to_sum_col.append(tosum_filter_list)
            ending_sum_col.append(sum_column)

    #         ending_to_sum_col = ending_to_sum_col[0]
    #         ending_sum_col = ending_sum_col[0]
            ending_to_sum_col = ending_to_sum_col[0] if ending_to_sum_col else None
            ending_sum_col = ending_sum_col[0] if ending_sum_col else None
            # in the loop according to list listx A until X we want to get sum and paste the value directly main_df or output_df 
            df_output[ending_to_sum_col] = df_output[ending_to_sum_col].apply(pd.to_numeric, errors='coerce')
            df_output[ending_to_sum_col] = df_output[ending_to_sum_col].fillna(0).astype(int)

            df_output[ending_sum_col] = df_output[ending_sum_col].apply(pd.to_numeric, errors='coerce')
            df_output[ending_sum_col] = df_output[ending_sum_col].fillna(0).astype(int)

            #recheck before sum 

            result_mod7_before = df_output[ending_sum_col].sum(axis=0)
            input_mod7_before = df_output[ending_to_sum_col].sum(axis=0)
            input_mod7_before = input_mod7_before.sum()

            sum_1to9 = df_output[ending_to_sum_col].sum(axis=1)
            df_output[ending_sum_col] = sum_1to9.values.reshape(-1, 1)
    #         recheck main_df after sum

            result_mod7_after = df_output[ending_sum_col].sum(axis=0)
            input_mod7_after = df_output[ending_to_sum_col].sum(axis=0)
            input_mod7_after = input_mod7_after.sum()

        if result_mod7_after[0] == input_mod7_after:
            print(f'{ending_sum_col} successfully aggregated & match with source value which initially source : {result_mod7_before[0]} from {input_mod7_before} &  result : {result_mod7_after[0]} from {input_mod7_after}')
        else:
            print(f'{ending_sum_col} warning, result and source do not match, recheck data')



# In[ ]:





# In[76]:


# variables list 
df = df_vabs
df_output = df_final_raw_data
group_num = dynamic_keys[0]
aggregate_data(df, df_output, group_num)


# In[77]:


# variables list 
df = df_vabs
df_output = df_final_raw_data
group_num = dynamic_keys[1]
aggregate_data(df, df_output, group_num)


# In[78]:


# variables list 
df = df_vabs
df_output = df_final_raw_data
group_num = dynamic_keys[2]
aggregate_data(df, df_output, group_num)


# In[79]:


Q = str(df_final_raw_data['QUARTER'].unique()[0])
Y = str(df_final_raw_data['YEAR'].unique()[0])


# In[80]:


engine = create_engine('postgresql+psycopg2://admin:admin@10.251.49.51:5432/postgres')
connection = engine.connect()
print(connection)


# In[81]:


schema='production_micro_frd_sgtgu_quarterly'


# In[82]:


# df_final_raw_data.to_excel(files_output_path_result_final+'FRD_'+Q+'_'+Y'.xlsx')
df_final_raw_data.to_sql(f'FRDQ{Q}Y{Y}',con=engine,schema=schema,if_exists='replace',index=False)


# In[83]:


print(f'FRDQ{Q}Y{Y} Ingested into Database')


# In[84]:


end_time = time.time() # get the end time 
time_running = end_time - start_time  # Calculate the time difference
minutes = time_running / 60  # Convert time_running to minutes
print(f'it took {minutes} minutes to run the whole process')


# In[85]:


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


# In[86]:


path = files_input_path_t01
destination_folder = bin_path
mover(path, destination_folder)

path = files_input_path_raw
destination_folder = bin_path
mover(path, destination_folder)



path = files_output_path_result
destination_folder = bin_path
mover(path, destination_folder)


# In[ ]:




