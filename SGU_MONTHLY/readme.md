## PRODUCTION_CLEANING SGU MONTHLY

### INSTRUCTION 
** PLEASE DO REMOVE .gitignore when running the script or else it will be buggy  . It was there to allow folder to be included during push in github.

<!-- 
## ARIF SHALL CLONE THIS REPO 

1. git clone <url> to local host 
<!-- 1. Copy whole directories into local machine together with files 
2. Adjust the PATH_DECLARATION in [STB_ANNUAL_SCRIPT](/STB_QUARTERLY.py) , run forticlient, adjust the database host IP, schema_name
3. Prepare all files dependency and place the files in directory provided below such as 
- [T01 template (user upload template files](./INPUT_T01) 
- [T02 template (user upload template files](./INPUT_T02)
- [T03 template (user upload template files](./INPUT_T03)
- [RAW_DATA_JR4](./INPUT_RAWDATA_STB_JR4)
- [RAW_DATA_JR2](./INPUT_RAWDATA_STB_JR2)
- [RIN_STRATA MAPPING Template](./INPUT_MAP_RINSTRATA)
- [BIN folder (to relocate used files and clear working folder) ](./BIN) --> -->

* All of the test files sources can be found here in [here](https://drive.google.com/drive/folders/1oSKHtxLEUstAJexxW-gQdXz2cZ_ngk66?usp=drive_link)


### STEPS 

1. Read JR5 raw data (rule: column NG, DP, DB, BP, BP2, Converted BP, ST, NOTK, NOIR, S, NP, PKIS, HMIS, J, KET and B as string)
2. Set selected columns as string with specified len: 2, 2, 3, 3, 3, 3, 1, 4, 2, 1, 3, 2, 2, 1, 4, 2
3. Create 38 digit ID by concanate process with the selected columns by given order
4. Clear whitespace and uppercase NAMA column
5. Repeat step 1, 2, 3 and 4 for JR4 final
6. Read all 38 digit ID and name in JR5
7. Find corresponding 38 digit ID from JR5 in JR4 and extract Pemberat Final into JR5
8. If 38 digit in JR5 does not found any match, then will extract Pemberat Final by corresponding name
9. If not found through name and 38 digit ID, flagged (Preferable to be shown to user)(Show: Year, Month, Name, 38 Digit ID)
10. Detect for any duplication (If needed)
11. Extract JR5 with Pemberat Final as FC
