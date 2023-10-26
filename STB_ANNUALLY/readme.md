## PRODUCTION_CLEANING STB ANNUALLY 

### INSTRUCTION 
** PLEASE DO REMOVE .gitignore when running the script or else it will be buggy  . It was there to allow folder to be included during push in github. 

1. Copy whole directories into local machine together with files 
2. Adjust the PATH_DECLARATION in [STB_ANNUAL_SCRIPT](/STB_QUARTERLY.py) , run forticlient, adjust the database host IP, schema_name
3. Prepare all files dependency and place the files in directory provided below such as 
- [T01 template (user upload template files](./INPUT_T01) 
- [T02 template (user upload template files](./INPUT_T02)
- [T03 template (user upload template files](./INPUT_T03)
- [RAW_DATA_JR4](./INPUT_RAWDATA_STB_JR4)
- [RAW_DATA_JR2](./INPUT_RAWDATA_STB_JR2)
- [RIN_STRATA MAPPING Template](./INPUT_MAP_RINSTRATA)
- [BIN folder (to relocate used files and clear working folder) ](./BIN)

* All of the test files sources can be found here in [here](https://drive.google.com/drive/folders/1KLO2sNB6C4ADaL8Hrq5WUYAE2eYjoXiG?usp=sharing)

### SCRIPT CLEANING DETAILS

![Alt text](<Untitled (1)-1.png>)

### ADDITIONAL STEPS TAKEN SPECIFICALLY FOR QUARTER PROCESS 

- USER INPUT TO PROCESS 
	- HAFLYEAR 
	- YEAR 

`INSERT INTO reference_data."USER_INPUT" ("year") VALUES(2021);`

- SCRIPT WILL QUERY IN DB FROM USER LATEST FILLED INFORMATION

- SCRIPT WILL SEARCH UNDER DATA FINAL SCHEMA TO FIND TABLE RELATED MONTHS FOR QUARTER & YEAR FILLED BY USER -> STORE IN A LIST THE 3 TABLE NAME 

- SCRIPT WILL STANDARDIZE THE COLUMN NAME BEFORE CONCAT INTO 1 DATAFRAME

- SCRIPT WILL GENERATE 1 DATAFRAME 

- SCRIPT WILL HIGHLIGHT WHAT COLUMN THAT IS NOT INTERSECT BETWEEN EACH OTHER FOR USER INFORMATION 

- SCRITP WILL RUN COMMON STEPS AS FOLLOWS 

** STEPS TAKEN ALMOST THE SAME WITH STB MONTHLY 

### COMMON STEPS

1. READ FILE JR4
2. BAHAGIKAN DATA KEPADA 7 BAHAGIAN (CATEGORY BARU) 
- CATEGORY A DALAM JANGKA UMUR SETIAP 5 TAHUN
	
	- GROUP BY DALAM UMUR 5 TAHUN 
	- GROUP BY DALAM UMUR 10 TAHUN 
	- GROUP BY DALAM ETNIK SEMENANJUNG 
	- GROUP BY ETNIK SABAH (BUMIPUTRA SABAH)
	- GROUP BY ETNIK SARAWAK 
	- GROUP BY CIT_NONCIT 
	- GROUP BY RIN_STRATA 

** Build a new column based on criteria and fill the value based on group (1-10) 

3. CHECK DUPLICATE VARIBALE 
- B, NG, DP, DB, BP, BP2, CBP, ST, NOTK, NOIR, S dan NP USING PRIMARYFIRST1 
- mark primary first the duplicates and count the first duplicates 
- Create new column for each variables and fill in value 
	- Primary first =1 
	- Secondary = 0 
- Extract the value of this primary key into a new df based on NG, COL FILTERED AND THE PRIMARY KEY AS A REPORT 

4. Adjusted weight calculation 

    4.1 Calculate Bil Respon

    * Build 1 template for user to fill in and csv and fill in the values.(tahunan) 
    * System will extract the value as per same format 
    - It will calculate from JR4 dataframe and it will extract portion of data only:- 
        - Calculate respon using pivot method whereby:- 
            - col = RIN_STRATA 
            - row = NG 
            - values = count primary first 
            bring the value into the template format based on KOD negeri 

    * It must include 2 category (1 bandar & 2 luar bandar) 
    * count 1 & 2 (bandar & luar bandar)
    * get the value as per the negeri and pecah into 1 & 2 -- refer as per template 
    * stored in a new dataframe

    4.2 Calculate the value adjusted weight 
    
    * Adjusted Weight = Data Bilangan isi rumah Tahunan Setiap Negeri / Bil IR Respon Selesai
    
    * Extract into new column 

    4.3 Semakan adjusted weight (comparison) compare adjusted weight from BMP 
    
    * User will upload this value into the template as above
    * in sytem kita calculate (AW(calc) - AW(BMP - user fill)) 

    * If 0 correct , and if got >< 0 -> Alert user that AW contain discrepancies 

4.3 Masukkan adjusted weight calculated in template dataframe into main dataframe 

* Masukkan AW into all 7 grouped dataframe based on KOD NEGERI 
* Use plain extraction from template to main 7 dataframe 

4.4 Semakan adjusted weight dalam main dataframe 

* semakk if AW column in new dataframe contain null values 

* if yes alert users. 

5. POP_FAC CALCULATION SECTION

* MBLS WILL FILL IN FORM BPPD 



5.1 From template JADUAL A1 (DF A1)

* standby to extract value into trend "trend" dataframe 

* From template "TREND" (DF T) 
-BPPD PIC will fill in the form 4 set of templates 
	- SEMENANJUNG 
	- SABAH 
	- BUKAN WARGA 
	- SARAWAK 
* Simplify template 

* CALCULATION TREND (DF CT) 

- From template Jadual A1 & Trend , extract value into a new dataframe it merge for further calculation 

- Once merged, add bukan warganegara for that negeri  calculate 
[Calculation Process] 

- check if blank in jadual A1 (from 1 to 7) 

- if contain blank, then sum the value from BPPD that is blank -> store the sum value in a new variable A

- sum overall value from BPPD and store the value in a new variable B
- C = A - B 
- D = [+A/B]*100
- If D >= 1 , USE ADD 
EXPRESSION to calculate F : 
	- In new column beside BPPD , add (BPPD_1+(BPPD_1/C)*A) and round to no decimal point
	- do for all value from 1 to 7 for each kumpulan umur except if blank value exist  
else 
	IGNORE 

Do this for Male & Female
Do this for all Negeri 

- Focus on F value for each etnik (1-7) 

5.2 POPULATION FACTOR MERGING (G) 

- add new column PF_(1 to 7) where the formula is :
	- A1_X /BPPD_X (X = (1 to 7) 
	- IF value A1_X is blank, ignore. 

5.3 POPULATION KEY IN POP_FAC 
- Create new column PK_X (X = 1 to 7) 
- ROUND THE VALUE OF G to 2 decimal point 

** CONTINUE TO DO FOR ALL 4 GROUP 

5.4 EXTRACT PK_X _ BY NEGERI INTO A NEW DATAFRAME = POP FAC VALUE 

* DATA
COLUMN( Kumpulan Umur | Male | Female | Non Citizen )



5.4 Create new df_jr4 new dataframe 
* revised popfac and split into 3 dataframe non cit , cit 
* add KU-5 Column , each 18 row (1-18) 
* add etnik 1 until 5 for each 18 row counts 
* Convert column female stacked under male column and add new column to label male and female 
	- split column male and female 
	- add column to each dataframe male & female 
	- merge column male female into 1 dataframe 
* rename column 1 into J 
* split non citizen column to a new data frame 

5.5 Merge new df_jr4 & revised popfac for all NG
- based on G3, NG, J, KU-5