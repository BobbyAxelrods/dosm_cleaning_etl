## PRODUCTION_CLEANING SGTGU QUARTERLY 

### INSTRUCTION 
** PLEASE DO REMOVE .gitignore when running the script or else it will be buggy  . It was there to allow folder to be included during push in github. 

1. Copy whole directories into local machine together with files 
2. Adjust the PATH_DECLARATION in [SGTGU SCRIPT](/SGTGU_QUATERLY.py) , run forticlient, adjust the database host IP, schema_name
3. Prepare all files dependency and place the files in directory provided below such as 
- [T01_template](./FILES_INPUT\T01) 
- [RAW_DATA_EJOB](./FILES_INPUT/RAW_DATA)
- [BIN folder (to relocate used files and clear working folder) ](./BIN)
* All of the test files sources can be found here in [here](https://drive.google.com/drive/folders/16SrTkh8FowwR6VQHvTy0NGhOWw-0L3Hn?usp=sharing)

### SCRIPT CLEANING DETAILS
[Click to view detail process SGTGU](<SGTGU PROCESS REV 3.html>)


### ADDITIONAL STEPS TAKEN SPECIFICALLY FOR QUARTER PROCESS 
** IN PRODUCTION, SYSTEM SHALL EXTRACRT DATA FROM EXTERNAL SYSTEM EJOBS WHICH TRIGGERED MANUALLY EACH TIME 

### STEPS

1. MODULE 1 READ FILE UPLOAD & RAW DATA FROM EJOB DB 
2. MODULE 2 REARRANGES COLUMN FROM RAW DATA ACCORDING TO FILE UPLOAD HEADER_MAIN
3. MODULE 3 FILLUP SECTION, SECTION_SMPL , MSIC2008 & MSIC2008_SMPL FROM MSIC2008_MAINCV	
4. MODULE 4 FILL UP STRATA_EMPL COLUMNS 

	IF SECTION_SMPL == C AND F1310 == RANGE(0,5) -> FILL COLUMN STRATA_EMPL= 4
	IF SECTION_SMPL == C AND F1310 == RANGE(5,75) -> FILL COLUMN STRATA_EMPL= 3
	IF SECTION_SMPL == C AND F1310 == RANGE(75,201) -> FILL COLUMN STRATA_EMPL= 2
	IF SECTION_SMPL == C AND F1310 >= 201 -> FILL COLUMN STRATA_EMPL= 1

	IF SECTION_SMPL != C AND F1310 == RANGE(0,5) -> FILL COLUMN STRATA_EMPL= 4
	IF SECTION_SMPL != C AND F1310 == RANGE(5,30) -> FILL COLUMN STRATA_EMPL= 3
	IF SECTION_SMPL != C AND F1310 == RANGE(30,76) -> FILL COLUMN STRATA_EMPL= 2
	IF SECTION_SMPL != C AND F1310 >= 76 -> FILL COLUMN STRATA_EMPL= 1

	CONDITIONS
	TO FILL STRATA_EMPL COLUMN 
	IF SECTION_SMPL == C
				-COLUMN SECTION_SMPL = C 
				-F1310 RANGE AS BELOW:- 
					- IF PEKERJA 1-4 = MIKRO 4
					- PEKERJA 5-74 = KECIL 3
					- PEKERJA 75-200 = SEDERHANA 2
					- PEKERJA 201 - INFINITY = BESAR 1 

	B . REFER TO 
				-COLUMN SECTION_SMPL != C 
				-F1310 RANGE AS BELOW:- 
						IF PEKERJA 1-4 = MIKRO 4
						PEKERJA 5-29 = KECIL 3
						PEKERJA 30-75 = SEDERHANA 2
						PEKERJA 76 - INFINITY = BESAR 1 

5. MODULE 5 FILL ZERO ACCORDING TO CONDITION
	- newssid (NEWSSID) -12 digit
	- msic (MSIC2008_SMPL & MSIC2008) - 5 digit
	- state (STATE_CODE) - 2 digit
	- substate (SUBSTATE_CODE) - 2 digit

6. MODULE 6 
	1. Filter column that is in vab_list into separate dataframe for main df 
	vab_list = vab_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'O', 'P', 'R ','X']
	2. from df_main_copy , loop through the columns and loop through all vab_list and each vab in vab_list get the unique list of first 6 len and store in a list 
	3. from that list loop through that unique list 

	for i, x in enumerate(df_main_copy.columns):
	- if x[6:] in unique_list & x[-2:].isin(range(1,10)
	- store in a list 
	- sum the list 
	- paste the value in column x[-2:] == '10'
	- do it for all row

7. STEP TO GENERATED FINAL RAW DATA
	1. rename header (NEW_DATASET HEADER)
	2. susun ikut susunan (NEW_ORDER BY VAR)
	3. lookup maklumat section (SECTION_SMPL & SECTION) drpd fail msic
	4. lookup REGISTERED_NAME & TRADING_NAME drpd file sampel
	5. pengiraan utk strata based on bil. pekerja utk bulan ke 3 tempoh rujukan (STRATA_EMPL)
	6. tambah "0" utk cukupkn bilangan digit bg: 
	- newssid (NEWSSID) -12 digit
	- msic (MSIC2008_SMPL & MSIC2008) - 5 digit
	- state (STATE_CODE) - 2 digit
	- substate (SUBSTATE_CODE) - 2 digit
	7. summation of total salaries & wages (Q=L+M+N+O+P)
	8. summation of total separation (X=D+E+F)
	- L01 + M01 + N01 + O01 + P01 = Q01  Dan seterusnya sampai la 10
	9. summation all category (1-9) for variable A, B, C, D, E, F, G, H, I, J, L, M, N, O, P, R & X 
	10. formula utk IND_STATUS

8. Sort COLUMN VARIABLE A-Z 
	- DAPATKAN COLUMN BULAN 10-12 bagi setiap variable 
	- 10-11-12

	SUM # 1  untuk kategori pekerjaan & kekosongan & pekerja bergaji [10] 

	REFER COLUMN B & COLUMN D 
	CODE YANG TERLIBAT = dalam column B 01-09
	SUM KATEGORI PEKERJAAN (COLUMN B) dari 01-09 ->
	totalkan column D dalam kategori perkejaan 10 
	RUN FOR VARIABLE COLUMNN C (A-R) except K 
	ONCE 1 standard has established , we can 
	run into all column for kategori pekerjaan 10
	OUTPUT = 10  



	SUM #2 untuk data gaji & upah [Q]

	code yang terlibat dalam column C [ variable indikator = L M N O P ]

	SUM COLUMN C [L M N O P] -> sumkan column dekat column Q 
	ONCE 1 standard has established , we can run into all column for C variable indikator = Q sum total  ]


	EXCEPT FOR COLUMN B with 10 * 

	OUTPUT = Q 

	SUM #3 SUM untuk total separation 

	SUM KAN TOTAL X FROM [D E F] 
	RUN FOR ALL EXCEPT 10 & Q

	RE_ARRANGE 


	c. REMOVE ALL ROW RELATED TO Z VARIABLE IN COLUMN C 

	e. Transpose back into main sheet final raw 

	- Paste after EST_data to end
		- Transpose → value only no formula
	- copy for value only except column

	f. Remove column B [kategori pekerjaan] & C [ Variable indikator]


9. LOOKUP FROM SAMPLE FILE 

	REGISTERED NAME 
	TRADING NAME 
	EJOBID

	FROM EXCL FILES 
	-> lookup from file sample * different files (changes from time to time) * dynamic upload request (check directory available latest xlsx, and read date. 

	from pathlib import Path

	fld = '.'
	files = Path(fld).glob('*.csv')

	latest = max(files, key=lambda f: f.stat().st_mtime)


	->Based from NEWSSID 



	SECTION_SMPL -> Lookup from lookup file -> MSIC SECTION (a program too) based on MSCIC
	SECTION -> Lookup from lookup file ->  SECTION (a program too) based on MSCIC




