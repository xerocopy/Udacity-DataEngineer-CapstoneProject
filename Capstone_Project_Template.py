#!/usr/bin/env python
# coding: utf-8

# # Study of Immigration Data in the United States
# ### Data Engineering Capstone Project
# 
# #### Project Summary
# 
# In this project, we will be looking at the immigration data for the united states. More specifically, we're interested in looking at the following phenomena:
# * the effects of temperature on the volume of travellers, 
# * the seasonality of travel 
# * the connection between the volume of travel and the number of entry ports (ie airports) 
# * the connection between the volume of travel and the demographics of various cities
# 
# To accomplish this study, we will be using the following datasets:
# 
# * **I94 Immigration Data**: This data comes from the US National Tourism and Trade Office and includes the contents of the i94 form on entry to the united states. A data dictionary is included in the workspace.
#     * _countries.csv_ : table containing country codes used in the dataset, extracted from the data dictionary
#     * _i94portCodes.csv_: table containing city codes used in the dataset, extracted from the data dictionary
# 
# * **World Temperature Data**: This dataset comes from Kaggle and includes the temperatures of various cities in the world fomr 1743 to 2013.
# * **U.S. City Demographic Data**: This data comes from OpenSoft. It contains information about the demographics of all US cities and census-designated places with a population greater or equal to 65,000 and comes from the US Census Bureau's 2015 American Community Survey.
# * **Airport Code Table**: This is a simple table of airport codes and corresponding cities.
# 
# 
# In order to accomplish this, we will aggregate our data as follows:
# * aggregate based on time (year, month, day, etc...) 
# * aggregate data by cities and airports
# * look at the impact of temperatures on the in and ouflux of travelers
# * the impact on regional demographics
# 
# The project follows the follow steps:
# * Step 1: Scope the Project and Gather Data
# * Step 2: Explore and Assess the Data
# * Step 3: Define the Data Model
# * Step 4: Run ETL to Model the Data
# * Step 5: Complete Project Write Up

# In[1]:


# Do all imports and installs here
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, date_add
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc
from pyspark.sql.functions import sum as Fsum

import datetime

import numpy as np
import pandas as pd


# ### Step 1: Scope the Project and Gather Data
# 
# #### Scope 
# Explain what you plan to do in the project in more detail. What data do you use? What is your end solution look like? What tools did you use? etc>
# 
# #### Describe and Gather Data 
# Describe the data sets you're using. Where did it come from? What type of information is included? 

# #### Immigration Dataset

# The immigration dataset is rather large, containing approximately 3m lines. We have will use a subset of approx 1000 rows in a csv to explore it

# In[2]:


# Read in the data here
df_immig_sample = pd.read_csv('immigration_data_sample.csv')


# This data set contains details about traveler entering the united states.

# In[3]:


df_immig_sample.columns


# The definition of the fiels is included in the file **I94_SAS_Labels_Descriptions.SAS**
# We'll primarily be interested in the following fields:
# * i94cit : country of citizenship
# * i94res : country of residence
# * i94port: arrival airport
# * arrdate: arrival date. 
# * i94mode
# * i94addr
# * depdate
# * i94bir
# * i94visa
# * occup
# * biryear
# * dtaddto
# * gender
# * insnum
# * airline
# * admnum
# * fltno
# * visatype
# 
# 
# 

# Let's increase the number of columns that can be displayed at once to have a better look at the data

# In[4]:


pd.set_option('display.max_columns', 50)
df_immig_sample.head(10)


# #### Adding data dictionaries

# In the next step, we add details from the data dictionnary. These will be replace the codes in our data model since we work with denormalized data models.

# First we add the dictionary for columns I94CIT & I94RES (countries.csv) which we assume corresponds to country of citizenship and country of residence of the travelers.

# In[5]:


df_countryCodes = pd.read_csv('countries.csv')


# In[6]:


df_countryCodes.shape


# In[7]:


df_countryCodes.head()


# Next, we add a correspondance table between i94port codes and city that was the port of entry (i94portCodes.csv).

# In[8]:


i94portCodes = pd.read_csv('i94portCodes.csv')


# In[9]:


i94portCodes.shape


# In[10]:


i94portCodes.head()


# #### Demographic data

# Our dataset contains details on the reason for traveling. It might be interesting to see if there is a connection between the flow of immigration and the demographic data of various US cities.
# 
# Let's load the cities demographics
# The separator used in the csv is a ';' rather than a ','

# In[11]:


# Read in the data here
df_demographics = pd.read_csv('us-cities-demographics.csv', sep=';')


# In[12]:


df_demographics.shape


# In[13]:


df_demographics.head()


# In[14]:


# we look at the list of available columns in the dataset
df_demographics.columns


# Next, we load the airport codes. This data will allow us to connect airport data to the airport codes

# #### Airport data

# Since airports are the point of entry for these immigrants, we will include airport information in our data.

# In[15]:


# Read in the data here
df_airports = pd.read_csv('airport-codes_csv.csv')


# In[16]:


df_airports.columns


# In[17]:


df_airports.head()


# #### World Temperature data
# We expect there to be a connection between climate and tourism data. Therefore, we will add the world temperature data to see if climate has any impact on the volume of tourists.
# In[18]:


fname = '../../data2/GlobalLandTemperaturesByCity.csv'
df_temperature = pd.read_csv(fname)


# In[19]:


df_temperature.shape


# In[20]:


df_temperature.head()


# #### Full immigration dataset
#Finally, we load the full immigration dataset into a spark dataframe since it is quite large
# In[21]:


from pyspark.sql import SparkSession
#spark = SparkSession.builder.config("spark.jars.packages","saurfang:spark-sas7bdat:2.0.0-s_2.11").enableHiveSupport().getOrCreate()

spark = SparkSession.builder.\
config("spark.jars.repositories", "https://repos.spark-packages.org/").\
config("spark.jars.packages", "saurfang:spark-sas7bdat:2.0.0-s_2.11").\
enableHiveSupport().getOrCreate()
df_immigration = spark.read.format('com.github.saurfang.sas.spark').load('../../data/18-83510-I94-Data-2016/i94_apr16_sub.sas7bdat')


# In[22]:


df_immigration.count()


# In[23]:


df_immigration.printSchema()


# We see here that the schema is identical to the sample dataset.

# In[24]:


#write to parquet
df_immigration.write.parquet("sas_data")
df_immigration=spark.read.parquet("sas_data")


# ### Step 2: Explore and Assess the Data
# #### Explore the Data 
# Identify data quality issues, like missing values, duplicate data, etc.
# 
# #### Cleaning Steps
# Document steps necessary to clean the data

# ##### Temperature data
# First, we start looking at the temperature dataset

# In[25]:


df_temperature.shape

#Let's see if we need to keep all this data
# In[26]:


df_temperature.head()


# In[27]:


df_temperature['Country'].nunique()


# The dataframe contains temperature data for 159 countries since the year 1743.
# We will reduce the size of the dataset to make it more manageable.

# In[28]:


# Keep only data for the United States
df_temperature = df_temperature[df_temperature['Country']=='United States']


# In[29]:


# Convert the date to datetime objects
df_temperature['convertedDate'] = pd.to_datetime(df_temperature.dt)


# Since we are focusing on air travel, we'll exclude any data prior to 1950 since commercial air travel did't develop until after the second world war in the 1950s

# In[30]:


# Remove all dates prior to 1950
df_temperature=df_temperature[df_temperature['convertedDate']>"1950-01-01"].copy()


# In[31]:


df_temperature.shape


# In[32]:


# Let's check the most recent date in the dataset
df_temperature['convertedDate'].max()


# No temperature that's available here can be joined with our immigration dataset. We will assume for the purposes of this project that we'd have data that can be joined with our immigration dataset

# Now, let's check for missing data

# In[33]:


# Let's check for null values.
df_temperature.isnull().sum()


# In[34]:


df_temperature[df_temperature.AverageTemperature.isnull()]


# Normally, we would have to fix the null values present in the temperature dataset for Anchorage in September 2019. Two possible straightforward strategies for fixing this issue would be:
# 
# * Average out the values of August and October 2019 for Anchorage;
# * Using an average of the historical data for the month of September for Anchorage;
# 
# Option 2 while feasiable is less desirable since temperatures appear to be higher in 2013 compared to previous years and might create an outlier in the dataset.
# 
# Normally our dataset would include data all the way to aprl 2016 to allow us to join the data with our immigration dataset, making it possible to use option 1.
# However, since no data is available in this set beyond 2013-09-01 and because our immigration set only covers the month of april 2016, we'll leave this missing data problem as is.

# Finally, let's make sure the combination of city and date can be used as a primary key.
# We've assumed that each row represents a combination of city and date. Let's check that

# In[35]:


df_temperature.shape


# In[36]:


df_temperature[['City','convertedDate']].drop_duplicates().shape


# It looks like there can be multiple entries for a give city. Let's try to look at an example of multiple entries

# In[37]:


df_temperature[df_temperature[['City','convertedDate']].duplicated()].head()


# In[38]:


df_temperature[(df_temperature['City'] == 'Arlington') & (df_temperature.dt == '1950-02-01')]


# It looks like the temperature is measured in multiple locations for each city.
# When creating the dimension table, we'll compute the average temperatures and uncertainties per city

# #### Airport data

# Next, we look at the contents of the airport dataset

# In[39]:


df_airports.shape


# In[40]:


df_airports.head()


# Let's check the countries where these airports are located

# In[41]:


df_airports.groupby('iso_country')['iso_country'].count()


# This dataset contains airport data for numerous countries. Our immigration dataset only contains entries into the US, thus via airports based in the united states. Therefore, we'll be reducing the size of this dataset
# 
# Before we do that, let's make sure there is no missing data in the iso_county field.

# In[42]:


df_airports[df_airports['iso_country'].isna()].shape


# In[43]:


# Let's quickly check the missing country values to see if the continent data is filled out
df_airports[df_airports['iso_country'].isna()].groupby('continent')['continent'].count()


# All the missing country data is for airports based in Africa.
# 
# Therefore, the dataset can safely be reduced

# In[44]:


# Since all missing values are in africa, we simly remove them from the dataset
df_airports = df_airports[df_airports['iso_country'].fillna('').str.upper().str.contains('US')].copy()


# The type column  contains several values. Let's look at them:

# In[45]:


df_airports.groupby('type')['type'].count()


# We will assume the following here:
# * closed indicates the aireport is closed
# * No immigration data is collected from balloonports, seaplane bases or heliport since these means of transportation are used for recreational purposes or very short distances
# 
# Therefore, we filter out all rows with these values

# In[46]:


excludedValues = ['closed', 'heliport', 'seaplane_base', 'balloonport']
df_airports = df_airports[~df_airports['type'].str.strip().isin(excludedValues)].copy()


# Let's look for other missing values

# In[47]:


#We check againvalues:
df_airports.isnull().sum()


# The ident code cannot be used to join airport data with the immigration set. Ineed, the airport linked to the ident, local or iata code columns are very different from the definitions found in the data dictionary. Therefore we must use the municipality names to join our datasets.
# 
# From our previous validation, we know that we have 50 values missing from our dataset.
# Let's look at some of these missing values to see if we can get the municipality name through some other means

# In[48]:


# We also verify that the municipality field is available for all airports
df_airports[df_airports.municipality.isna()].head()


# None of these appear to be useable in a way that could be automated if we were building a pipeline. Therefore, we'll just remove them from our dataset.

# In[49]:


df_airports = df_airports[~df_airports['municipality'].isna()].copy()


# We convert the municipality column to upper case in order to be able to join it with our other datasets.

# In[50]:


df_airports.municipality = df_airports.municipality.str.upper()


# In[51]:


df_airports.groupby('iso_region')['iso_region'].count()


# Looking at the state data, U-A seems like an error. State is used in combination with city name to join with city demographics

# In[52]:


# apply len to the iso_region field to see which ones are longer than 5 characters since the field is a combination of US and state code
df_airports['len'] = df_airports["iso_region"].apply(len)
# let's remove the codes that are incorrect.
df_airports = df_airports[df_airports['len']==5].copy()
# finally, let's extract the state code
df_airports['state'] = df_airports['iso_region'].str.strip().str.split("-", n = 1, expand = True)[1]


# #### Demographic data

# Let's look at the demographic dataset

# In[53]:


df_demographics.shape


# Let's first convert the city to upper case and remove any leading and trailing spaces

# In[54]:


df_demographics.City = df_demographics.City.str.upper().str.strip()


# Let's look at missing values

# In[55]:


df_demographics.isnull().sum()


# This dataset looks relatively clean with few missing values of significance.

# We will not try to fix any of the missing data for now.
# We'll fix any issues with these missing rows when loading our dimension tables.

# In[56]:


# remove any leading or trailing spaces and convert to upper case
df_demographics.City = df_demographics.City.str.strip().str.upper()


# Let's check whether city and race would work as a primary key for this table

# In[57]:


#primary key will be the combination of city name and race
df_demographics[df_demographics[['City','Race']].duplicated()].head()


# Clearly, the combination of city and race is not sufficient to work as a primary key. Let's look at a specific example

# In[58]:


df_demographics[(df_demographics.City == 'WILMINGTON') & (df_demographics.Race == 'Asian')]


# The differene between these two rows is the state. Let's add it in the primary key combination

# In[59]:


df_demographics[df_demographics[['City', 'State','Race']].duplicated()]


# There are no duplicate rows when we use this combination. We will use it as our primary key.

# #### Immigration data

# The data dictionary provided already contains a lot of details on the missing data

# In[60]:


df_immigration.show(5)


# First, let's check to see if cicid can be used as a primary key

# In[61]:


# We create a view of the immigration dataset
df_immigration.createOrReplaceTempView("immig_table")


# In[62]:


df_immigration.count()


# In[63]:


spark.sql("""
SELECT COUNT (DISTINCT cicid)
FROM immig_table
""").show()


# We've been provided with a data dictionary for i94port where all the codes are 3 character long
# Let's check if the same applies to the codes in the dataset

# In[64]:


spark.sql("""
SELECT LENGTH (i94port) AS len
FROM immig_table
GROUP BY len
""").show()


# No processing is needed to join this to our dictionary

# Next, we need to do is to convert the arrdate field into something that can be used.

# All dates in SAS correspond to the number of days since 1960-01-01.
# Therfore, we compute the arrival dates by adding arrdate to 1960-01-01

# In[65]:


df_immigration = spark.sql("SELECT *, date_add(to_date('1960-01-01'), arrdate) AS arrival_date FROM immig_table")
df_immigration.createOrReplaceTempView("immig_table")


# Next, we replace the data in the I94VISA columns
# The three categories are:
# *   1 = Business
# *   2 = Pleasure
# *   3 = Student

# In[66]:


spark.sql("""SELECT *, CASE 
                        WHEN i94visa = 1.0 THEN 'Business' 
                        WHEN i94visa = 2.0 THEN 'Pleasure'
                        WHEN i94visa = 3.0 THEN 'Student'
                        ELSE 'N/A' END AS visa_type 
                        
                FROM immig_table""").createOrReplaceTempView("immig_table")


# In[67]:


spark.sql("""SELECT *, CASE 
                        WHEN depdate >= 1.0 THEN date_add(to_date('1960-01-01'), depdate)
                        WHEN depdate IS NULL THEN NULL
                        ELSE 'N/A' END AS departure_date 
                        
                FROM immig_table""").createOrReplaceTempView("immig_table")


# In[68]:


#Let's check the results from our previous query to make sure there are no N/A values
spark.sql("SELECT count(*) FROM immig_table WHERE departure_date = 'N/A'").show()


# Now, let's make sure that departure_date > arrival_date

# In[69]:


spark.sql("""
SELECT COUNT(*)
FROM immig_table
WHERE departure_date <= arrival_date
""").show()

#We have 120 rows where the departure_date appears to be wrong. Let's see if we can make sense of the dates
# In[70]:


spark.sql("""
SELECT arrival_date, departure_date
FROM immig_table
WHERE departure_date <= arrival_date
""").show(10)


# It's impossible to know how to fix these errors. Since the number of affected rows is relatively small, we'll simply drop the rows

# In[71]:


spark.sql("""
SELECT *
FROM immig_table
WHERE departure_date >= arrival_date
""").createOrReplaceTempView("immig_table")


# Lastly, let's check how many distinct values we get in the arrival and departure dates to see if we need to merge our two sets for the time dimension table we'll be using in our model

# In[72]:


#check distinct departure dates
spark.sql("SELECT COUNT (DISTINCT departure_date) FROM immig_table ").show()
#check distinct arrival dates
spark.sql("SELECT COUNT (DISTINCT arrival_date) FROM immig_table ").show()
#check the common values between the two sets
spark.sql("""   SELECT COUNT(DISTINCT departure_date) 
                FROM immig_table 
                WHERE departure_date IN (
                    SELECT DISTINCT arrival_date FROM immig_table
                ) 
                """).show()


# Since one value is missing, we will merge the two data sets to allow for our dim table to include both departure and arrival dates.

# Let's check the data for the various arrival modes

# In[73]:


spark.sql("""
SELECT i94mode, count(*)
FROM immig_table
GROUP BY i94mode
""").show()


# The arrival modes definition as per the dictonary is:
# * 1 = 'Air'
# * 2 = 'Sea'
# * 3 = 'Land'
# * 9 = 'Not reported'
# We will only keep Air arrival since we're joining this with airport datasets

# We will keep only arrival by air to ensure that our dataset can work with the airports dataset

# Let's check if there are any missing values in the age column

# In[74]:


spark.sql("""
SELECT COUNT(*)
FROM immig_table
WHERE i94bir IS NULL
""").show()


# Since we have some missing values here, let's check the birthyear instead to see if it can be used

# In[75]:


spark.sql("SELECT COUNT(biryear) FROM immig_table WHERE biryear IS NULL").show()


# Let's check if the year of birth makes sense too

# In[76]:


spark.sql("SELECT MAX(biryear), MIN(biryear) FROM immig_table WHERE biryear IS NOT NULL").show()


# Let's look at the frequency of travellers who are at least 80 years old, ie born in 1936 or earlier

# In[77]:


#Number of travellers who are older than 80
spark.sql("""
SELECT COUNT(*)
FROM immig_table 
WHERE biryear IS NOT NULL
AND biryear <= 1936
""").show()

# frequency of travellers by birth year
spark.sql("""
SELECT biryear, COUNT(*)
FROM immig_table 
WHERE biryear IS NOT NULL
AND biryear <= 1936
GROUP BY biryear
ORDER BY biryear ASC
""").show()


# The age of the travellers who are 105 years old are outliera with only 8 observations (out of 3 millions) and olny 0.6% of the travellers are over the age of 80 which seems reasonable.

# Since the birth year is available for each row, we can compute the age. Let's check if computed values match the age

# In[78]:


spark.sql("SELECT (2016-biryear)-i94bir AS difference, count(*) FROM immig_table WHERE i94bir IS NOT NULL GROUP BY difference").show()


# This technique yields the exact same results as the age when the field is available. We will use that instead of the age to fill in the missing values.

# Let's check the gender to see if the data is useable

# In[79]:


spark.sql("""
SELECT gender, count(*) 
FROM immig_table
GROUP BY gender
""").show()


# Since we'd like to retain the gender of the various travellers, we will filter out all the rows where the gender is missing or incorrect

# In[80]:


spark.sql("""SELECT * FROM immig_table WHERE gender IN ('F', 'M')""").createOrReplaceTempView("immig_table")


# Let's check the citizenship and residence data to see if any values are missing

# In[81]:


#citizenship countries
spark.sql("""
SELECT count(*) 
FROM immig_table
WHERE i94cit IS NULL
""").show()

#residence countries
spark.sql("""
SELECT count(*) 
FROM immig_table
WHERE i94res IS NULL
""").show()

#reported address
spark.sql("""
SELECT count(*) 
FROM immig_table
WHERE i94addr IS NULL
""").show()


# The addresses (really the state of residence) are missing quite often. We won't use this field, relying instead on the port of entry as a proxy for the traveller's address

# In[82]:


spark.sql("""
SELECT COUNT(*)
FROM immig_table
WHERE visatype IS NULL
""").show()


# Detailed vista type is available for all rows. 
# Let's check the aggregation to make sure the visa types are unique to each category.

# In[83]:


spark.sql("""
SELECT visa_type, visatype, count(*)
FROM immig_table
GROUP BY visa_type, visatype
ORDER BY visa_type, visatype
""").show()


# The definitions for various detailed visa types are listed below. Some are unknown.
# We couldn't find definitions for all the visa types. We will retain the details since it might be of interest from a demographic standpoint

# * B1 visa is for business visits valid for up to a year
# * B2 visa is for pleasure visits valid for up to a year
# * CP could not find a definition
# * E2 investor visas allows foreign investors to enter and work inside of the United States based on a substantial investment
# * F1 visas are used by non-immigrant students for Academic and Language training Courses. 
# * F2 visas are used by the dependents of F1 visa holders
# * GMT could not find a definition
# * M1 for students enrolled in non-academic or “vocational study”. Mechanical, language, cooking classes, etc...
# * WB Waiver Program (WT/WB Status) travel to the United States for tourism or business for stays of 90 days or less without obtaining a visa.
# * WT Waiver Program (WT/WB Status) travel to the United States for tourism or business for stays of 90 days or less without obtaining a visa.

# Since we have little information besides the detailed visa type and the aggregate visa type, we will simply keep the information in our dimension table. 

# Let's take a look at the occupation field.

# In[84]:


spark.sql("""
SELECT occup, COUNT(*) AS n
FROM immig_table
GROUP BY occup
ORDER BY n DESC, occup
""").show()


# The field is missing most of the time and the values provided are abbreviations. We won't be using it in our data model

# Several other fields are missing a lot of values or simply not used or documented and will be dropped.

# We've completed our analysis of the tables. Let's build a conceptual model now

# In[85]:


df_immigration = spark.sql("""SELECT * FROM immig_table""") 


# ### Step 3: Define the Data Model
# #### 3.1 Conceptual Data Model
# Since we're interested in the flow of travellers through the united states. The i94 data will serve as our fact table.
# Our **fact_immigration** table will be :
# * cicid,
# * citizenship_country,
# * residence_country,
# * city,
# * state,
# * arrival_date,
# * departure_date,
# * age,
# * visa_type,
# * detailed_visa_type,
# 
# For our dimension tables, since our dataset only contains one month of data we will keep a record of the daily entries and provide the uses with four dimensions to aggregate our data:\
# 
# **dim_time** : to aggregate the data suing various time units: The fileds available will be:
# * date, 
# * year, 
# * month, 
# * day, 
# * week,
# * weekday,
# * dayofyear
# 
# **dim_airports**: Used to determine the areas with the largest flow of travelers. Fileds included will be:
# * ident,
# * type, 
# * name, 
# * elevation_ft, 
# * state,
# * municipality, 
# * iata_code
# 
# **dim_city_demographics**: To look at the demographic data of the areas with the most travelers and potentially look at the impact of the flow of travellers on the demographic data (if it were updated on a regular basis). The fiels available will be:
# * City, 
# * state, 
# * median_age, 
# * male_population, 
# * female_population, 
# * total population
# * Foreign_born, 
# * Average_Household_Size, 
# * Race, 
# * Count,
# 
# **dim_temperatures**: to look at the temperature data of the cities where traveller entry and departure is being reported. The fields included will be: 
# * date, 
# * City,
# * average temperature, 
# * average temperature uncertainty 
# 
# 
# #### 3.2 Mapping Out Data Pipelines
# 
# Many of data data cleaning steps were documented in adetailed fashion in the section 2. Here are the steps again:
# ##### Data Extraction:
# * Load all the datasets from CSV and SAS data files;
# 
# ##### Data Transformation and Loading:
# 
# ##### fact_immigration:
# * Drop rows where the mode of arrival is not air travel
# * Drop rows with incorrect gender data
# * convert arrival and departure dates;
# * replace country codes with the character string equivalents
# * replace visa_type with character string
# * replace port of entry with city and state
# * filter out any row where the port of entry is not in the US
# * compute age in a new row using birth year and year of our current date.
# * insert data into our fact table
# * Write to parquet
# 
# ##### dim_temperature:
# * For the temperature table, drop all data for cities outside the united states;
# * For the temperature table, drop all data for dates before 1950 since airtravel wasn't possible before that date;
# * Convert city to upper case
# * Compute the average temperature and uncertainty over date+city partitions
# * Insert into the temperature table as is since our dataset since our dataset may include new cities in future dates;
# * Write to parquet
# 
# ##### dim_time:
# * Get all the arrival dates from the immigration data_set;
# * extract year, month, day, week from the date and insert all the values in the dim_time table;
# * Write to parquet
# 
# ##### dim_airports:
# * Remove all non us airports
# * Remove all invalid port of entries, ie: ['closed', 'heliport', 'seaplane_base', 'balloonport']
# * Remove all rows where municipalities are missing.
# * Convert municipality to upper case
# * Insert to our table
# * Write to parquet
# 
# ##### dim_city_demographics:
# * Convert to city names to upper case
# * Insert to our table
# * Write to parquet
# 

# ### Step 4: Run Pipelines to Model the Data 
# #### 4.1 Create the data model

# Note that spark automatically reads all fields as strings in our CSV files whereas pandas usually correctly autodectects the data types (as seen below). Consquently, we'll read all the csv files using pandas dataframes and then convert them to spark dataframes.

# In[86]:


df_demographics_spark = spark.read.format("csv").option("header", "true").option("delimiter", ";").load('us-cities-demographics.csv')


# In[87]:


df_demographics_spark.printSchema()


# In[88]:


df_demographics.dtypes


# In[89]:


spark.createDataFrame(df_demographics).printSchema()


# #### Staging the data

# In[90]:


# load dictionary data
df_countryCodes = pd.read_csv('countries.csv')
df_i94portCodes = pd.read_csv('i94portCodes.csv')

# load the various csv files into pandas dataframes
df_demographics = pd.read_csv('us-cities-demographics.csv', sep=';')
df_temperature = pd.read_csv('../../data2/GlobalLandTemperaturesByCity.csv')

# load the SAS data
df_immigration=spark.read.parquet("sas_data")


# #### Transforming the data

# In[91]:


# let's convert the data dictionaries to views in our spark context to perform SQL operations with them
spark_df_countryCodes = spark.createDataFrame(df_countryCodes)
spark_df_countryCodes .createOrReplaceTempView("countryCodes")


# In[92]:


# remove all entries with null values as they are either un reported or outside the US
df_i94portCodes = df_i94portCodes[~df_i94portCodes.state.isna()].copy()


# In[93]:


# We need to exclude values for airports outside of the US. 
nonUSstates = ['CANADA', 'Canada', 'NETHERLANDS', 'NETH ANTILLES', 'THAILAND', 'ETHIOPIA', 'PRC', 'BERMUDA', 'COLOMBIA', 'ARGENTINA', 'MEXICO', 
               'BRAZIL', 'URUGUAY', 'IRELAND', 'GABON', 'BAHAMAS', 'MX', 'CAYMAN ISLAND', 'SEOUL KOREA', 'JAPAN', 'ROMANIA', 'INDONESIA',
               'SOUTH AFRICA', 'ENGLAND', 'KENYA', 'TURK & CAIMAN', 'PANAMA', 'NEW GUINEA', 'ECUADOR', 'ITALY', 'EL SALVADOR']


# In[94]:


df_i94portCodes = df_i94portCodes[~df_i94portCodes.state.isin(nonUSstates)].copy()


# In[95]:


spark_df_i94portCodes = spark.createDataFrame(df_i94portCodes)
spark_df_i94portCodes .createOrReplaceTempView("i94portCodes")


# In[96]:


df_immigration.createOrReplaceTempView("immig_table")


# In[97]:


# Remove all entries into the united states that weren't via air travel
spark.sql("""
SELECT *
FROM immig_table
WHERE i94mode = 1
""").createOrReplaceTempView("immig_table")


# In[98]:


# drop rows where the gender values entered is undefined
spark.sql("""SELECT * FROM immig_table WHERE gender IN ('F', 'M')""").createOrReplaceTempView("immig_table")


# In[99]:


# convert the arrival dates into a useable value
spark.sql("SELECT *, date_add(to_date('1960-01-01'), arrdate) AS arrival_date FROM immig_table").createOrReplaceTempView("immig_table")


# In[100]:


# convert the departure dates into a useable value
spark.sql("""SELECT *, CASE 
                        WHEN depdate >= 1.0 THEN date_add(to_date('1960-01-01'), depdate)
                        WHEN depdate IS NULL THEN NULL
                        ELSE 'N/A' END AS departure_date 
                        
                FROM immig_table""").createOrReplaceTempView("immig_table")


# In[101]:


# we use an inner join to drop invalid codes
#country of citizenship
spark.sql("""
SELECT im.*, cc.country AS citizenship_country
FROM immig_table im
INNER JOIN countryCodes cc
ON im.i94cit = cc.code
""").createOrReplaceTempView("immig_table")


# In[102]:


#country of residence
spark.sql("""
SELECT im.*, cc.country AS residence_country
FROM immig_table im
INNER JOIN countryCodes cc
ON im.i94res = cc.code
""").createOrReplaceTempView("immig_table")


# In[103]:


# Add visa character string aggregation
spark.sql("""SELECT *, CASE 
                        WHEN i94visa = 1.0 THEN 'Business' 
                        WHEN i94visa = 2.0 THEN 'Pleasure'
                        WHEN i94visa = 3.0 THEN 'Student'
                        ELSE 'N/A' END AS visa_type 
                        
                FROM immig_table""").createOrReplaceTempView("immig_table")


# In[104]:


# Add entry_port names and entry port states to the view
spark.sql("""
SELECT im.*, pc.location AS entry_port, pc.state AS entry_port_state
FROM immig_table im 
INNER JOIN i94portCodes pc
ON im.i94port = pc.code
""").createOrReplaceTempView("immig_table")


# In[105]:


# Compute the age of each individual and add it to the view
spark.sql("""
SELECT *, (2016-biryear) AS age 
FROM immig_table
""").createOrReplaceTempView("immig_table")


# In[106]:


# Insert the immigration fact data into a spark dataframe
fact_immigration = spark.sql("""
                        SELECT 
                            cicid, 
                            citizenship_country,
                            residence_country,
                            TRIM(UPPER (entry_port)) AS city,
                            TRIM(UPPER (entry_port_state)) AS state,
                            arrival_date,
                            departure_date,
                            age,
                            visa_type,
                            visatype AS detailed_visa_type

                        FROM immig_table
""")


# In[107]:


# extract all distinct dates from arrival and departure dates to create dimension table
dim_time = spark.sql("""
SELECT DISTINCT arrival_date AS date
FROM immig_table
UNION
SELECT DISTINCT departure_date AS date
FROM immig_table
WHERE departure_date IS NOT NULL
""")
dim_time.createOrReplaceTempView("dim_time_table")


# In[108]:


# extract year, month, day, weekofyear, dayofweek and weekofyear from the date and insert all the values in the dim_time table;
dim_time = spark.sql("""
SELECT date, YEAR(date) AS year, MONTH(date) AS month, DAY(date) AS day, WEEKOFYEAR(date) AS week, DAYOFWEEK(date) as weekday, DAYOFYEAR(date) year_day
FROM dim_time_table
ORDER BY date ASC
""")


# In[109]:


# Keep only data for the United States
df_temperature = df_temperature[df_temperature['Country']=='United States'].copy()

# Convert the date to datetime objects
df_temperature['date'] = pd.to_datetime(df_temperature.dt)

# Remove all dates prior to 1950
df_temperature=df_temperature[df_temperature['date']>"1950-01-01"].copy()


# In[110]:


# convert the city names to upper case
df_temperature.City = df_temperature.City.str.strip().str.upper() 


# In[111]:


# convert the dataframes from pandas to spark
spark_df_temperature = spark.createDataFrame(df_temperature)
spark_df_temperature .createOrReplaceTempView("temperature")


# In[112]:


dim_temperature = spark.sql("""
SELECT
    DISTINCT date, city,
    AVG(AverageTemperature) OVER (PARTITION BY date, City) AS average_temperature, 
    AVG(AverageTemperatureUncertainty)  OVER (PARTITION BY date, City) AS average_termperature_uncertainty
    
FROM temperature
""")


# In[113]:


df_demographics.City = df_demographics.City.str.strip().str.upper()
df_demographics['State Code'] = df_demographics['State Code'].str.strip().str.upper()
df_demographics.Race = df_demographics.Race.str.strip().str.upper()


# In[114]:


# convert the dataframes from pandas to spark
spark_df_demographics = spark.createDataFrame(df_demographics)
spark_df_demographics.createOrReplaceTempView("demographics")


# In[115]:


# insert data into the demographics dim table
dim_demographics = spark.sql("""
                                SELECT  City, 
                                        State, 
                                        `Median Age` AS median_age, 
                                        `Male Population` AS male_population, 
                                        `Female Population` AS female_population, 
                                        `Total Population` AS total_population, 
                                        `Foreign-born` AS foreign_born, 
                                        `Average Household Size` AS average_household_size, 
                                        `State Code` AS state_code, 
                                        Race, 
                                        Count
                                FROM demographics
""")


# In[116]:


#The airport dataset contains a lot of nulls. We'll load the csv directly into a spark dataframe to avoid having to deal with converting pandas NaN into nulls
spark_df_airports = spark.read.format("csv").option("header", "true").load('airport-codes_csv.csv')
spark_df_airports.createOrReplaceTempView("airports")


# In[117]:


#equivalent to the following pandas code:
# df_airports = df_airports[df_airports['iso_country'].fillna('').str.upper().str.contains('US')].copy()
spark.sql("""
SELECT *
FROM airports
WHERE iso_country IS NOT NULL
AND UPPER(TRIM(iso_country)) LIKE 'US'
""").createOrReplaceTempView("airports")


# In[118]:


#equivalent to the following pandas code:
# excludedValues = ['closed', 'heliport', 'seaplane_base', 'balloonport']
# df_airports = df_airports[~df_airports['type'].str.strip().isin(excludedValues)].copy()
# df_airports = df_airports[~df_airports['municipality'].isna()].copy()
# df_airports = df_airports[~df_airports['municipality'].isna()].copy()
# df_airports['len'] = df_airports["iso_region"].apply(len)
# df_airports = df_airports[df_airports['len']==5].copy()

spark.sql("""
SELECT *
FROM airports
WHERE LOWER(TRIM(type)) NOT IN ('closed', 'heliport', 'seaplane_base', 'balloonport')
AND municipality IS NOT NULL
AND LENGTH(iso_region) = 5
""").createOrReplaceTempView("airports")


# In[119]:


dim_airports = spark.sql("""
SELECT TRIM(ident) AS ident, type, name, elevation_ft, SUBSTR(iso_region, 4) AS state, TRIM(UPPER(municipality)) AS municipality, iata_code
FROM airports
""")


# In[120]:


# Saving the data in parquet format
dim_demographics.write.parquet("dim_demographics")
dim_time.write.parquet("dim_time")
dim_airports.write.parquet("dim_airports")
dim_temperature.write.parquet("dim_temperature")
fact_immigration.write.parquet("fact_immigration")


# #### 4.2 Data Quality Checks
# Explain the data quality checks you'll perform to ensure the pipeline ran as expected. These could include:
#  * Integrity constraints on the relational database (e.g., unique key, data type, etc.)
#  * Unit tests for the scripts to ensure they are doing the right thing
#  * Source/Count checks to ensure completeness
#  
# Run Quality Checks

# In[121]:


#Let's check some things in our data
dim_demographics.createOrReplaceTempView("dim_demographics")
dim_time.createOrReplaceTempView("dim_time")
dim_airports.createOrReplaceTempView("dim_airports")
dim_temperature.createOrReplaceTempView("dim_temperature")
fact_immigration.createOrReplaceTempView("fact_immigration")


# First, let's make sure the columns used as primary keys don't contain any null values.
# We define a function that could be incorporated in an automated data pipeline

# In[122]:


# we define the following function to check for null values
def nullValueCheck(spark_ctxt, tables_to_check):
    """
    This function performs null value checks on specific columns of given tables received as parameters and raises a ValueError exception when null values are encountered.
    It receives the following parameters:
    spark_ctxt: spark context where the data quality check is to be performed
    tables_to_check: A dictionary containing (table, columns) pairs specifying for each table, which column is to be checked for null values.   
    """  
    for table in tables_to_check:
        print(f"Performing data quality check on table {table}...")
        for column in tables_to_check[table]:
            returnedVal = spark_ctxt.sql(f"""SELECT COUNT(*) as nbr FROM {table} WHERE {column} IS NULL""")
            if returnedVal.head()[0] > 0:
                raise ValueError(f"Data quality check failed! Found NULL values in {column} column!")
        print(f"Table {table} passed.")


# Next we run the data quality check on all the tables in our data model

# In[123]:


#dictionary of tables and columns to be checked
tables_to_check = { 'fact_immigration' : ['cicid'], 'dim_time':['date'], 'dim_demographics': ['City','state_code'], 'dim_airports':['ident'], 'dim_temperature':['date','City']}

#We call our function on the spark context
nullValueCheck(spark, tables_to_check)


# The data quality check was successful.
# 
# Let's do a more detailed check for each table

# In[124]:


#time dimension verification

#check the number of rows in our time table : 192 expected
spark.sql("""
SELECT COUNT(*) - 192
FROM dim_time
""").show()

# make sure each row has a distinct date key : 192 expected
spark.sql("""
SELECT COUNT(DISTINCT date) - 192
FROM dim_time
""").show()

# we could also subtract the result of one query from the other


# and make sure all dates from the fact table are included in the time dimension (NULL is the expected result)
spark.sql("""
SELECT DISTINCT date
FROM dim_time

MINUS

(SELECT DISTINCT arrival_date AS date
FROM immig_table
UNION
SELECT DISTINCT departure_date AS date
FROM immig_table
WHERE departure_date IS NOT NULL)

""").show()


# In[125]:


#immigration verification

# The number of primary key from the staging table (2165257 expected)
spark.sql("""
SELECT count(distinct cicid) - 2165257
FROM immig_table
""").show()

#should match the primary key count from the fact table (2165257 expected)
spark.sql("""
SELECT count(distinct cicid) - 2165257
FROM fact_immigration
""").show()

#and should match the row count from the fact table since it is also the primary key (2165257 expected)
spark.sql("""
SELECT count(*) - 2165257
FROM fact_immigration
""").show()


# In[126]:


# Let's check the demographics dimension table (2891 expected) 
spark.sql("""
SELECT count(*) - 2891
FROM dim_demographics
""").show()

spark.sql("""
SELECT COUNT(DISTINCT city, state, race) - 2891
FROM dim_demographics
""").show()


# In[127]:


# Let's check the primary key for airports (expected 14529)
spark.sql("""
SELECT count(*) - 14529
FROM dim_airports
""").show()

spark.sql("""
SELECT COUNT(DISTINCT ident) - 14529
FROM dim_airports
""").show()


# In[128]:


#finally, city + date is our primary key for the temperature (expected 189472)

spark.sql("""
SELECT count(*) - 189472
FROM dim_temperature
""").show()

spark.sql("""
SELECT COUNT(DISTINCT date, city) - 189472
FROM dim_temperature
""").show()


# Now, let's check what happens when we join our dimensions and fact tables

# In[129]:


# First, we join airport and immigration
fact_immigration.show(2)
dim_airports.show(2)


# Since a given city can have more than one airport and airport data is not provided in the immigration dataset, let's try to see how many city & state combinations are common to the two datasets.
# 
# We're looking at immigrant influx based on cities. Thus, we'd like to check whether the use of city and state combination works well to match the data between dim_airport and fact_immigration 

# In[130]:


#here are the distinct combinations of city and state in our fact table
spark.sql("""
SELECT COUNT(DISTINCT city, state)
FROM fact_immigration
""").show()

# and the combinations of city and state that are common to both
spark.sql("""
SELECT COUNT(*)
FROM
(
SELECT DISTINCT city, state
FROM fact_immigration
) fi
INNER JOIN 
(
SELECT DISTINCT municipality, state
FROM dim_airports 
) da
ON fi.city = da.municipality
AND fi.state = da.state
""").show(2)


# Roughly two thirds of our data in the fact table can be paired with data in the airport fact table. Considering that the immigration table only includes one month of data, this is quite good. We would normally use a left join

# Let's check the same thing with the demographics table.
# We expect the results of the join to be lower since the table doesn't include all cities in the united states.

# In[131]:


fact_immigration.show(2)
dim_demographics.show(2)


# In[132]:


#here are the distinct combinations of city and state in our fact table
spark.sql("""
SELECT COUNT(DISTINCT city, state)
FROM fact_immigration
""").show()

# and the combinations of city and state that are common to both the fact table and the demographics table
spark.sql("""
SELECT COUNT(*)
FROM
(
SELECT DISTINCT city, state
FROM fact_immigration
) fi
INNER JOIN 
(
SELECT DISTINCT City, state_code
FROM dim_demographics 
) da
ON fi.city = da.City
AND fi.state = da.state_code
""").show(2)


# A little less than half the cities are accounted for in our demographics database which isn't surprising but still quite good.

# We have the option of filtering out non existent city/state combinations from the data using a query similar to the one below:

# In[133]:


# We use a count to see how many rows we would keep using this strategy
spark.sql("""
SELECT COUNT(*)
FROM fact_immigration
WHERE CONCAT(city, state) IN (
    SELECT CONCAT(fi.city, fi.state)
    FROM
    (
        SELECT DISTINCT city, state
        FROM fact_immigration
    ) fi
    INNER JOIN 
    (
        SELECT DISTINCT municipality, state
        FROM dim_airports 
    ) da
    ON fi.city = da.municipality
    AND fi.state = da.state
)
""").show(2)


# We drop from 2 165 257 rows to 1 983 869 rows which is still quite good.
# However, we are assuming that our datasets are incomplete, especially the demographic data since it only includes cities with populations larger than 65,000 inhabitants and prefer to minimize the amount of data that is being left out of our final result.

# #### 4.3 Data dictionary 
# Create a data dictionary for your data model. For each field, provide a brief description of what the data is and where it came from. You can include the data dictionary in the notebook or in a separate file.

# #### Step 5: Complete Project Write Up
# 

# Consdiering the significant size of the immigration dataset (~ 3 million rows) for only a month, combined with the temperature, airport and demographic dataset, the most sensible technology choice for such an approach would be spark, especially if we were to process data over a longer period of time.

# We stated at the beginning of this project that we were interested in:
# * the effects of temperature aon the volume of travellers, 
# * the seasonality of travel 
# * the connection between the volume of travel and the number of entry ports (ie airports) 
# * the connection between the volume of travel and the demographics of various cities
# 
# None of these phenomenons require a rapid update of our data. A monthly or quarterly update would be sufficient for the needs of this study 

# #### Alternate requirement scenarios:

# How would our approach change if the problem had the following requireements:
# * The data was increased by 100x: Our data would be stored in an Amazon S3 bucket (instead of storing it in the EMR cluster along with the staging tables) and loaded to our staging tables. We would still use spark as it as our data processing platform since it is the best suited platform for very large datasets.
# * The data populates a dashboard that must be updated on a daily basis by 7am every day: We would use Apache Airflow to perform the ETL and data qualtiy validation.
# * The database needed to be accessed by 100+ people: Once the data is ready to be consumed, it would be stored in a postgres database on a redshift cluster that easily supports multiuser access. 

# In[ ]:




