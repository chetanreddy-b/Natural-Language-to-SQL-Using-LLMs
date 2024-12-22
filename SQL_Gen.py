import pandas as pd
import openai
import os
from groq import Groq
import chardet
import shutil

base_dir = '/content/drive/MyDrive/RTS_Chetan_Chathurvedhi'

sql_dir = base_dir + '/SQL'

sql_path = {
    "Crime_Simple":'/crime_simple.csv',
    "Crime_Complex":'/crime_complex.csv',
    "Shooting_Simple":'/shooting_simple.csv',
    "Shooting_Complex":'/shooting_complex.csv',
    "Housing_Simple":'/housing_simple.csv',
    "Housing_Complex":'/housing_complex.csv',
    "Complete":'/complete.csv',
    "Simple":'/simple.csv',
    "Complex":'/complex.csv'
}

df_crime_simple = pd.read_csv(sql_dir+sql_path["Crime_Simple"])
df_crime_complex = pd.read_csv(sql_dir+sql_path["Crime_Complex"])
df_shooting_simple = pd.read_csv(sql_dir+sql_path["Shooting_Simple"])
df_shooting_complex = pd.read_csv(sql_dir+sql_path["Shooting_Complex"])
df_housing_simple = pd.read_csv(sql_dir+sql_path["Housing_Simple"])
df_housing_complex = pd.read_csv(sql_dir+sql_path["Housing_Complex"])

df_simple = pd.concat([df_crime_simple, df_shooting_simple, df_housing_simple], ignore_index=True)
df_complex = pd.concat([df_crime_complex, df_shooting_complex, df_housing_complex], ignore_index=True)

df_complete = pd.concat([df_simple, df_complex], ignore_index=True)

# Save the dataframes

df_simple.to_csv(sql_dir+'/simple.csv', index=False)
df_complex.to_csv(sql_dir+'/complex.csv', index=False)
df_complete.to_csv(sql_dir+'/complete.csv', index=False)