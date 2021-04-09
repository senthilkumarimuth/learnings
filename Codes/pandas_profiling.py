import pandas as pd
import pandas_profiling as pf

df=pd.read_excel('fda.xlsx')

d=pf.ProfileReport(df)

d.to_file(output_file="output.html")
