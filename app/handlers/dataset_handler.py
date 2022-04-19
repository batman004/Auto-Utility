import pandas as pd
import numpy as np
from ..utility.pre_processing import clean

def IE_brand(brand):
  path = "data/Scraped_Car_Review_" + brand + ".csv"
  df = pd.read_csv(path,delimiter=',', nrows = 100)
  df['Review_clean'] = df['Review'].apply(clean)
  df['Review_clean'][2]
  reviews = df['Review_clean'][0:5]
  reviews = np.array(reviews)
  df2 = df["Rating"].mean()
  return (df2/5) * 10

def get_dataframe_head(brand):
  path = "data/Scraped_Car_Review_" + brand + ".csv"
  df = pd.read_csv(path,delimiter=',', nrows = 5)
  return df.to_dict()

