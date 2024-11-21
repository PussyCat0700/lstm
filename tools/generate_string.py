import pandas as pd


df = pd.read_csv("/data1/yfliu/solar_baseline/solar/china_data/china_info.csv")
arr = []
f = open('string.txt', 'w')
for idx, row in df.iterrows():
    plant_no = row["PLANT_NO"]
    f.write(str(plant_no))
    f.write(',')
f.close()