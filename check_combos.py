import pandas as pd

df = pd.read_csv('data/MBTA_Bus_Ridership_by_Time_Period,_Season,_Route_Line,_and_Stop_-_Fall.csv', low_memory=False)
combos = df.groupby(['day_type_name', 'time_period_name']).size().reset_index(name='count')
print('Bus day_type + time_period combinations:')
for idx, row in combos.iterrows():
    print(f"  {row['day_type_name']:10} + {row['time_period_name']:20} : {row['count']:,} records")

