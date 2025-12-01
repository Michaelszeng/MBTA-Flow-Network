import pandas as pd

df = pd.read_csv('data/MBTA_Bus_Ridership_by_Time_Period,_Season,_Route_Line,_and_Stop_-_Fall.csv', low_memory=False)

print("SATURDAY time periods in bus data:")
sat = df[df['day_type_name'] == 'saturday']
print(sat['time_period_name'].unique())

print("\nSUNDAY time periods in bus data:")
sun = df[df['day_type_name'] == 'sunday']
print(sun['time_period_name'].unique())

print("\nWEEKDAY time periods in bus data:")
week = df[df['day_type_name'] == 'weekday']
print(week['time_period_name'].unique())

