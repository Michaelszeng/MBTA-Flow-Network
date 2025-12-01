import pandas as pd

print("Checking bus data...")
df = pd.read_csv('data/MBTA_Bus_Ridership_by_Time_Period,_Season,_Route_Line,_and_Stop_-_Fall.csv', low_memory=False)
print(f"Total rows: {len(df)}")

sat_evening = df[(df['day_type_name'] == 'saturday') & (df['time_period_name'] == 'EVENING')]
print(f"Saturday + EVENING: {len(sat_evening)}")

sat_offpeak = df[(df['day_type_name'] == 'saturday') & (df['time_period_name'] == 'OFF_PEAK')]
print(f"Saturday + OFF_PEAK: {len(sat_offpeak)}")

# Show unique combinations
print("\nUnique day_type_name values:", df['day_type_name'].unique())
print("Unique time_period_name values:", df['time_period_name'].unique())

