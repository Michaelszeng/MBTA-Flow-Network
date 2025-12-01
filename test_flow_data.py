import pandas as pd
from pathlib import Path
from data_extraction import add_bus_flow_to_edges, add_rail_flow_to_edges
from utils import load_gtfs_data, filter_routes_by_type

# Load GTFS data
data_folder = Path('data')
gtfs_folder = data_folder / 'MBTA_GTFS'
stops, stop_times, trips, routes = load_gtfs_data(gtfs_folder)

# Test rail flow data
print("=" * 60)
print("TESTING RAIL FLOW DATA")
print("=" * 60)

rail_ridership_file = data_folder / 'Fall_2023_MBTA_Rail_Ridership_Data_by_SDP_Time,_Period_Route_Line,_and_Stop.csv'
stop_orders_file = data_folder / 'MBTA_Rapid_Transit_Stop_Orders.csv'

# Load and inspect rail ridership data
rail_data = pd.read_csv(rail_ridership_file)
filtered_rail = rail_data[
    (rail_data['day_type_name'] == 'Saturday') &
    (rail_data['time_period_name'] == 'EVENING')
]
print(f"\nRail ridership records matching Saturday + EVENING: {len(filtered_rail)}")
if len(filtered_rail) > 0:
    print(f"Sample rail ridership data:")
    print(filtered_rail[['route_id', 'parent_station', 'dir_id', 'average_flow']].head())

# Create a simple test edge for rail
test_edges = pd.DataFrame([
    {'source_id': 'place-andrw', 'target_id': 'place-jfk', 'route_id': 'Red'}  # Andrew to JFK on Red Line
])
print(f"\nTest edge: {test_edges.to_dict('records')}")

edges_with_flow = add_rail_flow_to_edges(
    test_edges, rail_ridership_file, stop_orders_file,
    day_type='Saturday', time_period='EVENING'
)
print(f"Result: {edges_with_flow.to_dict('records')}")

# Test bus flow data
print("\n" + "=" * 60)
print("TESTING BUS FLOW DATA")
print("=" * 60)

bus_ridership_file = data_folder / 'MBTA_Bus_Ridership_by_Time_Period,_Season,_Route_Line,_and_Stop_-_Fall.csv'

# Load and inspect bus ridership data
bus_data = pd.read_csv(bus_ridership_file, low_memory=False)
filtered_bus = bus_data[
    (bus_data['day_type_name'] == 'saturday') &
    (bus_data['time_period_name'] == 'EVENING')
]
print(f"\nBus ridership records matching saturday + EVENING: {len(filtered_bus)}")
if len(filtered_bus) > 0:
    print(f"Sample bus ridership data for route 1:")
    sample = filtered_bus[filtered_bus['route_id'].astype(str) == '1']
    print(sample[['route_id', 'stop_id', 'stop_name', 'stop_sequence', 'average_load']].head(10))

# Create a test edge for bus route 1
# Need to find two consecutive stops first
if len(filtered_bus) > 0:
    route_1_data = filtered_bus[filtered_bus['route_id'].astype(str) == '1'].sort_values('stop_sequence')
    if len(route_1_data) > 1:
        stop1 = str(route_1_data.iloc[0]['stop_id'])
        stop2 = str(route_1_data.iloc[1]['stop_id'])
        
        test_bus_edges = pd.DataFrame([
            {'source_id': stop1, 'target_id': stop2, 'route_id': '1'}
        ])
        print(f"\nTest bus edge: {test_bus_edges.to_dict('records')}")
        
        bus_edges_with_flow = add_bus_flow_to_edges(
            test_bus_edges, bus_ridership_file, stops,
            day_type='saturday', time_period='EVENING'
        )
        print(f"Result: {bus_edges_with_flow.to_dict('records')}")

