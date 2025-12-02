from pathlib import Path

import pandas as pd
from utils import filter_routes_by_type, load_gtfs_data

# Time period definitions (approximate based on MBTA standards)
TIME_PERIOD_RANGES = {
    'VERY_EARLY_MORNING': ('00:00:00', '05:59:59'),
    'EARLY_AM': ('06:00:00', '06:59:59'),
    'AM_PEAK': ('07:00:00', '08:59:59'),
    'MIDDAY_SCHOOL': ('09:00:00', '11:59:59'),
    'MIDDAY_BASE': ('12:00:00', '14:59:59'),
    'PM_PEAK': ('15:00:00', '17:59:59'),
    'EVENING': ('18:00:00', '21:59:59'),
    'LATE_EVENING': ('22:00:00', '23:59:59'),
    'NIGHT': ('00:00:00', '02:59:59'),
    'OFF_PEAK': ('09:00:00', '14:59:59')  # Fallback for generic off-peak
}

# Average headway (minutes between trains) for rail lines
AVERAGE_HEADWAY_MINUTES = 5.0  # Assume one train every 5 minutes

def calculate_trips_from_headway(time_period):
    """
    Calculate number of trips based on time period duration and assumed headway.
    Simple approach: assumes one train every AVERAGE_HEADWAY_MINUTES in each direction.
    
    Parameters:
    -----------
    time_period : str
        Time period name (e.g., 'PM_PEAK', 'EVENING')
    
    Returns:
    --------
    int
        Estimated number of trips in one direction during the time period
    """
    # Get time range for the period
    if time_period not in TIME_PERIOD_RANGES:
        print(f"    Warning: Time period '{time_period}' not defined. Using 4-hour default.")
        duration_minutes = 240  # 4 hours default
    else:
        start_time, end_time = TIME_PERIOD_RANGES[time_period]
        # Parse times to get duration
        start_parts = start_time.split(':')
        end_parts = end_time.split(':')
        start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
        end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])
        
        # Handle case where end_time is earlier than start_time (crosses midnight)
        if end_minutes < start_minutes:
            duration_minutes = (24 * 60 - start_minutes) + end_minutes
        else:
            duration_minutes = end_minutes - start_minutes
    
    # Calculate number of trips: duration / headway
    num_trips = int(duration_minutes / AVERAGE_HEADWAY_MINUTES)
    
    print(f"    Using simplified trip calculation: {num_trips} trips per direction")
    print(f"    ({duration_minutes} minutes ÷ {AVERAGE_HEADWAY_MINUTES} min/train = ~{num_trips} trains)")
    
    return num_trips


def add_bus_flow_to_edges(edges_df, ridership_filepath, stops_df,
                          day_type='weekday', time_period='PM_PEAK',
                          normalize=False):
    """
    Add ridership flow data to edges DataFrame for bus routes.
    
    Parameters:
    -----------
    edges_df : pd.DataFrame
        DataFrame with columns: source_id, target_id, route_id
    ridership_filepath : str or Path
        Path to bus ridership CSV
    stops_df : pd.DataFrame
        GTFS stops data to map stop IDs to parent stations
    day_type : str
        Day type to filter (e.g., 'weekday', 'saturday', 'sunday')
    time_period : str
        Time period to filter (e.g., 'PM_PEAK', 'AM_PEAK', 'OFF_PEAK')
    normalize : bool
        Not used for buses (average_load is already per-bus)
    
    Returns:
    --------
    pd.DataFrame
        edges_df with added 'flow' column (average_load per bus)
    """
    # Load ridership data
    ridership = pd.read_csv(ridership_filepath)
    ridership = ridership[
        (ridership['day_type_name'] == day_type) &
        (ridership['time_period_name'] == time_period)
    ].copy()
    
    # Create a mapping from stop_id to parent_station for bus stops
    stops_dict = stops_df.set_index('stop_id')['parent_station'].to_dict()
    
    # Function to convert stop_id to parent_station (or keep as-is if no parent)
    def get_parent_or_self(stop_id):
        stop_id_str = str(stop_id)
        if stop_id_str in stops_dict:
            parent = stops_dict[stop_id_str]
            if pd.notna(parent) and parent != '':
                return parent
        return stop_id_str
    
    # Add parent_station column to ridership data
    ridership['parent_stop_id'] = ridership['stop_id'].apply(get_parent_or_self)
    
    # Copy edges_df to avoid modifying original
    edges_with_flow = edges_df.copy()
    edges_with_flow['flow'] = 0.0
    
    # Process each edge
    for idx, edge in edges_with_flow.iterrows():
        source_id = edge['source_id']
        target_id = edge['target_id']
        route_id = edge['route_id']
        
        # Convert route_id to string to match ridership data
        route_id_str = str(route_id)
        
        # Find matching bus ridership data
        # Group by route_variant and direction to handle different patterns
        route_data = ridership[ridership['route_id'].astype(str) == route_id_str]
        
        if len(route_data) == 0:
            continue
        
        # Try to find consecutive stops in any route variant/direction
        for (variant, direction), variant_data in route_data.groupby(['route_variant', 'direction_id']):
            variant_data = variant_data.sort_values('stop_sequence')
            
            # Find source and target stops using parent_stop_id
            source_records = variant_data[variant_data['parent_stop_id'] == source_id]
            target_records = variant_data[variant_data['parent_stop_id'] == target_id]
            
            if len(source_records) == 0 or len(target_records) == 0:
                continue
            
            # Check if they are consecutive
            for _, source_rec in source_records.iterrows():
                for _, target_rec in target_records.iterrows():
                    if target_rec['stop_sequence'] == source_rec['stop_sequence'] + 1:
                        # Found consecutive stops! Use the load at source as flow
                        flow_value = source_rec['average_load']
                        edges_with_flow.at[idx, 'flow'] = flow_value
                        break
                if edges_with_flow.at[idx, 'flow'] > 0:
                    break
            
            if edges_with_flow.at[idx, 'flow'] > 0:
                break
    
    # Note: Bus data already represents average_load (per-bus), so normalization not needed
    # This section is kept for compatibility but normalize should be False for buses
    
    return edges_with_flow


def add_rail_flow_to_edges(edges_df, ridership_filepath, stop_orders_filepath, 
                           day_type='Weekday', time_period='PM_PEAK',
                           normalize=True):
    """
    Add ridership flow data to edges DataFrame for rail lines.
    
    Parameters:
    -----------
    edges_df : pd.DataFrame
        DataFrame with columns: source_id, target_id, route_id
    ridership_filepath : str or Path
        Path to Fall 2023 rail ridership CSV
    stop_orders_filepath : str or Path
        Path to MBTA_Rapid_Transit_Stop_Orders.csv
    day_type : str
        Day type to filter (e.g., 'Weekday', 'Saturday', 'Sunday')
    time_period : str
        Time period to filter (e.g., 'PM_PEAK', 'AM_PEAK', 'OFF_PEAK')
    normalize : bool
        If True, normalize flow to average passengers per train (assumes 5-minute headways)
    
    Returns:
    --------
    pd.DataFrame
        edges_df with added 'flow' column (normalized if normalize=True)
    """
    # Load ridership data
    ridership = pd.read_csv(ridership_filepath)
    ridership = ridership[
        (ridership['day_type_name'] == day_type) &
        (ridership['time_period_name'] == time_period)
    ].copy()
    
    # Load stop orders to understand sequences
    stop_orders = pd.read_csv(stop_orders_filepath)
    
    # Create a mapping of route colors to route names for Green Line
    # (ridership data uses 'Green' but GTFS might use 'Green-B', 'Green-C', etc.)
    route_mapping = {
        'Green-B': 'Green',
        'Green-C': 'Green',
        'Green-D': 'Green',
        'Green-E': 'Green',
        'Mattapan': 'Red'  # Mattapan is part of Red Line
    }
    
    # Map direction codes: ridership uses EB/WB/NB/SB, GTFS uses 0/1
    # For each route, we need to figure out which direction is which
    direction_mapping = {
        'Blue': {0: 'WB', 1: 'EB'},  # 0=Wonderland->Bowdoin (WB), 1=Bowdoin->Wonderland (EB)
        'Orange': {0: 'SB', 1: 'NB'},  # 0=Oak Grove->Forest Hills (SB), 1=Forest Hills->Oak Grove (NB)
        'Red': {0: 'SB', 1: 'NB'},  # 0=Alewife->Ashmont/Braintree (SB), 1=Ashmont/Braintree->Alewife (NB)
        'Green': {0: 'WB', 1: 'EB'}  # Green Line branches
    }
    
    # Copy edges_df to avoid modifying original
    edges_with_flow = edges_df.copy()
    edges_with_flow['flow'] = 0.0
    edges_with_flow['direction_id'] = -1  # Track which direction this edge belongs to
    
    # Process each edge
    for idx, edge in edges_with_flow.iterrows():
        source_id = edge['source_id']
        target_id = edge['target_id']
        route_id = edge['route_id']
        
        # Map route_id if needed (for Green Line variants)
        ridership_route = route_mapping.get(route_id, route_id)
        
        # Find the direction by checking stop orders
        # Look for this edge in stop orders for each direction
        edge_flows = []
        edge_directions = []
        
        for direction in [0, 1]:
            # Get all stop sequences for this route and direction
            # Group by (first_stop, last_stop) to handle branches separately
            route_stops = stop_orders[
                (stop_orders['route_id'] == ridership_route) &
                (stop_orders['direction_id'] == direction)
            ]
            
            if len(route_stops) == 0:
                continue
            
            # Group by route pattern (first_stop, last_stop) to handle branches
            for pattern_key, pattern_stops in route_stops.groupby(['first_stop', 'last_stop']):
                pattern_stops = pattern_stops.sort_values('stop_order')
                stop_list = pattern_stops['stop_id'].tolist()
                
                try:
                    source_idx = stop_list.index(source_id)
                    if source_idx + 1 < len(stop_list) and stop_list[source_idx + 1] == target_id:
                        # Found the edge in this direction!
                        dir_code = direction_mapping.get(ridership_route, {}).get(direction)
                        if dir_code:
                            # Look up ridership for this station, route, and direction
                            flow_data = ridership[
                                (ridership['route_id'] == ridership_route) &
                                (ridership['parent_station'] == source_id) &
                                (ridership['dir_id'] == dir_code)
                            ]
                            
                            if len(flow_data) > 0:
                                flow_value = flow_data['average_flow'].iloc[0]
                                edge_flows.append(flow_value)
                                edge_directions.append(direction)
                                break  # Found a match, no need to check other patterns
                except ValueError:
                    # source_id not in this pattern's stop list, try next pattern
                    continue
        
        # Store flow and direction
        # If edge exists in both directions, sum flows and use first direction found
        if edge_flows:
            edges_with_flow.at[idx, 'flow'] = sum(edge_flows)
            edges_with_flow.at[idx, 'direction_id'] = edge_directions[0] if edge_directions else -1
    
    # Normalize by trip count if requested
    if normalize:
        print("  Normalizing rail flow by estimated trip count...")
        # Calculate trips based on headway assumption (simple approach)
        trips_per_direction = calculate_trips_from_headway(time_period)
        
        # Normalize: flow per train = total flow / number of trips
        # The flow data is already directional, so we divide by trips in that direction
        for idx, edge in edges_with_flow.iterrows():
            flow = edge.get('flow', 0)
            if flow > 0:
                edges_with_flow.at[idx, 'flow'] = flow / trips_per_direction
        
        print(f"  Normalized: average flow per train = total flow / {trips_per_direction} trips")
    
    return edges_with_flow