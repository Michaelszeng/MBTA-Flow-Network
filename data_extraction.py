from pathlib import Path

import pandas as pd
from utils import filter_routes_by_type, load_gtfs_data


def add_bus_flow_to_edges(edges_df, ridership_filepath, stops_df,
                          day_type='weekday', time_period='PM_PEAK'):
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
    
    Returns:
    --------
    pd.DataFrame
        edges_df with added 'flow' column
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
    
    return edges_with_flow


def add_rail_flow_to_edges(edges_df, ridership_filepath, stop_orders_filepath, 
                           day_type='Weekday', time_period='PM_PEAK'):
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
    
    Returns:
    --------
    pd.DataFrame
        edges_df with added 'flow' column
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
                                break  # Found a match, no need to check other patterns
                except ValueError:
                    # source_id not in this pattern's stop list, try next pattern
                    continue
        
        # Aggregate flows (if edge exists in both directions, sum them)
        # Or use the maximum, or average - depends on what you want
        if edge_flows:
            edges_with_flow.at[idx, 'flow'] = sum(edge_flows)  # Sum bidirectional flows
    
    return edges_with_flow