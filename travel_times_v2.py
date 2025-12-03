"""
Simple helper functions for adding MBTA travel time data to graph edges.
"""

from pathlib import Path

import pandas as pd


def load_travel_times(data_folder):
    """
    Load and aggregate travel time data for CONSECUTIVE stops only.
    
    The raw data contains cumulative travel times from each starting station to all 
    downstream stations. This function filters to only consecutive stop pairs by
    grouping by trip and identifying adjacent stops.
    
    Parameters:
    -----------
    data_folder : Path or str
        Path to the data folder containing TravelTimes_2025 subdirectory
    
    Returns:
    --------
    pd.DataFrame
        Aggregated travel times with columns:
        - from_parent_station: Source station ID
        - to_parent_station: Destination station ID  
        - travel_time_sec: Median travel time in seconds (for consecutive stops only)
    """
    data_folder = Path(data_folder)
    travel_times_folder = data_folder / 'TravelTimes_2025'
    
    if not travel_times_folder.exists():
        raise FileNotFoundError(f"Travel times folder not found: {travel_times_folder}")
    
    # Load all CSV files in the folder
    all_files = list(travel_times_folder.glob('*.csv'))
    if not all_files:
        raise ValueError("No travel time files found!")
    
    print(f"  Loading {len(all_files)} travel time files...")
    all_data = []
    
    for filepath in all_files:
        df = pd.read_csv(filepath)
        # Keep full data including trip_id and departure times to identify consecutive stops
        all_data.append(df[['trip_id', 'from_parent_station', 'to_parent_station', 
                            'from_stop_departure_sec', 'travel_time_sec']])
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"  Loaded {len(combined_df):,} travel time records (all station pairs)")
    
    # Filter to consecutive stops only
    # Strategy: Group by trip_id, sort by departure time, and keep only records
    # where the next stop in the trip matches the to_parent_station
    consecutive_pairs = []
    
    for trip_id, trip_group in combined_df.groupby('trip_id'):
        # Sort by departure time to get stop sequence
        trip_sorted = trip_group.sort_values('from_stop_departure_sec')
        
        # For each stop, check if there's a record starting from the destination
        # of the current record at the very next departure time
        for idx in range(len(trip_sorted) - 1):
            current = trip_sorted.iloc[idx]
            next_stop = trip_sorted.iloc[idx + 1]
            
            # If current destination matches next origin, this is a consecutive pair
            if current['to_parent_station'] == next_stop['from_parent_station']:
                consecutive_pairs.append({
                    'from_parent_station': current['from_parent_station'],
                    'to_parent_station': current['to_parent_station'],
                    'travel_time_sec': current['travel_time_sec']
                })
    
    consecutive_df = pd.DataFrame(consecutive_pairs)
    print(f"  Filtered to {len(consecutive_df):,} consecutive stop pairs")
    
    # Aggregate by station pairs using median (robust to outliers)
    agg_df = consecutive_df.groupby(
        ['from_parent_station', 'to_parent_station']
    )['travel_time_sec'].median().reset_index()
    
    print(f"  Aggregated to {len(agg_df)} unique consecutive station pairs")
    print(f"  Average travel time: {agg_df['travel_time_sec'].mean():.1f} seconds ({agg_df['travel_time_sec'].mean()/60:.1f} min)")
    
    return agg_df


def add_travel_times_to_graph(G, travel_times_df):
    """
    Add travel time weights to graph edges.
    
    Adds 'travel_time_sec' attribute to edges that have matching travel time data.
    Applies to both directions (A→B and B→A use the same travel time).
    
    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        Graph to add travel times to (modified in place)
    travel_times_df : pd.DataFrame
        Travel times from load_travel_times()
    
    Returns:
    --------
    int
        Number of edges that received travel time data
    """
    # Create lookup dictionary
    travel_times_dict = {}
    for _, row in travel_times_df.iterrows():
        from_station = row['from_parent_station']
        to_station = row['to_parent_station']
        travel_time = row['travel_time_sec']
        
        # Add both directions
        travel_times_dict[(from_station, to_station)] = travel_time
        travel_times_dict[(to_station, from_station)] = travel_time
    
    # Add to graph edges
    edges_matched = 0
    for source, target in G.edges():
        if (source, target) in travel_times_dict:
            G[source][target]['travel_time_sec'] = travel_times_dict[(source, target)]
            edges_matched += 1
    
    print(f"  Added travel times to {edges_matched}/{G.number_of_edges()} edges")
    
    return edges_matched



