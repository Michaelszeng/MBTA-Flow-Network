"""
Calculate travel times between consecutive stops from GTFS stop_times.txt data.

This module parses GTFS scheduled arrival/departure times to compute travel times
between consecutive stops on each trip, then aggregates across all trips.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def parse_gtfs_time(time_str):
    """
    Parse GTFS time string (HH:MM:SS) to seconds since midnight.
    
    GTFS times can exceed 24:00:00 for trips that continue past midnight
    (e.g., "25:30:00" = 1:30 AM the next day).
    
    Parameters:
    -----------
    time_str : str
        Time string in HH:MM:SS format
    
    Returns:
    --------
    int
        Seconds since midnight (can be > 86400 for post-midnight times)
    """
    if pd.isna(time_str) or time_str == '':
        return None
    
    parts = time_str.strip().split(':')
    if len(parts) != 3:
        return None
    
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, TypeError):
        return None


def load_travel_times_from_gtfs(gtfs_folder, stops_df=None):
    """
    Calculate travel times between consecutive stops from GTFS stop_times.txt.
    
    For each trip, this function:
    1. Sorts stops by stop_sequence
    2. Calculates time difference between consecutive stops
    3. Aggregates across all trips using median (robust to outliers)
    
    Parameters:
    -----------
    gtfs_folder : Path or str
        Path to the GTFS data folder containing stop_times.txt
    stops_df : pd.DataFrame, optional
        GTFS stops dataframe for consolidating to parent stations.
        If not provided, uses stop_id directly.
    
    Returns:
    --------
    pd.DataFrame
        Travel times with columns:
        - from_stop: Source stop/station ID
        - to_stop: Destination stop/station ID
        - travel_time_sec: Median travel time in seconds
        - num_samples: Number of trip samples used for calculation
        - travel_time_std: Standard deviation of travel times (variability indicator)
    """
    gtfs_folder = Path(gtfs_folder)
    stop_times_file = gtfs_folder / 'stop_times.txt'
    
    if not stop_times_file.exists():
        raise FileNotFoundError(f"stop_times.txt not found: {stop_times_file}")
    
    print(f"Loading stop_times.txt from {gtfs_folder}...")
    
    # Load stop_times - only the columns we need
    stop_times = pd.read_csv(
        stop_times_file,
        usecols=['trip_id', 'stop_id', 'stop_sequence', 'arrival_time', 'departure_time'],
        dtype={'trip_id': str, 'stop_id': str, 'stop_sequence': int},
        low_memory=False
    )
    
    print(f"  Loaded {len(stop_times):,} stop time records")
    print(f"  Unique trips: {stop_times['trip_id'].nunique():,}")
    
    # Parse times to seconds
    print("  Parsing arrival and departure times...")
    stop_times['arrival_sec'] = stop_times['arrival_time'].apply(parse_gtfs_time)
    stop_times['departure_sec'] = stop_times['departure_time'].apply(parse_gtfs_time)
    
    # Drop rows with invalid times
    initial_count = len(stop_times)
    stop_times = stop_times.dropna(subset=['arrival_sec', 'departure_sec'])
    dropped = initial_count - len(stop_times)
    if dropped > 0:
        print(f"  Dropped {dropped:,} records with invalid times")
    
    # Consolidate to parent stations if stops_df provided
    if stops_df is not None:
        print("  Consolidating to parent stations...")
        # Create stop_id -> parent_station mapping
        stops_dict = {}
        for _, row in stops_df.iterrows():
            stop_id = str(row['stop_id'])
            parent = row.get('parent_station', '')
            # Use parent if it exists, otherwise use stop_id itself
            if pd.notna(parent) and parent != '':
                stops_dict[stop_id] = str(parent)
            else:
                stops_dict[stop_id] = stop_id
        
        # Map stop_ids to parent stations
        stop_times['stop_id'] = stop_times['stop_id'].astype(str).map(
            lambda x: stops_dict.get(x, x)
        )
    
    # Sort by trip and stop sequence
    stop_times = stop_times.sort_values(['trip_id', 'stop_sequence']).reset_index(drop=True)
    
    # Calculate travel times between consecutive stops using vectorized operations
    print("  Calculating travel times between consecutive stops...")
    
    # Create shifted columns for next stop in each trip
    stop_times['next_trip_id'] = stop_times['trip_id'].shift(-1)
    stop_times['next_stop_id'] = stop_times['stop_id'].shift(-1)
    stop_times['next_arrival_sec'] = stop_times['arrival_sec'].shift(-1)
    
    # Only keep rows where next row is part of same trip (consecutive stops)
    consecutive = stop_times[stop_times['trip_id'] == stop_times['next_trip_id']].copy()
    
    # Calculate travel time
    consecutive['travel_time_sec'] = consecutive['next_arrival_sec'] - consecutive['departure_sec']
    
    # Filter to reasonable travel times (10 sec to 2 hours)
    consecutive = consecutive[
        (consecutive['travel_time_sec'] >= 10) & 
        (consecutive['travel_time_sec'] <= 7200)
    ]
    
    # Keep only relevant columns
    travel_times_df = consecutive[['stop_id', 'next_stop_id', 'travel_time_sec', 'trip_id']].copy()
    travel_times_df.columns = ['from_stop', 'to_stop', 'travel_time_sec', 'trip_id']
    
    print(f"  Calculated {len(travel_times_df):,} consecutive stop pair travel times")
    
    # Aggregate by stop pairs
    print("  Aggregating across trips...")
    agg_df = travel_times_df.groupby(['from_stop', 'to_stop']).agg({
        'travel_time_sec': ['median', 'mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = ['from_stop', 'to_stop', 'travel_time_sec', 
                      'travel_time_mean', 'travel_time_std', 'num_samples']
    
    # Round to nearest second
    agg_df['travel_time_sec'] = agg_df['travel_time_sec'].round().astype(int)
    agg_df['travel_time_mean'] = agg_df['travel_time_mean'].round(1)
    agg_df['travel_time_std'] = agg_df['travel_time_std'].round(1)
    
    print(f"\n  ✓ Aggregated to {len(agg_df)} unique consecutive stop pairs")
    print(f"  ✓ Median travel time: {agg_df['travel_time_sec'].median():.0f} seconds ({agg_df['travel_time_sec'].median()/60:.1f} min)")
    print(f"  ✓ Average travel time: {agg_df['travel_time_sec'].mean():.1f} seconds ({agg_df['travel_time_sec'].mean()/60:.1f} min)")
    print(f"  ✓ Average samples per pair: {agg_df['num_samples'].mean():.0f} trips")
    
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
        Travel times from load_travel_times_from_gtfs()
    
    Returns:
    --------
    int
        Number of edges that received travel time data
    """
    # Create lookup dictionary
    travel_times_dict = {}
    for _, row in travel_times_df.iterrows():
        from_stop = row['from_stop']
        to_stop = row['to_stop']
        travel_time = row['travel_time_sec']
        
        # Add both directions
        travel_times_dict[(from_stop, to_stop)] = travel_time
        travel_times_dict[(to_stop, from_stop)] = travel_time
    
    # Add to graph edges
    edges_matched = 0
    edges_missing = []
    
    for source, target in G.edges():
        if (source, target) in travel_times_dict:
            G[source][target]['travel_time_sec'] = travel_times_dict[(source, target)]
            edges_matched += 1
        else:
            edges_missing.append((source, target))
    
    coverage_pct = (edges_matched / G.number_of_edges() * 100) if G.number_of_edges() > 0 else 0
    print(f"\n  ✓ Added travel times to {edges_matched}/{G.number_of_edges()} edges ({coverage_pct:.1f}% coverage)")
    
    if edges_missing and len(edges_missing) <= 10:
        print(f"\n  Missing travel time data for {len(edges_missing)} edges:")
        for src, tgt in edges_missing[:10]:
            print(f"    - {src} → {tgt}")
    elif edges_missing:
        print(f"\n  Missing travel time data for {len(edges_missing)} edges")
    
    return edges_matched

if __name__ == '__main__':
    """
    Example usage: Calculate travel times from GTFS and optionally compare with actual times.
    """
    from utils import load_gtfs_data

    # Paths
    data_folder = Path('data')
    gtfs_folder = data_folder / 'MBTA_GTFS'
    
    # Load GTFS data
    print("Loading GTFS data...")
    stops, stop_times, trips, routes = load_gtfs_data(gtfs_folder)
    
    # Calculate travel times from GTFS
    print("\n" + "="*70)
    print("CALCULATING TRAVEL TIMES FROM GTFS SCHEDULES")
    print("="*70)
    travel_times = load_travel_times_from_gtfs(gtfs_folder, stops_df=stops)
    
    # Show some examples
    print("\n" + "="*70)
    print("SAMPLE TRAVEL TIMES (first 20)")
    print("="*70)
    print(travel_times.head(20).to_string(index=False))
    
    # Save to CSV
    output_file = data_folder / 'gtfs_scheduled_travel_times.csv'
    travel_times.to_csv(output_file, index=False)
    print(f"\n✓ Saved travel times to {output_file}")
