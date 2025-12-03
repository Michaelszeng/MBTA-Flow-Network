"""
Rerouting logic for MBTA network simulation.

This module implements a Gravity Model-based approach to synthesize Origin-Destination (OD)
demand from station boardings/alightings and reroute that demand through the network
when lines are closed.
"""

import networkx as nx
import numpy as np
import pandas as pd
from shortest_path import calculate_distance


def synthesize_od_demand(stops_data, decay_factor=1.5):
    """
    Synthesize trip demand between stations using a Gravity Model.
    
    T_ij = K * (Ons_i * Offs_j) / (Distance_ij ^ decay_factor)
    
    Parameters:
    -----------
    stops_data : list of dicts
        List of stop objects containing:
        - 'stop_id': Station ID
        - 'ons': Average boardings
        - 'offs': Average alightings
        - 'lat', 'lon': Coordinates
    decay_factor : float
        Exponent for distance decay (friction factor).
        Higher values discourage long trips more strongly.
        
    Returns:
    --------
    list of dicts : Synthesized trips [{'source': id, 'target': id, 'flow': count}, ...]
    """
    trips = []
    stops = sorted(stops_data, key=lambda x: x['stop_id'])
    n = len(stops)
    
    total_ons = sum(s['ons'] for s in stops)
    total_offs = sum(s['offs'] for s in stops)
    
    print(f"    Gravity model parameters:")
    print(f"      Stations: {n}")
    print(f"      Decay factor: {decay_factor}")
    print(f"      Total productions (ons): {total_ons:.0f}")
    print(f"      Total attractions (offs): {total_offs:.0f}")
    
    # Pre-calculate distances and raw gravity values
    gravity_matrix = np.zeros((n, n))
    
    for i in range(n):
        source = stops[i]
        if source['ons'] <= 0:
            continue
            
        for j in range(n):
            target = stops[j]
            if i == j or target['offs'] <= 0:
                continue
                
            dist = calculate_distance(source['lon'], source['lat'], target['lon'], target['lat'])
            if dist == 0:
                dist = 0.001  # Avoid division by zero
                
            # Gravity formula numerator: Mass_i * Mass_j
            # Denominator: Distance^alpha
            attraction = (source['ons'] * target['offs'])
            impedance = dist ** decay_factor
            
            gravity_matrix[i][j] = attraction / impedance
            
    # Normalize rows (Production Constrained Gravity Model)
    # We ensure the sum of trips leaving station i equals its total 'ons'
    for i in range(n):
        source = stops[i]
        total_gravity = np.sum(gravity_matrix[i])
        
        if total_gravity > 0:
            # K factor for this row
            k = source['ons'] / total_gravity
            
            for j in range(n):
                if gravity_matrix[i][j] > 0:
                    flow = k * gravity_matrix[i][j]
                    if flow >= 1.0:  # Filter negligible trips
                        trips.append({
                            'source_id': stops[i]['stop_id'],
                            'target_id': stops[j]['stop_id'],
                            'flow': flow
                        })
    
    # Summary
    total_synthesized = sum(t['flow'] for t in trips)
    print(f"    Generated {len(trips)} OD pairs")
    print(f"    Total synthesized flow: {total_synthesized:.0f} passengers")
    print(f"    Conservation check: {total_ons:.0f} ons -> {total_synthesized:.0f} trips (should be equal)")
                        
    return trips

def find_nearest_active_node(closed_node_id, G, pos, stops_df, closed_routes, max_dist=0.015):
    """
    Find the nearest node in the active graph G to a closed node.
    
    Parameters:
    -----------
    closed_node_id : str
        ID of the closed station
    G : nx.DiGraph
        The active network graph
    pos : dict
        Positions of nodes in G
    stops_df : pd.DataFrame
        Stops data to get position of closed_node
    closed_routes : list
        List of route IDs that are closed (to avoid mapping to other closed nodes)
    max_dist : float
        Maximum walking distance in degrees (approx 1 mile)
        
    Returns:
    --------
    str or None : ID of nearest active node
    """
    # Get position of closed node
    if closed_node_id in pos:
        # If node is in G, it might still be active (shared station like Park St)
        # We need to check if it serves any active routes
        node_data = G.nodes[closed_node_id]
        # Note: We'd need route info on nodes to check this perfectly
        # For now, if it's in the graph, we assume it's usable (transfer station)
        return closed_node_id
        
    # Look up in stops dataframe if not in pos map
    stop_row = stops_df[stops_df['stop_id'] == closed_node_id]
    if len(stop_row) == 0:
        # Try parent station mapping if needed
        return None
        
    c_lat = stop_row.iloc[0]['stop_lat']
    c_lon = stop_row.iloc[0]['stop_lon']
    
    nearest_node = None
    min_dist = float('inf')
    
    for node in G.nodes():
        if node not in pos:
            continue
            
        n_lon, n_lat = pos[node]
        dist = calculate_distance(c_lon, c_lat, n_lon, n_lat)
        
        if dist < min_dist and dist <= max_dist:
            min_dist = dist
            nearest_node = node
            
    return nearest_node

def reroute_demand(G, pos, stops_df, closed_line_edges, closed_routes, 
                  ridership_data=None, stops_dict=None, normalization_factor=1.0):
    """
    Main function to synthesize demand from closed lines and reroute it.
    
    Parameters:
    -----------
    normalization_factor : float
        Factor to divide raw ridership by (to match graph's flow units, e.g., per train)
    """
    print("\n  ========================================")
    print("  === REROUTING ANALYSIS (GRAVITY MODEL) ===")
    print("  ========================================")
    
    # 1. Identify stations on the closed line(s)
    # We need unique stations that were serving the closed routes
    closed_stations = set()
    # We can get this from the original edges before they were removed
    # But since we don't have them passed directly, we might need to infer or pass them
    # For now, let's extract them from the ridership data if available
    
    if ridership_data is None:
        print("  Error: Ridership data required for rerouting.")
        return 0
    
    print(f"\n  Step 1: Extracting demand from closed routes...")
    print(f"  Closed routes: {closed_routes}")
    print(f"  Normalization factor: {normalization_factor}")
        
    # Filter ridership for the closed routes
    # Handle both exact and prefix matching for closed routes
    # matches_closure function logic duplicated here for dataframe filtering
    if isinstance(closed_routes, str):
        closure_val = closed_routes
        mask = (ridership_data['route_id'] == closure_val) | \
               (ridership_data['route_id'].astype(str).str.startswith(closure_val + "-"))
    else:
        # Fallback if list
        mask = ridership_data['route_id'].isin(closed_routes)
        
    closed_line_data = ridership_data[mask].copy()
    
    if len(closed_line_data) == 0:
        print(f"  No ridership data found for closed routes: {closed_routes}")
        return 0
        
    print(f"  Found {len(closed_line_data)} ridership records for closed routes")
    print(f"  Unique route IDs in closed data: {sorted(closed_line_data['route_id'].unique())}")
    
    # 2. Aggregate Ons/Offs per station
    # A station might appear multiple times (diff directions, time periods)
    # We want average total daily (or time period) load per station
    print(f"\n  Step 2: Aggregating boardings/alightings per station...")
    station_stats = closed_line_data.groupby('parent_stop_id').agg({
        'average_ons': 'sum',
        'average_offs': 'sum'
    }).reset_index()
    
    print(f"  Aggregated {len(station_stats)} unique stations")
    print(f"  Total ons across all stations: {station_stats['average_ons'].sum():.0f}")
    print(f"  Total offs across all stations: {station_stats['average_offs'].sum():.0f}")
    
    # Show top 5 stations by boardings
    top_ons = station_stats.nlargest(5, 'average_ons')
    print(f"\n  Top 5 stations by boardings:")
    for idx, row in top_ons.iterrows():
        print(f"    - {row['parent_stop_id']}: {row['average_ons']:.0f} ons, {row['average_offs']:.0f} offs")
    
    # Add coordinates
    stops_data = []
    missing_coords = 0
    for _, row in station_stats.iterrows():
        stop_id = row['parent_stop_id']
        if stop_id in stops_dict:
            stop_info = stops_dict[stop_id]
            stops_data.append({
                'stop_id': stop_id,
                'ons': row['average_ons'],
                'offs': row['average_offs'],
                'lat': stop_info['stop_lat'],
                'lon': stop_info['stop_lon']
            })
        else:
            missing_coords += 1
    
    if missing_coords > 0:
        print(f"  Warning: {missing_coords} stations missing coordinate data")
            
    print(f"\n  Step 3: Synthesizing OD demand using Gravity Model...")
    print(f"  Using {len(stops_data)} stations with coordinate data")
    
    # 3. Synthesize Trips
    synthesized_trips = synthesize_od_demand(stops_data)
    total_trips = sum(t['flow'] for t in synthesized_trips)
    print(f"  Generated {len(synthesized_trips)} unique OD pairs")
    print(f"  Total synthesized passenger flow: {total_trips:.0f} passengers")
    
    # Show top 5 OD pairs by volume
    if synthesized_trips:
        sorted_trips = sorted(synthesized_trips, key=lambda x: x['flow'], reverse=True)
        print(f"\n  Top 5 OD pairs by volume:")
        for trip in sorted_trips[:5]:
            print(f"    - {trip['source_id']} -> {trip['target_id']}: {trip['flow']:.0f} passengers")
    
    # 4. Reroute Trips
    print(f"\n  Step 4: Rerouting {len(synthesized_trips)} OD pairs to remaining network...")
    trips_rerouted = 0
    trips_failed_no_path = 0
    trips_failed_mapping = 0
    passengers_rerouted = 0
    
    # Apply normalization to flow
    normalized_total = 0
    
    # Track edge flow additions for debugging
    edge_flow_additions = {}
    
    # Sample a few trips for detailed output
    sample_indices = np.linspace(0, len(synthesized_trips)-1, min(5, len(synthesized_trips)), dtype=int)
    
    for trip_idx, trip in enumerate(synthesized_trips):
        source = trip['source_id']
        target = trip['target_id']
        raw_flow = trip['flow']
        
        # Normalize the flow to match graph units (passengers per train/vehicle)
        flow = raw_flow / normalization_factor
        
        show_debug = trip_idx in sample_indices
        
        if show_debug:
            print(f"\n  DEBUG Trip {trip_idx+1}: {source} -> {target}")
            print(f"    Raw flow: {raw_flow:.0f} passengers")
            print(f"    Normalized flow: {flow:.2f} (per vehicle)")
        
        # Find nearest active nodes
        new_source = find_nearest_active_node(source, G, pos, stops_df, closed_routes)
        new_target = find_nearest_active_node(target, G, pos, stops_df, closed_routes)
        
        if show_debug:
            print(f"    Mapped to: {new_source} -> {new_target}")
        
        if not new_source or not new_target:
            trips_failed_mapping += 1
            if show_debug:
                print(f"    FAILED: Could not map to active network")
            continue
            
        if new_source == new_target:
            if show_debug:
                print(f"    SKIPPED: Source and target map to same station")
            continue
            
        try:
            # Find shortest path on remaining network
            # Use travel time as weight
            def edge_weight(u, v, d):
                return d.get('travel_time_sec', 300.0)
            
            path = nx.shortest_path(G, new_source, new_target, weight=edge_weight)
            
            if show_debug:
                print(f"    Path found: {len(path)} stops, {len(path)-1} edges")
                print(f"    Path: {' -> '.join(path[:4])}{'...' if len(path) > 4 else ''}")
            
            # Add flow to edges along path
            edges_affected = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if G.has_edge(u, v):
                    old_flow = G[u][v].get('flow', 0)
                    if 'flow' not in G[u][v]:
                        G[u][v]['flow'] = 0
                    G[u][v]['flow'] += flow
                    new_flow = G[u][v]['flow']
                    
                    # Mark edge as receiving rerouted traffic for visualization
                    G[u][v]['rerouted_flow'] = G[u][v].get('rerouted_flow', 0) + flow
                    
                    # Track for summary
                    edge_key = (u, v)
                    if edge_key not in edge_flow_additions:
                        edge_flow_additions[edge_key] = {'old': old_flow, 'added': 0, 'new': 0}
                    edge_flow_additions[edge_key]['added'] += flow
                    edge_flow_additions[edge_key]['new'] = new_flow
                    
                    edges_affected += 1
            
            if show_debug:
                print(f"    Added {flow:.2f} flow to {edges_affected} edges")
            
            trips_rerouted += 1
            passengers_rerouted += raw_flow
            normalized_total += flow
            
        except nx.NetworkXNoPath:
            trips_failed_no_path += 1
            if show_debug:
                print(f"    FAILED: No path found in remaining network")
            continue
    
    print(f"\n  Step 5: Rerouting Summary")
    print(f"  -------------------------")
    print(f"  Successfully rerouted: {trips_rerouted} OD pairs")
    print(f"  Failed (no path): {trips_failed_no_path} OD pairs")
    print(f"  Failed (mapping): {trips_failed_mapping} OD pairs")
    print(f"  Total displaced passengers: {passengers_rerouted:.0f}")
    print(f"  Total added to network (normalized): {normalized_total:.1f} (per train/vehicle units)")
    
    # Show edges with highest flow additions
    if edge_flow_additions:
        sorted_edges = sorted(edge_flow_additions.items(), 
                             key=lambda x: x[1]['added'], reverse=True)
        print(f"\n  Top 10 edges receiving rerouted flow:")
        for (u, v), flows in sorted_edges[:10]:
            print(f"    - {u} -> {v}:")
            print(f"      Old flow: {flows['old']:.1f}, Added: {flows['added']:.1f}, New: {flows['new']:.1f}")
    
    print("  ========================================\n")
    
    return passengers_rerouted

