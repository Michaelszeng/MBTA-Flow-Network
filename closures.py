"""
Closure simulation logic for MBTA network.

This module handles:
- Line closures (removing entire transit lines)
- Station closures (bypassing specific stations and redistributing passengers)
- Demand rerouting using gravity-based models
"""

import networkx as nx
import numpy as np
import pandas as pd
from reroute import reroute_demand, synthesize_od_demand


def calculate_distance(lon1, lat1, lon2, lat2):
    """Calculate Euclidean distance between two (lon, lat) points."""
    return np.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)


def filter_closed_lines(edges_df, closure_value):
    """
    Filter out edges belonging to closed lines.
    
    Parameters:
    -----------
    edges_df : pd.DataFrame
        DataFrame of edges with 'route_id' column
    closure_value : str
        Line to close (e.g., "47", "Red", "Green")
    
    Returns:
    --------
    tuple : (filtered_edges_df, matched_routes, removed_count)
    """
    print(f"\n*** SIMULATING LINE CLOSURE: {closure_value} ***")
    all_route_ids = sorted(edges_df['route_id'].unique())
    print(f"  Available route IDs in graph: {all_route_ids}")
    
    original_edges = len(edges_df)
    
    # Check which edges match the closed line (exact match or prefix match)
    # Prefix match allows "Green" to match "Green-B", "Green-C", etc.
    def matches_closure(route_id):
        route_id_str = str(route_id)
        # Exact match
        if route_id_str == closure_value:
            return True
        # Prefix match with hyphen (e.g., "Green" matches "Green-B" but not "Green1")
        if route_id_str.startswith(closure_value + "-"):
            return True
        return False
    
    matching_edges = edges_df[edges_df['route_id'].apply(matches_closure)]
    matched_routes = sorted(matching_edges['route_id'].unique())
    
    print(f"  Found {len(matching_edges)} edges matching closed line '{closure_value}'")
    print(f"  Matched route IDs: {matched_routes}")
    
    # Show sample of what's being removed
    if len(matching_edges) > 0:
        sample = matching_edges[['source_id', 'target_id', 'route_id']].head(3)
        print(f"  Sample edges to remove:\n{sample}")
    
    # Filter out edges from the closed line
    filtered_df = edges_df[~edges_df['route_id'].apply(matches_closure)].copy()
    
    removed_edges = original_edges - len(filtered_df)
    print(f"  Removed {removed_edges} edges from closed lines {matched_routes}")
    print(f"  Remaining edges: {len(filtered_df)}")
    print(f"  Remaining route IDs: {sorted(filtered_df['route_id'].unique())}")
    
    if len(filtered_df) == 0:
        print("  ERROR: No edges remaining after line closure!")
        return None, matched_routes, removed_edges
    
    if removed_edges == 0:
        print(f"  WARNING: No edges were removed! Line '{closure_value}' not found in route IDs.")
        print(f"  Did you mean one of these? {all_route_ids}")
    
    return filtered_df, matched_routes, removed_edges


def close_station(G, pos, node_labels, station_name, edges_df):
    """
    Close a specific station by removing it from the graph and connecting adjacent stations.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The network graph (modified in-place)
    pos : dict
        Dictionary mapping node IDs to (lon, lat) tuples
    node_labels : dict
        Dictionary mapping node IDs to station names
    station_name : str
        Name of the station to close
    edges_df : pd.DataFrame
        Original edges dataframe (for reference)
    
    Returns:
    --------
    dict or None : Information about the closure including closed_nodes, bypassed_edges
    """
    print(f"\n*** SIMULATING STATION CLOSURE: {station_name} ***")
    
    # Find matching station node(s) by name
    matching_nodes = [node_id for node_id, name in node_labels.items() if name == station_name]
    
    if not matching_nodes:
        print(f"  ERROR: No station found matching '{station_name}'")
        print(f"  Available stations: {sorted(set(node_labels.values()))[:20]}...")
        return None
    
    if len(matching_nodes) > 1:
        print(f"  WARNING: Multiple nodes match '{station_name}': {matching_nodes}")
        print(f"  Closing all matching nodes")
    
    closure_info = {
        'closed_nodes': matching_nodes,
        'bypassed_edges': [],
        'station_name': station_name
    }
    
    for closed_node in matching_nodes:
        print(f"\n  Closing node: {closed_node} ({station_name})")
        print(f"    Location: {pos.get(closed_node, 'unknown')}")
        
        # Get incoming and outgoing edges
        predecessors = list(G.predecessors(closed_node))
        successors = list(G.successors(closed_node))
        
        print(f"    Predecessors: {[node_labels.get(p, p) for p in predecessors]}")
        print(f"    Successors: {[node_labels.get(s, s) for s in successors]}")
        
        # For each route passing through this station, connect predecessor to successor
        # Group by route to handle each route separately
        route_connections = {}
        
        # Collect incoming edges by route
        for pred in predecessors:
            edge_data = G[pred][closed_node]
            routes = edge_data.get('routes', set())
            for route in routes:
                if route not in route_connections:
                    route_connections[route] = {'preds': set(), 'succs': set()}
                route_connections[route]['preds'].add(pred)
        
        # Collect outgoing edges by route
        for succ in successors:
            edge_data = G[closed_node][succ]
            routes = edge_data.get('routes', set())
            for route in routes:
                if route not in route_connections:
                    route_connections[route] = {'preds': set(), 'succs': set()}
                route_connections[route]['succs'].add(succ)
        
        # Create bypass edges for each route
        bypass_count = 0
        for route, connections in route_connections.items():
            preds = connections['preds']
            succs = connections['succs']
            
            # Connect each predecessor to each successor for this route
            for pred in preds:
                for succ in succs:
                    if pred == succ:
                        continue  # Skip self-loops
                    
                    # Add or update edge from pred to succ
                    if G.has_edge(pred, succ):
                        # Edge already exists, add this route to it
                        if 'routes' not in G[pred][succ]:
                            G[pred][succ]['routes'] = set()
                        G[pred][succ]['routes'].add(route)
                        print(f"    Added route {route} to existing edge: {node_labels.get(pred, pred)} -> {node_labels.get(succ, succ)}")
                    else:
                        # Create new bypass edge
                        # Try to preserve flow and travel time attributes
                        edge_attrs = {
                            'weight': 1,
                            'routes': {route},
                            'bypass': True  # Mark as bypass edge
                        }
                        
                        # Sum travel times from the two original segments if available
                        pred_to_closed_time = G[pred][closed_node].get('travel_time_sec', None)
                        closed_to_succ_time = G[closed_node][succ].get('travel_time_sec', None)
                        if pred_to_closed_time and closed_to_succ_time:
                            edge_attrs['travel_time_sec'] = pred_to_closed_time + closed_to_succ_time
                        
                        G.add_edge(pred, succ, **edge_attrs)
                        print(f"    Created bypass edge [{route}]: {node_labels.get(pred, pred)} -> {node_labels.get(succ, succ)}")
                        bypass_count += 1
                        closure_info['bypassed_edges'].append((pred, succ, route))
        
        print(f"    Created {bypass_count} new bypass connections")
        
        # Remove the closed node from the graph
        G.remove_node(closed_node)
        print(f"    Removed node {closed_node} from graph")
    
    print(f"\n  Station closure complete. {len(matching_nodes)} node(s) removed, {len(closure_info['bypassed_edges'])} bypass edges created")
    return closure_info


def reroute_line_closure(G, pos, stops, closure_value, ridership_data, stops_coord_dict, normalization_factor):
    """
    Reroute demand from a closed line using gravity-based model.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The network graph (modified in-place to add rerouted flows)
    pos : dict
        Node positions
    stops : pd.DataFrame
        Stops dataframe
    closure_value : str
        The closed line identifier
    ridership_data : pd.DataFrame
        Combined ridership data with columns: route_id, parent_stop_id, average_ons, average_offs
    stops_coord_dict : dict
        Dictionary mapping stop_id to coordinate info
    normalization_factor : float
        Factor to normalize flows (trips per period)
    
    Returns:
    --------
    int : Number of passengers rerouted
    """
    print("\n  Calculating effects of line closure on network flow...")
    print("  Loading detailed ridership data for OD synthesis...")
    
    try:
        passengers_rerouted = reroute_demand(
            G, pos, stops, None, closure_value, 
            ridership_data=ridership_data, 
            stops_dict=stops_coord_dict,
            normalization_factor=normalization_factor
        )
        return passengers_rerouted
    except Exception as e:
        print(f"  Error during rerouting simulation: {e}")
        import traceback
        traceback.print_exc()
        return 0


def reroute_station_closure(G, pos, station_closure_info, all_ridership, stops_coord_dict, normalization_factor):
    """
    Reroute demand from a closed station using gravity-based model.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The network graph (modified in-place to add rerouted flows)
    pos : dict
        Node positions (note: closed nodes already removed from G but still in pos)
    station_closure_info : dict
        Information about the closure (closed_nodes, station_name, etc.)
    all_ridership : pd.DataFrame
        Combined ridership data with columns: route_id, parent_stop_id, average_ons, average_offs
    stops_coord_dict : dict
        Dictionary mapping stop_id to coordinate info
    normalization_factor : float
        Factor to normalize flows (trips per period)
    
    Returns:
    --------
    tuple : (trips_applied, total_flow_added)
    """
    print("\n  Calculating effects of station closure on network flow...")
    
    try:
        # Extract ridership for the closed station(s)
        closed_nodes = station_closure_info['closed_nodes']
        closed_station_data = all_ridership[all_ridership['parent_stop_id'].isin(closed_nodes)]
        
        if len(closed_station_data) == 0:
            print(f"  Warning: No ridership data found for closed station(s)")
            return 0, 0.0
        
        # Aggregate ons/offs for the closed station
        total_ons = closed_station_data['average_ons'].sum()
        total_offs = closed_station_data['average_offs'].sum()
        
        print(f"  Closed station ridership:")
        print(f"    Total boardings: {total_ons:.0f}")
        print(f"    Total alightings: {total_offs:.0f}")
        print(f"  Normalizing rerouted flow by factor of {normalization_factor} (estimated trips)")
        
        # Create a synthetic "closed station" entry for the gravity model
        # We'll place it at the centroid of all closed nodes
        if len(closed_nodes) == 1:
            closed_node_pos = pos.get(closed_nodes[0], (0, 0))
        else:
            # Average position if multiple nodes
            valid_positions = [pos.get(node) for node in closed_nodes if node in pos]
            if valid_positions:
                closed_node_pos = (
                    np.mean([p[0] for p in valid_positions]),
                    np.mean([p[1] for p in valid_positions])
                )
            else:
                closed_node_pos = (0, 0)
        
        print(f"  Redistributing {total_ons:.0f} boardings and {total_offs:.0f} alightings to network...")
        
        # Get all active stations in the network (excluding closed ones)
        active_stations = [node for node in G.nodes() if node not in closed_nodes]
        
        # Create stops_data for active stations with their ridership
        active_stops_data = []
        for node in active_stations:
            if node in stops_coord_dict:
                # Get ridership for this station
                station_ridership = all_ridership[all_ridership['parent_stop_id'] == node]
                station_ons = station_ridership['average_ons'].sum()
                station_offs = station_ridership['average_offs'].sum()
                
                if station_ons > 0 or station_offs > 0:
                    coords = stops_coord_dict[node]
                    active_stops_data.append({
                        'stop_id': node,
                        'ons': station_ons,
                        'offs': station_offs,
                        'lat': coords['stop_lat'],
                        'lon': coords['stop_lon']
                    })
        
        # Add the closed station as a source/destination
        closed_station_id = f'CLOSED_{closed_nodes[0]}'
        active_stops_data.append({
            'stop_id': closed_station_id,
            'ons': total_ons,
            'offs': total_offs,
            'lat': closed_node_pos[1],
            'lon': closed_node_pos[0]
        })
        
        # Generate OD trips
        print(f"  Synthesizing OD demand with {len(active_stops_data)} stations...")
        synthesized_trips = synthesize_od_demand(active_stops_data, decay_factor=2.0)
        
        # Filter trips that involve the closed station
        rerouted_trips = [t for t in synthesized_trips 
                         if t['source_id'] == closed_station_id or t['target_id'] == closed_station_id]
        
        print(f"  Found {len(rerouted_trips)} trips involving closed station")
        
        # Apply these trips to the network
        trips_applied = 0
        total_flow_added = 0
        
        for trip in rerouted_trips:
            source = trip['source_id']
            target = trip['target_id']
            raw_flow = trip['flow']
            flow = raw_flow / normalization_factor
            
            # Replace closed station ID with nearest active station
            if source == closed_station_id:
                # Find nearest station to closed station
                nearest = min(active_stations, 
                            key=lambda n: calculate_distance(closed_node_pos[0], closed_node_pos[1],
                                                             pos[n][0], pos[n][1]) if n in pos else float('inf'))
                source = nearest
            
            if target == closed_station_id:
                nearest = min(active_stations,
                            key=lambda n: calculate_distance(closed_node_pos[0], closed_node_pos[1],
                                                             pos[n][0], pos[n][1]) if n in pos else float('inf'))
                target = nearest
            
            if source == target or source not in G or target not in G:
                continue
            
            try:
                # Find shortest path
                def edge_weight(u, v, d):
                    return d.get('travel_time_sec', 300.0)
                
                path = nx.shortest_path(G, source, target, weight=edge_weight)
                
                # Add flow to edges
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    if G.has_edge(u, v):
                        if 'flow' not in G[u][v]:
                            G[u][v]['flow'] = 0
                        G[u][v]['flow'] += flow
                        G[u][v]['rerouted_flow'] = G[u][v].get('rerouted_flow', 0) + flow
                
                trips_applied += 1
                total_flow_added += flow
            
            except nx.NetworkXNoPath:
                continue
        
        print(f"  Successfully rerouted {trips_applied} trips")
        print(f"  Total flow added to network: {total_flow_added:.1f} (normalized units)")
        
        return trips_applied, total_flow_added
    
    except Exception as e:
        print(f"  Error during station closure rerouting: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0.0

