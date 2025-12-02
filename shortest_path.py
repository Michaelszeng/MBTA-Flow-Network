"""
Shortest path computation for MBTA network simulation.

This module provides functions to:
1. Add zero-cost transfer edges between nearby stops
2. Find stations near key/campus locations
3. Compute shortest paths considering travel times
"""

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


def calculate_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate Euclidean distance between two (lon, lat) points.
    
    For small distances, this is a reasonable approximation.
    
    Parameters:
    -----------
    lon1, lat1 : float
        Coordinates of first point
    lon2, lat2 : float
        Coordinates of second point
    
    Returns:
    --------
    float : Distance in degrees
    """
    return np.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)


def add_transfer_edges(G: nx.DiGraph, pos: Dict, transfer_radius: float, 
                      transfer_time: float = 0.0, node_routes: Dict = None, 
                      node_labels: Dict = None, debug_pairs: list = None) -> None:
    """
    Add zero-cost or low-cost transfer edges between nearby stops.
    
    This allows passengers to switch lines when stops are close together.
    Only adds transfer edges between stops that serve DIFFERENT routes.
    Modifies the graph in-place.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The network graph
    pos : dict
        Dictionary mapping node IDs to (lon, lat) tuples
    transfer_radius : float
        Maximum distance in degrees for transfer connections
    transfer_time : float
        Time cost for transfers in seconds (default: 0.0 for zero-cost)
    node_routes : dict, optional
        Dictionary mapping node IDs to sets of route IDs they serve
    node_labels : dict, optional
        Dictionary mapping node IDs to station names (for debugging)
    debug_pairs : list, optional
        List of (node_id1, node_id2) pairs to debug why transfer wasn't added
    """
    print(f"  Adding transfer edges for stops within {transfer_radius} degrees...")
    nodes = list(G.nodes())
    transfers_added = 0
    
    # Check each pair of nodes for proximity
    for i, node1 in enumerate(nodes):
        if node1 not in pos:
            continue
        
        lon1, lat1 = pos[node1]
        
        for node2 in nodes[i+1:]:
            if node2 not in pos:
                continue
            
            # Skip if nodes are the same
            if node1 == node2:
                continue
            
            lon2, lat2 = pos[node2]
            distance = calculate_distance(lon1, lat1, lon2, lat2)
            
            # Debug specific pairs if requested
            should_debug = False
            if debug_pairs and node_labels:
                for debug_node1, debug_node2 in debug_pairs:
                    label1 = node_labels.get(node1, node1)
                    label2 = node_labels.get(node2, node2)
                    if ((debug_node1 in label1 and debug_node2 in label2) or 
                        (debug_node2 in label1 and debug_node1 in label2)):
                        should_debug = True
                        break
            
            if should_debug:
                label1 = node_labels.get(node1, node1)
                label2 = node_labels.get(node2, node2)
                print(f"\n  DEBUG: Checking '{label1}' <-> '{label2}'")
                print(f"    Distance: {distance:.6f} degrees (threshold: {transfer_radius})")
            
            if distance <= transfer_radius:
                # Check if a transfer edge is needed
                if node_routes is not None:
                    routes1 = node_routes.get(node1, set())
                    routes2 = node_routes.get(node2, set())
                    
                    if should_debug:
                        print(f"    Routes1: {routes1}")
                        print(f"    Routes2: {routes2}")
                        print(f"    Shared routes: {routes1 & routes2}")
                        print(f"    Already has edge node1->node2: {G.has_edge(node1, node2)}")
                        print(f"    Already has edge node2->node1: {G.has_edge(node2, node1)}")
                    
                    # Skip if either node has no routes
                    if not routes1 or not routes2:
                        if should_debug:
                            print(f"    SKIPPED: One or both nodes have no routes")
                        continue
                    
                    # Skip if there's already a direct transit edge between these nodes
                    # (transfer edge would be redundant with existing transit connection)
                    if G.has_edge(node1, node2) or G.has_edge(node2, node1):
                        if should_debug:
                            print(f"    SKIPPED: Direct transit edge already exists")
                        continue
                
                if should_debug:
                    print(f"    ADDED: Transfer edge created")
                
                # Add bidirectional transfer edges if they don't already exist
                if not G.has_edge(node1, node2):
                    G.add_edge(node1, node2, 
                             travel_time_sec=transfer_time,
                             is_transfer=True,
                             routes={'transfer'})
                    transfers_added += 1
                
                if not G.has_edge(node2, node1):
                    G.add_edge(node2, node1,
                             travel_time_sec=transfer_time,
                             is_transfer=True,
                             routes={'transfer'})
                    transfers_added += 1
            elif should_debug:
                print(f"    SKIPPED: Distance too far")
    
    print(f"  Added {transfers_added} transfer edges")


def find_nearest_station_to_location(location_name: str, all_locations: Dict,
                                     G: nx.DiGraph, pos: Dict, 
                                     node_labels: Dict) -> Optional[str]:
    """
    Find the nearest station to a given location name.
    
    Parameters:
    -----------
    location_name : str
        Name of the location (e.g., "West Campus", "Central Square")
    all_locations : dict
        Dictionary of all key/campus locations with 'lon' and 'lat'
    G : networkx.DiGraph
        The network graph
    pos : dict
        Dictionary mapping node IDs to (lon, lat) tuples
    node_labels : dict
        Dictionary mapping node IDs to station names
    
    Returns:
    --------
    str or None : Node ID of nearest station, or None if not found
    """
    if location_name not in all_locations:
        # Check if it's already a station name
        for node_id, label in node_labels.items():
            if label == location_name:
                return node_id
        print(f"  Warning: Location '{location_name}' not found")
        return None
    
    loc_coords = all_locations[location_name]
    loc_lon = loc_coords['lon']
    loc_lat = loc_coords['lat']
    
    # Find nearest station
    min_distance = float('inf')
    nearest_station = None
    
    for node_id in G.nodes():
        if node_id not in pos:
            continue
        
        node_lon, node_lat = pos[node_id]
        distance = calculate_distance(node_lon, node_lat, loc_lon, loc_lat)
        
        if distance < min_distance:
            min_distance = distance
            nearest_station = node_id
    
    if nearest_station:
        station_name = node_labels.get(nearest_station, nearest_station)
        print(f"  '{location_name}' -> nearest station: '{station_name}' (distance: {min_distance:.4f} degrees)")
    
    return nearest_station


def compute_shortest_path(G: nx.DiGraph, source_node: str, target_node: str,
                         weight_attr: str = 'travel_time_sec',
                         default_weight: float = 300.0) -> Optional[List[str]]:
    """
    Compute shortest path between two nodes using Dijkstra's algorithm.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The network graph
    source_node : str
        Source node ID
    target_node : str
        Target node ID
    weight_attr : str
        Edge attribute to use as weight (default: 'travel_time_sec')
    default_weight : float
        Default weight for edges without the weight attribute (default: 300.0 seconds)
    
    Returns:
    --------
    list or None : List of node IDs in the shortest path, or None if no path exists
    """
    try:
        # Use networkx's shortest path with Dijkstra's algorithm
        # If an edge doesn't have the weight attribute, use a reasonable default (5 minutes)
        def edge_weight(u, v, d):
            return d.get(weight_attr, default_weight)
        
        path = nx.shortest_path(G, source=source_node, target=target_node,
                               weight=edge_weight)
        return path
    except nx.NetworkXNoPath:
        print(f"  No path exists between {source_node} and {target_node}")
        return None
    except nx.NodeNotFound as e:
        print(f"  Node not found: {e}")
        return None


def get_path_edges(path: List[str]) -> List[Tuple[str, str]]:
    """
    Convert a path (list of nodes) to a list of edges.
    
    Parameters:
    -----------
    path : list
        List of node IDs in the path
    
    Returns:
    --------
    list : List of (source, target) tuples representing edges
    """
    if not path or len(path) < 2:
        return []
    
    return [(path[i], path[i+1]) for i in range(len(path) - 1)]


def compute_path_statistics(G: nx.DiGraph, path: List[str],
                            node_labels: Dict, default_weight: float = 300.0) -> Dict:
    """
    Compute statistics about a path (total time, number of transfers, etc.).
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The network graph
    path : list
        List of node IDs in the path
    node_labels : dict
        Dictionary mapping node IDs to station names
    default_weight : float
        Default weight for edges without travel_time_sec (default: 300.0 seconds)
    
    Returns:
    --------
    dict : Dictionary containing path statistics
    """
    if not path or len(path) < 2:
        return {
            'total_time_sec': 0,
            'total_time_min': 0,
            'num_stops': 0,
            'num_transfers': 0,
            'station_names': []
        }
    
    total_time = 0
    num_transfers = 0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = G[u][v]
        
        # Add travel time (use default_weight for edges without travel_time_sec)
        travel_time = edge_data.get('travel_time_sec', default_weight)
        total_time += travel_time
        
        # Count transfers
        if edge_data.get('is_transfer', False):
            num_transfers += 1
    
    station_names = [node_labels.get(node, node) for node in path]
    
    return {
        'total_time_sec': total_time,
        'total_time_min': total_time / 60.0,
        'num_stops': len(path),
        'num_transfers': num_transfers,
        'station_names': station_names,
        'path': path
    }


def find_and_compute_shortest_path(source_location: str, target_location: str,
                                   G: nx.DiGraph, pos: Dict, node_labels: Dict,
                                   all_locations: Dict, transfer_radius: float,
                                   transfer_time: float, node_routes: Dict) -> Tuple[Optional[List[str]], Dict, List[Tuple[str, str]]]:
    """
    Find and compute the shortest path between two location names.
    
    This function creates a copy of the graph, adds transfer edges to the copy,
    computes the shortest path, and returns the path along with any transfer edges used.
    
    Parameters:
    -----------
    source_location : str
        Name of source location
    target_location : str
        Name of target location
    G : networkx.DiGraph
        The original network graph (without transfer edges)
    pos : dict
        Dictionary mapping node IDs to (lon, lat) tuples
    node_labels : dict
        Dictionary mapping node IDs to station names
    all_locations : dict
        Dictionary of all key/campus locations
    transfer_radius : float
        Maximum distance for transfer connections
    transfer_time : float
        Time cost for transfers in seconds
    node_routes : dict
        Dictionary mapping node IDs to sets of route IDs
    
    Returns:
    --------
    tuple : (path, stats, transfer_edges) where:
            - path is a list of node IDs or None
            - stats is a dictionary of path statistics
            - transfer_edges is a list of (u, v) tuples representing transfer edges used in path
    """
    print(f"\nComputing shortest path from '{source_location}' to '{target_location}'...")
    
    # Find nearest stations for source and target
    source_node = find_nearest_station_to_location(source_location, all_locations,
                                                   G, pos, node_labels)
    target_node = find_nearest_station_to_location(target_location, all_locations,
                                                   G, pos, node_labels)
    
    if source_node is None or target_node is None:
        return None, {}, []
    
    # Create a copy of the graph to add transfer edges
    print("  Creating graph copy for shortest path computation...")
    G_with_transfers = G.copy()
    
    # Add transfer edges to the copy
    # Debug specific pairs to understand why certain transfers aren't being added
    debug_pairs = [("Mountfort St", "Commonwealth Ave")]
    add_transfer_edges(G_with_transfers, pos, transfer_radius, transfer_time, node_routes, 
                      node_labels=node_labels, debug_pairs=debug_pairs)
    
    # Compute shortest path on the graph with transfers
    path = compute_shortest_path(G_with_transfers, source_node, target_node)
    
    if path is None:
        return None, {}, []
    
    # Identify which edges in the path are transfer edges
    transfer_edges_in_path = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = G_with_transfers[u][v]
        if edge_data.get('is_transfer', False):
            transfer_edges_in_path.append((u, v))
    
    # Compute statistics (use same default_weight as pathfinding for consistency)
    stats = compute_path_statistics(G_with_transfers, path, node_labels, default_weight=300.0)
    
    # Print path information
    print(f"\n  Shortest Path Found:")
    print(f"  Total travel time: {stats['total_time_min']:.2f} minutes ({stats['total_time_sec']:.1f} seconds)")
    print(f"  Number of stops: {stats['num_stops']}")
    print(f"  Number of transfers: {stats['num_transfers']}")
    print(f"\n  Route:")
    for i, station_name in enumerate(stats['station_names']):
        if i < len(path) - 1:
            edge_data = G_with_transfers[path[i]][path[i+1]]
            is_transfer = edge_data.get('is_transfer', False)
            travel_time = edge_data.get('travel_time_sec', 300.0)
            routes = edge_data.get('routes', set())
            
            if is_transfer:
                print(f"    {i+1}. {station_name}")
                print(f"        └─> Walk to {stats['station_names'][i+1]} ({travel_time:.0f}s)")
            else:
                route_str = f" ({list(routes)[0]})" if routes and 'transfer' not in routes else ""
                print(f"    {i+1}. {station_name}")
                print(f"        └─> {stats['station_names'][i+1]}{route_str} - {travel_time:.0f}s")
        else:
            print(f"    {i+1}. {station_name} [DESTINATION]")
    
    return path, stats, transfer_edges_in_path

