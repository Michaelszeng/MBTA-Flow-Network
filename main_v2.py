from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from closures import (close_station, filter_closed_lines, reroute_line_closure,
                      reroute_station_closure)
from flow import (add_bus_flow_to_edges, add_rail_flow_to_edges,
                  calculate_trips_from_headway)
from shortest_path import (add_transfer_edges, find_and_compute_shortest_path,
                           get_path_edges)
from travel_times_v3 import (add_travel_times_to_graph,
                             load_travel_times_from_gtfs)
from utils import ROUTE_COLORS, filter_routes_by_type, load_gtfs_data

# Hardcoded variables
MODE = "bus_and_rail"  # Options: "bus_only", "rail_only", "bus_and_rail"
DIRECTED_EDGES = True  # True = show directional arrows, False = undirected edges
SHOW_FLOW_LABELS = True  # True = show flow numbers on edges, False = hide flow labels
NORMALIZE_FLOW = True  # True = normalize to avg passengers per train, False = total flow in time period (note: doesn't affect bus data which is already normalized to # ppl per bus)
SHOW_TRAVEL_TIMES_LABELS = True  # True = add travel time weights to edges, False = skip travel time data

# Ridership data filters
DAY_TYPE = "Weekday"  # Options: "Weekday", "Saturday", "Sunday" (case-sensitive for rail, lowercase for bus)
TIME_PERIOD = "EVENING"  # Options: "AM_PEAK", "PM_PEAK", "EVENING", "NIGHT", etc.

ROUTE_TYPES = {
    'rail_only': [0, 1],  # 0 = Tram/Light Rail, 1 = Subway/Metro
    'bus_only': [3],       # 3 = Bus
    'bus_and_rail': [0, 1, 3]
}

BUS_LINES = ["1", "70", "747", "64", "68", "47", "741", "83"]  # Only lines that >= 3 people listed (747=CT2, 741=SL1)

SHORTEST_PATH_SRC_DST = None
# SHORTEST_PATH_SRC_DST = ("East Campus", "Fenway/Kenmore")

SIMULATE_LINE_CLOSURE = ""  # List of lines to close (e.g. "70", "747", "Red", "Green-D", or empty string to indicate no line closure)

SIMULATE_STOP_STATION_CLOSURE = ""  # Station or stop name to close (e.g. "Kendall/MIT", "Park Street", "Amesbury St @ Vassar St", or empty string for no closure)

# Radius for identifying stations near key/campus locations (in degrees, approximately 200 meters)
KEY_LOCATION_RADIUS = 0.00275

# Key locations to mark on the map (longitude, latitude)
KEY_LOCATIONS = {
    'Central Square': {'lon': -71.1038, 'lat': 42.3655},
    'Harvard Square': {'lon': -71.1190, 'lat': 42.3736},
    'Back Bay': {'lon': -71.078147, 'lat': 42.348974},
    'Fenway/Kenmore': {'lon': -71.095569, 'lat': 42.348049},
    'Downtown': {'lon': -71.061225, 'lat': 42.356518}
}

# Campus locations to mark on the map (longitude, latitude)
CAMPUS_LOCATIONS = {
    'West Campus': {'lon': -71.10755, 'lat': 42.35515},  # Approximately West Briggs field
    'East Campus': {'lon': -71.085528, 'lat': 42.360472}
}


def consolidate_to_parent_stations(stop_id, stops_dict):
    """Map a stop_id to its parent station if it exists, otherwise return itself."""
    if stop_id not in stops_dict:
        return stop_id
    
    stop_info = stops_dict[stop_id]
    parent = stop_info.get('parent_station', '')
    
    # If has a parent station, use that; otherwise use the stop itself
    # Need to check for NaN/None explicitly since parent_station may be NaN from CSV
    if pd.notna(parent) and parent != '':
        return parent
    else:
        return stop_id


def calculate_distance(lon1, lat1, lon2, lat2):
    """Calculate Euclidean distance between two (lon, lat) points.
    
    For small distances, this is a reasonable approximation.
    """
    return np.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)


def identify_key_stations(G, pos, key_locations, campus_locations, radius):
    """Identify stations within radius of key/campus locations.
    
    Parameters:
    -----------
    G : networkx.Graph
        The network graph
    pos : dict
        Dictionary mapping node IDs to (lon, lat) tuples
    key_locations : dict
        Dictionary of key location names to {'lon': x, 'lat': y} dicts
    campus_locations : dict
        Dictionary of campus location names to {'lon': x, 'lat': y} dicts
    radius : float
        Distance threshold in degrees
    
    Returns:
    --------
    dict : Mapping of node_id to location_name for nodes near key/campus locations
    """
    key_stations = {}
    
    # Combine all locations
    all_locations = {}
    all_locations.update(key_locations)
    all_locations.update(campus_locations)
    
    # Check each node against each location
    for node_id in G.nodes():
        if node_id not in pos:
            continue
        
        node_lon, node_lat = pos[node_id]
        
        for location_name, coords in all_locations.items():
            loc_lon = coords['lon']
            loc_lat = coords['lat']
            
            distance = calculate_distance(node_lon, node_lat, loc_lon, loc_lat)
            
            if distance <= radius:
                # If node is near multiple locations, keep the closest one
                if node_id in key_stations:
                    # Check if this location is closer
                    existing_loc = key_stations[node_id]['location']
                    existing_coords = all_locations[existing_loc]
                    existing_dist = calculate_distance(
                        node_lon, node_lat, 
                        existing_coords['lon'], existing_coords['lat']
                    )
                    if distance < existing_dist:
                        key_stations[node_id] = {
                            'location': location_name,
                            'distance': distance
                        }
                else:
                    key_stations[node_id] = {
                        'location': location_name,
                        'distance': distance
                    }
    
    return key_stations


def create_edges_from_gtfs(stops, stop_times, trips, routes, route_types, 
                           ridership_filepath=None, stop_orders_filepath=None,
                           day_type='Weekday', time_period='PM_PEAK',
                           ridership_type='rail', specific_routes=None, normalize_flow=True,
                           trips_count=None):
    """
    Create network edges from GTFS data with optional ridership flow data.
    
    Parameters:
    -----------
    ...
    trips_count : int, optional
        Pre-calculated trip count for normalization.
    """
    # Filter routes by type (i.e. bus_only, rail_only, bus_and_rail)
    filtered_routes = filter_routes_by_type(routes, route_types)
    
    # Further filter by specific routes if provided
    if specific_routes is not None:
        # Convert specific_routes to strings to match route_id data type
        specific_routes_str = [str(r) for r in specific_routes]
        filtered_routes = filtered_routes[filtered_routes['route_id'].isin(specific_routes_str)]
        print(f"  Filtered to {len(filtered_routes)} specific routes")
    
    route_ids = set(filtered_routes['route_id'])
    
    # Filter trips for selected routes
    filtered_trips = trips[trips['route_id'].isin(route_ids)]
    trip_ids = set(filtered_trips['trip_id'])
    
    # Filter stop_times for selected trips
    filtered_stop_times = stop_times[stop_times['trip_id'].isin(trip_ids)].copy()
    
    # Create stops lookup dictionary
    stops_dict = stops.set_index('stop_id').to_dict('index')
    
    # Consolidate stops to parent stations
    print(f"  Consolidating stops to parent stations...")
    filtered_stop_times['parent_stop_id'] = filtered_stop_times['stop_id'].apply(
        lambda x: consolidate_to_parent_stations(x, stops_dict)
    )
    
    print(f"  Processing {len(filtered_trips)} trips...")
    
    edges = []
    
    # Group by trip_id to get stop sequences
    grouped = filtered_stop_times.groupby('trip_id')
    
    for trip_id, group in grouped:
        # Sort by stop_sequence
        group_sorted = group.sort_values('stop_sequence')
        
        # Get route info for this trip
        trip_info = filtered_trips[filtered_trips['trip_id'] == trip_id].iloc[0]
        route_id = trip_info['route_id']
        
        # Create edges between consecutive stops (using parent stations)
        for i in range(len(group_sorted) - 1):
            current_stop_id = group_sorted.iloc[i]['parent_stop_id']
            next_stop_id = group_sorted.iloc[i + 1]['parent_stop_id']
            
            # Skip self-loops
            if current_stop_id == next_stop_id:
                continue
            
            edges.append({
                'source_id': current_stop_id,
                'target_id': next_stop_id,
                'route_id': route_id,
                'trip_id': trip_id
            })
    
    # Remove duplicate edges (same source->target->route combinations)
    # Note: For directed graphs, A->B and B->A are different edges and both kept
    # Keep route_id in duplicate check so different routes can share the same physical edge
    edges_df = pd.DataFrame(edges)
    if len(edges_df) > 0:
        edges_df = edges_df.drop_duplicates(subset=['source_id', 'target_id', 'route_id'])
    
    # Add ridership flow data if paths provided
    if ridership_filepath:
        print(f"  Adding {ridership_type} ridership flow data ({day_type}, {time_period})...")
        
        if ridership_type == 'rail' and stop_orders_filepath:
            edges_df = add_rail_flow_to_edges(
                edges_df,
                ridership_filepath,
                stop_orders_filepath,
                day_type=day_type,
                time_period=time_period,
                normalize=normalize_flow,
                trips_count=trips_count
            )
        elif ridership_type == 'bus':
            edges_df = add_bus_flow_to_edges(
                edges_df,
                ridership_filepath,
                stops,
                day_type=day_type,
                time_period=time_period,
                normalize=False  # Bus data is already per-bus load
            )
        print(f"  Added flow data: {(edges_df['flow'] > 0).sum()} edges with ridership")
    
    return edges_df


def create_graph_with_positions(edges_df, stops):
    """Create a NetworkX directed graph with geographic positions."""
    G = nx.DiGraph()
    
    # Filter stops to parent stations (location_type=1) or stops without parents
    # This gives us one entry per physical station location
    parent_stops = stops[
        (stops['location_type'] == 1) | 
        (stops['parent_station'].isna()) | 
        (stops['parent_station'] == '')
    ].copy()
    
    # Create a lookup for stop info
    stops_dict = parent_stops.set_index('stop_id').to_dict('index')
    
    # Create position dictionary, node labels, and track routes
    pos = {}
    node_labels = {}
    node_routes = {}  # Track which routes serve each node
    edge_routes = {}  # Track which route each edge belongs to
    
    # First, filter edges to only include those where both stops exist
    valid_edges = []
    skipped_count = 0
    
    for _, edge in edges_df.iterrows():
        source_id = edge['source_id']
        target_id = edge['target_id']
        route_id = edge['route_id']
        
        # Only add edge if both stops exist in stops.txt
        if source_id in stops_dict and target_id in stops_dict:
            valid_edges.append(edge)
            
            # Add positions and labels for these stops
            if source_id not in pos:
                stop_info = stops_dict[source_id]
                pos[source_id] = (stop_info['stop_lon'], stop_info['stop_lat'])
                node_labels[source_id] = stop_info['stop_name']
                node_routes[source_id] = set()
            
            if target_id not in pos:
                stop_info = stops_dict[target_id]
                pos[target_id] = (stop_info['stop_lon'], stop_info['stop_lat'])
                node_labels[target_id] = stop_info['stop_name']
                node_routes[target_id] = set()
            
            # Track routes for nodes
            node_routes[source_id].add(route_id)
            node_routes[target_id].add(route_id)
        else:
            skipped_count += 1
    
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} edges due to missing stop data")
    
    # Add directed edges to graph
    for edge in valid_edges:
        source_id = edge['source_id']
        target_id = edge['target_id']
        route_id = edge['route_id']
        flow = edge.get('flow', None)
        
        if G.has_edge(source_id, target_id):
            G[source_id][target_id]['weight'] += 1
            # Track multiple routes on same edge
            if 'routes' not in G[source_id][target_id]:
                G[source_id][target_id]['routes'] = set()
            G[source_id][target_id]['routes'].add(route_id)
            # For flow: Keep the first non-zero flow value, don't sum
            # (Multiple branches like Green-B, Green-C on same segment should show
            # flow per train, not sum of all branches)
            if flow is not None and flow > 0 and 'flow' not in G[source_id][target_id]:
                G[source_id][target_id]['flow'] = flow
        else:
            edge_attrs = {'weight': 1, 'routes': {route_id}}
            if flow is not None:
                edge_attrs['flow'] = flow
            G.add_edge(source_id, target_id, **edge_attrs)
    
    return G, pos, node_labels, node_routes


def get_bezier_curve(x0, y0, x1, y1, fixed_bulge=0.002):
    """
    Generate a Bezier curve between two points with a fixed perpendicular offset.
    The fixed_bulge parameter is a constant distance in coordinate units.
    Positive bulge curves to the left, negative to the right.
    """
    # Calculate midpoint
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    
    # Calculate perpendicular vector
    dx, dy = x1 - x0, y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return [x0, x1], [y0, y1]
    
    # Perpendicular unit vector
    px, py = -dy / length, dx / length
    
    # Control point (fixed offset from midpoint, not proportional to edge length)
    cx, cy = mx + px * fixed_bulge, my + py * fixed_bulge
    
    # Generate curve points using quadratic Bezier
    t = np.linspace(0, 1, 50)
    curve_x = (1-t)**2 * x0 + 2*(1-t)*t * cx + t**2 * x1
    curve_y = (1-t)**2 * y0 + 2*(1-t)*t * cy + t**2 * y1
    
    return curve_x, curve_y


def create_arrow_triangle(curve_x, curve_y, color, arrow_distance=0.0019, arrow_length=0.002, arrow_width=0.0006):
    """
    Create a triangle arrow at the end of a curve to indicate direction.
    
    Parameters:
    - curve_x, curve_y: arrays of points defining the curve
    - color: color for the triangle
    - arrow_distance: distance from end of curve to place arrow (in coordinate units)
    - arrow_length: length of arrow from base to tip (in coordinate units)
    - arrow_width: half-width of the arrow base (in coordinate units)
    
    Returns:
    - go.Scatter trace representing the triangle
    """
    # Get the direction at the end of the curve
    # Use last two points to determine angle
    x_end, y_end = curve_x[-1], curve_y[-1]
    x_prev, y_prev = curve_x[-2], curve_y[-2]
    
    # Direction vector
    dx = x_end - x_prev
    dy = y_end - y_prev
    length = np.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return None
    
    # Normalize direction
    dx_norm = dx / length
    dy_norm = dy / length
    
    # Position arrow at a fixed distance from the end
    arrow_x = x_end - dx_norm * arrow_distance
    arrow_y = y_end - dy_norm * arrow_distance
    
    # Create triangle pointing in the direction of the curve
    # Triangle dimensions (in coordinate units)
    
    # Tip of arrow (in direction of travel)
    tip_x = arrow_x + dx_norm * arrow_length
    tip_y = arrow_y + dy_norm * arrow_length
    
    # Perpendicular vector for width
    perp_x = -dy_norm
    perp_y = dx_norm
    
    # Base points of triangle
    base1_x = arrow_x + perp_x * arrow_width
    base1_y = arrow_y + perp_y * arrow_width
    base2_x = arrow_x - perp_x * arrow_width
    base2_y = arrow_y - perp_y * arrow_width
    
    # Create triangle (close the shape by returning to first point)
    triangle_x = [tip_x, base1_x, base2_x, tip_x]
    triangle_y = [tip_y, base1_y, base2_y, tip_y]
    
    # Create filled triangle trace
    arrow_trace = go.Scatter(
        x=triangle_x,
        y=triangle_y,
        mode='lines',
        fill='toself',
        fillcolor=color,
        line=dict(width=0, color=color),
        hoverinfo='skip',
        showlegend=False
    )
    
    return arrow_trace


def visualize_graph(G, pos, node_labels, node_routes, mode, directed=True, show_flow_labels=True, show_travel_time_labels=True, shortest_path=None, transfer_edges=None):
    """Create an interactive network graph using Plotly with color-coded routes and curved bi-directional edges.
    
    Parameters:
    -----------
    directed : bool
        If True, show directional arrows on edges. If False, show undirected edges.
    show_flow_labels : bool
        If True, display flow numbers on edges. If False, hide flow labels.
    show_travel_time_labels : bool
        If True, display travel time labels on edges. If False, hide travel time labels.
    shortest_path : list or None
        If provided, list of node IDs representing shortest path to highlight in gold.
    transfer_edges : list or None
        If provided, list of (u, v) tuples representing transfer edges used in shortest path.
    """
    # Identify bi-directional edges
    bidirectional_edges = set()
    for u, v in G.edges():
        if G.has_edge(v, u):
            # Only add the edge pair once (use sorted tuple to avoid duplicates)
            edge_pair = tuple(sorted([u, v]))
            bidirectional_edges.add(edge_pair)
    
    print(f"  Found {len(bidirectional_edges)} bi-directional edge pairs")
    
    # Prepare shortest path data structures for highlighting
    shortest_path_edges = set()
    shortest_path_nodes = set()
    transfer_edges_set = set()
    if shortest_path:
        shortest_path_nodes = set(shortest_path)
        shortest_path_edges = set(get_path_edges(shortest_path))
        print(f"  Highlighting shortest path with {len(shortest_path_nodes)} nodes and {len(shortest_path_edges)} edges")
    if transfer_edges is not None and transfer_edges:
        transfer_edges_set = set(transfer_edges)
        print(f"  Highlighting {len(transfer_edges_set)} transfer edges in shortest path")
    
    # Group edges by route for colored visualization
    route_edges = {}
    for edge in G.edges(data=True):
        routes = edge[2].get('routes', set())
        # Use the first route for edge color (if multiple routes share an edge)
        primary_route = list(routes)[0] if routes else 'default'
        
        if primary_route not in route_edges:
            route_edges[primary_route] = []
        route_edges[primary_route].append((edge[0], edge[1], edge[2]))
    
    # Create edge traces grouped by route
    edge_traces = []
    flow_label_traces = []
    travel_time_label_traces = []
    drawn_undirected_edges = set()  # Track drawn edges when undirected to avoid duplicates
    
    for route_id, edges in route_edges.items():
        color = ROUTE_COLORS.get(route_id, ROUTE_COLORS['default'])
        for edge in edges:
            u, v, data = edge
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            weight = data.get('weight', 1)
            flow = data.get('flow', None)
            
            # Override color to gold if edge is in shortest path (but not a transfer edge)
            is_in_shortest_path = (u, v) in shortest_path_edges and (u, v) not in transfer_edges_set
            edge_color = 'gold' if is_in_shortest_path else color
            edge_width = 4.0 if is_in_shortest_path else 2.5
            
            # Check if this is part of a bi-directional edge pair
            edge_pair = tuple(sorted([u, v]))
            is_bidirectional = edge_pair in bidirectional_edges
            
            # Skip duplicate edges in undirected mode
            if not directed and is_bidirectional:
                if edge_pair in drawn_undirected_edges:
                    continue  # Already drew this edge in the opposite direction
                drawn_undirected_edges.add(edge_pair)
            
            # Use curved lines only when directed=True and edge is bidirectional
            if directed and is_bidirectional:
                # Use curved line for bi-directional edges
                # To ensure edges curve in opposite directions, we calculate the perpendicular
                # based on a consistent reference (sorted node order), not the actual edge direction
                sorted_pair = tuple(sorted([u, v]))
                
                # Get coordinates of sorted nodes for consistent perpendicular calculation
                # Fixed bulge distance in degrees (approximately 300-400 meters at Boston's latitude)
                fixed_bulge = 0.003
                
                if u == sorted_pair[0]:
                    # This edge goes from first to second in sorted order
                    ref_x0, ref_y0 = pos[sorted_pair[0]]
                    ref_x1, ref_y1 = pos[sorted_pair[1]]
                    offset_sign = 1  # Curve to the left of reference direction
                else:
                    # This edge goes from second to first (opposite direction)
                    ref_x0, ref_y0 = pos[sorted_pair[0]]
                    ref_x1, ref_y1 = pos[sorted_pair[1]]
                    offset_sign = -1  # Curve to the right of reference direction
                
                # Calculate curve using reference direction but plot using actual coordinates
                # Get the perpendicular based on reference, apply to actual edge
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                dx_ref, dy_ref = ref_x1 - ref_x0, ref_y1 - ref_y0
                length = np.sqrt(dx_ref**2 + dy_ref**2)
                
                if length > 0:
                    # Perpendicular unit vector based on reference direction
                    px, py = -dy_ref / length, dx_ref / length
                    # Control point offset from midpoint - FIXED distance, not proportional to length
                    cx = mx + px * offset_sign * fixed_bulge
                    cy = my + py * offset_sign * fixed_bulge
                    
                    # Generate curve from actual start to actual end
                    t = np.linspace(0, 1, 50)
                    curve_x = (1-t)**2 * x0 + 2*(1-t)*t * cx + t**2 * x1
                    curve_y = (1-t)**2 * y0 + 2*(1-t)*t * cy + t**2 * y1
                    
                    # Calculate midpoint of curve for label placement
                    mid_idx = len(curve_x) // 2
                    label_x, label_y = curve_x[mid_idx], curve_y[mid_idx]
                else:
                    curve_x, curve_y = [x0, x1], [y0, y1]
                    label_x, label_y = mx, my
                
                direction_symbol = '→' if directed else '—'
                hover_text = f'{node_labels.get(u, u)} {direction_symbol} {node_labels.get(v, v)}<br>Route: {route_id}<br>Weight: {weight}'
                if flow is not None and flow > 0:
                    hover_text += f'<br>Flow: {flow:.0f} riders'
                travel_time_sec = data.get('travel_time_sec', None)
                if travel_time_sec is not None:
                    hover_text += f'<br>Travel Time: {travel_time_sec:.0f} sec'
                
                edge_trace = go.Scatter(
                    x=curve_x,
                    y=curve_y,
                    mode='lines',
                    line=dict(width=edge_width, color=edge_color),
                    hoverinfo='text',
                    text=hover_text,
                    name=route_id,
                    showlegend=False,
                    legendgroup=route_id
                )
                edge_traces.append(edge_trace)
                
                # Add directional arrow triangle for curved edge
                if directed:
                    # Use half-size arrows for bus routes
                    is_bus_route = (color == ROUTE_COLORS['default'])
                    if is_bus_route:
                        arrow_trace = create_arrow_triangle(curve_x, curve_y, edge_color, 
                                                          arrow_length=0.001, arrow_width=0.0003)
                    else:
                        arrow_trace = create_arrow_triangle(curve_x, curve_y, edge_color)
                    if arrow_trace:
                        edge_traces.append(arrow_trace)
                
                # Add flow label if flow data exists
                if show_flow_labels and flow is not None and flow > 0:
                    flow_label = go.Scatter(
                        x=[label_x],
                        y=[label_y],
                        mode='text',
                        text=[f'{flow:.0f}'],
                        textfont=dict(size=9, color='black', weight='bold'),
                        textposition='middle center',
                        hoverinfo='skip',
                        showlegend=False
                    )
                    flow_label_traces.append(flow_label)
                
                # Add travel time label if travel time data exists
                travel_time_sec = data.get('travel_time_sec', None)
                if show_travel_time_labels and travel_time_sec is not None:
                    # Offset label position slightly below midpoint for travel times
                    offset_y = -0.0008  # Small offset in latitude
                    travel_time_label = go.Scatter(
                        x=[label_x],
                        y=[label_y + offset_y],
                        mode='text',
                        text=[f'{travel_time_sec:.0f}s'],
                        textfont=dict(size=8, color='darkblue'),
                        textposition='middle center',
                        hoverinfo='skip',
                        showlegend=False
                    )
                    travel_time_label_traces.append(travel_time_label)
                
            else:
                # Use straight line for uni-directional edges or when undirected
                curve_x, curve_y = np.array([x0, x1]), np.array([y0, y1])
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                
                direction_symbol = '→' if directed else '—'
                hover_text = f'{node_labels.get(u, u)} {direction_symbol} {node_labels.get(v, v)}<br>Route: {route_id}<br>Weight: {weight}'
                if flow is not None and flow > 0:
                    hover_text += f'<br>Flow: {flow:.0f} riders'
                
                edge_trace = go.Scatter(
                    x=curve_x,
                    y=curve_y,
                    mode='lines',
                    line=dict(width=edge_width, color=edge_color),
                    hoverinfo='text',
                    text=hover_text,
                    name=route_id,
                    showlegend=False,
                    legendgroup=route_id
                )
                edge_traces.append(edge_trace)
                
                # Add directional arrow triangle for straight edge
                if directed:
                    # Use half-size arrows for bus routes
                    is_bus_route = (color == ROUTE_COLORS['default'])
                    if is_bus_route:
                        arrow_trace = create_arrow_triangle(curve_x, curve_y, edge_color,
                                                          arrow_length=0.001, arrow_width=0.0003)
                    else:
                        arrow_trace = create_arrow_triangle(curve_x, curve_y, edge_color)
                    if arrow_trace:
                        edge_traces.append(arrow_trace)
                
                # Add flow label if flow data exists
                if show_flow_labels and flow is not None and flow > 0:
                    flow_label = go.Scatter(
                        x=[mx],
                        y=[my],
                        mode='text',
                        text=[f'{flow:.0f}'],
                        textfont=dict(size=9, color='black', weight='bold'),
                        textposition='middle center',
                        hoverinfo='skip',
                        showlegend=False
                    )
                    flow_label_traces.append(flow_label)
                
                # Add travel time label if travel time data exists
                travel_time_sec = data.get('travel_time_sec', None)
                if show_travel_time_labels and travel_time_sec is not None:
                    # Offset label position slightly below midpoint for travel times
                    offset_y = -0.0008  # Small offset in latitude
                    travel_time_label = go.Scatter(
                        x=[mx],
                        y=[my + offset_y],
                        mode='text',
                        text=[f'{travel_time_sec:.0f}s'],
                        textfont=dict(size=8, color='darkblue'),
                        textposition='middle center',
                        hoverinfo='skip',
                        showlegend=False
                    )
                    travel_time_label_traces.append(travel_time_label)
    
    # Create a legend trace for each route
    # Consolidate all bus routes (default gray color) into a single "Bus Route" entry
    legend_traces = []
    has_bus_routes = False
    
    for route_id in route_edges.keys():
        color = ROUTE_COLORS.get(route_id, ROUTE_COLORS['default'])
        
        # Check if this is a bus route (uses default gray color)
        if color == ROUTE_COLORS['default']:
            has_bus_routes = True
            # Skip individual legend entries for bus routes
            continue
        
        # Create legend entry for named transit lines (Red, Orange, Blue, Green, etc.)
        legend_trace = go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(width=4, color=color),
            name=route_id,
            showlegend=True,
            legendgroup=route_id
        )
        legend_traces.append(legend_trace)
    
    # Add a single consolidated legend entry for all bus routes
    if has_bus_routes:
        bus_legend_trace = go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(width=4, color=ROUTE_COLORS['default']),
            name='Bus Route',
            showlegend=True,
            legendgroup='bus_routes'
        )
        legend_traces.append(bus_legend_trace)
    
    # Visualize transfer edges used in shortest path as dotted golden lines
    transfer_edge_traces = []
    if transfer_edges_set:
        for u, v in transfer_edges_set:
            if u not in pos or v not in pos:
                continue
            
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            # Draw as straight dotted line
            transfer_trace = go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(width=3, color='gold', dash='dot'),
                hoverinfo='text',
                text=f'{node_labels.get(u, u)} → {node_labels.get(v, v)}<br>Transfer Connection',
                showlegend=False
            )
            transfer_edge_traces.append(transfer_trace)
        
        # Add legend entry for transfer edges
        if transfer_edge_traces:
            transfer_legend = go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(width=3, color='gold', dash='dot'),
                name='Transfer (Walking)',
                showlegend=True
            )
            legend_traces.append(transfer_legend)
    
    # Create node traces grouped by route (for nodes served by that route)
    node_traces = []
    bus_node_traces_with_labels = []
    bus_node_traces_without_labels = []
    
    for route_id in route_edges.keys():
        node_x = []
        node_y = []
        node_text = []
        
        for node in G.nodes():
            routes = node_routes.get(node, set())
            if route_id in routes:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node_labels.get(node, node))
        
        if node_x:  # Only add trace if there are nodes for this route
            color = ROUTE_COLORS.get(route_id, ROUTE_COLORS['default'])
            is_bus_route = (color == ROUTE_COLORS['default'])
            
            if is_bus_route:
                # Create two versions for bus routes: with and without labels
                # Version with labels (initially visible)
                node_trace_with_labels = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition='top center',
                    textfont=dict(size=8, color='black'),
                    marker=dict(
                        size=10,
                        color=color,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate='<b>%{text}</b><br>Route: ' + route_id + '<br>Lon: %{x:.4f}<br>Lat: %{y:.4f}<extra></extra>',
                    showlegend=False,
                    visible=True,
                    name=f'bus_nodes_with_labels_{route_id}'
                )
                bus_node_traces_with_labels.append(node_trace_with_labels)
                
                # Version without labels (initially hidden)
                node_trace_without_labels = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers',
                    text=node_text,
                    marker=dict(
                        size=10,
                        color=color,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate='<b>%{text}</b><br>Route: ' + route_id + '<br>Lon: %{x:.4f}<br>Lat: %{y:.4f}<extra></extra>',
                    showlegend=False,
                    visible=False,
                    name=f'bus_nodes_without_labels_{route_id}'
                )
                bus_node_traces_without_labels.append(node_trace_without_labels)
            else:
                # Rail routes keep standard behavior (always show labels)
                node_trace = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers+text',
                    text=node_text,
                    textposition='top center',
                    textfont=dict(size=8, color='black'),
                    marker=dict(
                        size=10,
                        color=color,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate='<b>%{text}</b><br>Route: ' + route_id + '<br>Lon: %{x:.4f}<br>Lat: %{y:.4f}<extra></extra>',
                    showlegend=False
                )
                node_traces.append(node_trace)
    
    # Create markers for shortest path nodes (draw before key stations so they're visible)
    shortest_path_node_traces = []
    if shortest_path_nodes:
        sp_node_x = []
        sp_node_y = []
        sp_node_text = []
        
        for node in shortest_path_nodes:
            if node in pos:
                x, y = pos[node]
                sp_node_x.append(x)
                sp_node_y.append(y)
                station_name = node_labels.get(node, node)
                sp_node_text.append(station_name)
        
        if sp_node_x:
            sp_node_trace = go.Scatter(
                x=sp_node_x,
                y=sp_node_y,
                mode='markers+text',
                text=sp_node_text,
                textposition='top center',
                textfont=dict(size=10, color='darkgoldenrod', weight='bold'),
                marker=dict(
                    size=15,
                    color='gold',
                    line=dict(width=3, color='darkgoldenrod')
                ),
                hovertemplate='<b>%{text}</b><br>[Shortest Path]<extra></extra>',
                showlegend=True,
                name='Shortest Path'
            )
            shortest_path_node_traces.append(sp_node_trace)
    
    # Create star markers for stations near key/campus locations
    key_station_traces = []
    key_station_x = []
    key_station_y = []
    key_station_text = []
    
    for node in G.nodes():
        if G.nodes[node].get('is_key_station', False):
            x, y = pos[node]
            key_station_x.append(x)
            key_station_y.append(y)
            location_name = G.nodes[node].get('key_location', 'Unknown')
            station_name = node_labels.get(node, node)
            key_station_text.append(f'{station_name}<br>Near: {location_name}')
    
    if key_station_x:
        key_station_trace = go.Scatter(
            x=key_station_x,
            y=key_station_y,
            mode='markers',
            marker=dict(
                symbol='star',
                size=6,
                color='gold',
                line=dict(width=1, color='orange')
            ),
            hovertemplate='<b>%{text}</b><extra></extra>',
            text=key_station_text,
            showlegend=True,
            name='Stations Near Key/Campus Locations'
        )
        key_station_traces.append(key_station_trace)
    
    # Create markers for key locations
    key_location_traces = []
    for location_name, coords in KEY_LOCATIONS.items():
        # Main marker (triangle pointing down like a map pin)
        marker_trace = go.Scatter(
            x=[coords['lon']],
            y=[coords['lat']],
            mode='markers+text',
            marker=dict(
                symbol='triangle-down',
                size=20,
                color='red',
                line=dict(width=2, color='darkred')
            ),
            text=[location_name],
            textposition='top center',
            textfont=dict(size=20, color='darkred', family='Arial Black'),
            hovertemplate=f'<b>{location_name}</b><br>Lon: {coords["lon"]:.4f}<br>Lat: {coords["lat"]:.4f}<extra></extra>',
            showlegend=False,
            name=location_name
        )
        key_location_traces.append(marker_trace)
    
    # Create markers for campus locations (blue)
    campus_location_traces = []
    for location_name, coords in CAMPUS_LOCATIONS.items():
        # Main marker (triangle pointing down like a map pin)
        marker_trace = go.Scatter(
            x=[coords['lon']],
            y=[coords['lat']],
            mode='markers+text',
            marker=dict(
                symbol='triangle-down',
                size=20,
                color='blue',
                line=dict(width=2, color='darkblue')
            ),
            text=[location_name],
            textposition='top center',
            textfont=dict(size=20, color='darkblue', family='Arial Black'),
            hovertemplate=f'<b>{location_name}</b><br>Lon: {coords["lon"]:.4f}<br>Lat: {coords["lat"]:.4f}<extra></extra>',
            showlegend=False,
            name=location_name
        )
        campus_location_traces.append(marker_trace)
    
    # Add a single legend entry for all key locations
    key_locations_legend = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            symbol='triangle-down',
            size=15,
            color='red',
            line=dict(width=2, color='darkred')
        ),
        name='Key Locations',
        showlegend=True
    )
    
    # Add a single legend entry for all campus locations
    campus_locations_legend = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            symbol='triangle-down',
            size=15,
            color='blue',
            line=dict(width=2, color='darkblue')
        ),
        name='Campus Locations',
        showlegend=True
    )
    
    # Create figure
    # Note: Traces are added in order, with later traces appearing on top
    # Transfer edges are added after regular edges, shortest path nodes are added after regular nodes
    mode_display = mode.replace('_', ' ').title()
    fig = go.Figure(data=legend_traces + [key_locations_legend, campus_locations_legend] + edge_traces + transfer_edge_traces + 
                    flow_label_traces + travel_time_label_traces + node_traces + bus_node_traces_with_labels + 
                    bus_node_traces_without_labels + shortest_path_node_traces + key_station_traces + key_location_traces + 
                    campus_location_traces)
    
    # Calculate trace indices for toggle buttons
    num_legend_traces = len(legend_traces) + 2  # +2 for key_locations_legend and campus_locations_legend
    num_edge_traces = len(edge_traces)
    num_transfer_edge_traces = len(transfer_edge_traces)
    num_flow_label_traces = len(flow_label_traces)
    num_travel_time_label_traces = len(travel_time_label_traces)
    num_node_traces = len(node_traces)
    num_bus_traces = len(bus_node_traces_with_labels)
    num_shortest_path_node_traces = len(shortest_path_node_traces)
    num_key_station_traces = len(key_station_traces)
    num_key_location_traces = len(key_location_traces)
    num_campus_location_traces = len(campus_location_traces)
    
    # Calculate trace positions
    idx = 0
    legend_idx = idx
    idx += num_legend_traces
    edge_idx = idx
    idx += num_edge_traces
    transfer_edge_idx = idx
    idx += num_transfer_edge_traces
    flow_idx = idx
    idx += num_flow_label_traces
    travel_time_idx = idx
    idx += num_travel_time_label_traces
    node_idx = idx
    idx += num_node_traces
    bus_with_labels_idx = idx
    idx += num_bus_traces
    bus_without_labels_idx = idx
    idx += num_bus_traces
    shortest_path_node_idx = idx
    idx += num_shortest_path_node_traces
    key_station_idx = idx
    idx += num_key_station_traces
    key_location_idx = idx
    idx += num_key_location_traces
    campus_location_idx = idx
    
    total_traces = idx + num_campus_location_traces
    
    # Create list of trace indices for each type
    flow_trace_indices = list(range(flow_idx, flow_idx + num_flow_label_traces))
    travel_time_trace_indices = list(range(travel_time_idx, travel_time_idx + num_travel_time_label_traces))
    bus_with_labels_indices = list(range(bus_with_labels_idx, bus_with_labels_idx + num_bus_traces))
    bus_without_labels_indices = list(range(bus_without_labels_idx, bus_without_labels_idx + num_bus_traces))
    
    # Add toggle buttons (using restyle to only modify specific traces)
    updatemenus = []
    
    # Flow labels toggle button
    if num_flow_label_traces > 0:
        updatemenus.append(
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": True}, flow_trace_indices],
                        label="Flow",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": False}, flow_trace_indices],
                        label="No Flow",
                        method="restyle"
                    )
                ],
                pad={"r": 5, "t": 5},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.12,
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.8)",
                borderwidth=1,
                bordercolor="gray",
                font=dict(size=10)
            )
        )
    
    # Travel time labels toggle button
    if num_travel_time_label_traces > 0:
        updatemenus.append(
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": True}, travel_time_trace_indices],
                        label="Times",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": False}, travel_time_trace_indices],
                        label="No Times",
                        method="restyle"
                    )
                ],
                pad={"r": 5, "t": 5},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.08,
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.8)",
                borderwidth=1,
                bordercolor="gray",
                font=dict(size=10)
            )
        )
    
    # Bus labels toggle button
    if num_bus_traces > 0:
        # Combine both sets of bus trace indices
        all_bus_indices = bus_with_labels_indices + bus_without_labels_indices
        # Create visibility patterns: show with_labels + hide without_labels
        show_labels_pattern = [True] * num_bus_traces + [False] * num_bus_traces
        hide_labels_pattern = [False] * num_bus_traces + [True] * num_bus_traces
        
        updatemenus.append(
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": show_labels_pattern}, all_bus_indices],
                        label="Bus Labels",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": hide_labels_pattern}, all_bus_indices],
                        label="No Bus Labels",
                        method="restyle"
                    )
                ],
                pad={"r": 5, "t": 5},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.04,
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.8)",
                borderwidth=1,
                bordercolor="gray",
                font=dict(size=10)
            )
        )
    
    fig.update_layout(
        # title=dict(
        #     text=f'MBTA Network - {mode_display}',
        #     font=dict(size=20)
        # ),
        xaxis=dict(
            title='Longitude',
            showgrid=True,
            zeroline=False,
            showticklabels=True
        ),
        yaxis=dict(
            title='Latitude',
            showgrid=True,
            zeroline=False,
            showticklabels=True,
            scaleanchor='x',
            scaleratio=0.74  # Adjusted for Boston's latitude (~42°N): cos(42°) ≈ 0.74 for proper geographic proportions
        ),
        hovermode='closest',
        plot_bgcolor='white',
        autosize=True,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            title=dict(text='Transit Lines'),
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        updatemenus=updatemenus
    )
    
    # Save to HTML
    output_filename = f'network_topology_{mode}.html'
    fig.write_html(output_filename)
    print(f"Saved to {output_filename}")
    
    # Show in browser
    fig.show()


def main():
    # Define data folder path
    data_folder = Path(__file__).parent / 'data'
    gtfs_folder = data_folder / 'MBTA_GTFS'
    rail_ridership_file = data_folder / 'Fall_2023_MBTA_Rail_Ridership_Data_by_SDP_Time,_Period_Route_Line,_and_Stop.csv'
    bus_ridership_file = data_folder / 'MBTA_Bus_Ridership_by_Time_Period,_Season,_Route_Line,_and_Stop_-_Fall.csv'
    stop_orders_file = data_folder / 'MBTA_Rapid_Transit_Stop_Orders.csv'
    
    print(f"Mode: {MODE}")
    print("Loading GTFS data...")
    
    # Load GTFS files
    stops, stop_times, trips, routes = load_gtfs_data(gtfs_folder)
    print(f"  Loaded GTFS data: {len(stops)} stops, {len(trips)} trips")
    
    # Get route types for the selected mode
    route_types = ROUTE_TYPES.get(MODE, [])
    if not route_types:
        print("Error: Invalid MODE specified.")
        return
    
    # Note: We don't pre-calculate trips_count here because different routes have
    # different headways. The normalization logic in flow.py will calculate per-edge
    # based on each route's specific headway and how many branches use each edge.
    trips_count = None
    
    # Create edges from GTFS data with ridership flow data
    print("\nCreating network edges...")
    
    if MODE == 'rail_only':
        # Rail only with ridership
        edges_df = create_edges_from_gtfs(
            stops, stop_times, trips, routes, [0, 1],
            ridership_filepath=rail_ridership_file,
            stop_orders_filepath=stop_orders_file,
            day_type=DAY_TYPE,
            time_period=TIME_PERIOD,
            ridership_type='rail',
            normalize_flow=NORMALIZE_FLOW,
            trips_count=trips_count
        )
        print(f"  Created {len(edges_df)} edges")
    elif MODE == 'bus_only':
        # Bus only with ridership (filtered to specific bus lines)
        # Note: Bus data already shows average_load (passengers per bus), so don't normalize
        edges_df = create_edges_from_gtfs(
            stops, stop_times, trips, routes, [3],
            ridership_filepath=bus_ridership_file,
            day_type=DAY_TYPE.lower(),  # Bus data uses lowercase
            time_period=TIME_PERIOD,
            ridership_type='bus',
            specific_routes=BUS_LINES,
            normalize_flow=False,  # Bus data is already per-bus load
            trips_count=None
        )
        print(f"  Created {len(edges_df)} edges")
    else:  # bus_and_rail
        # Create rail edges with rail ridership
        print("  Creating rail edges...")
        rail_edges = create_edges_from_gtfs(
            stops, stop_times, trips, routes, [0, 1],
            ridership_filepath=rail_ridership_file,
            stop_orders_filepath=stop_orders_file,
            day_type=DAY_TYPE,
            time_period=TIME_PERIOD,
            ridership_type='rail',
            normalize_flow=NORMALIZE_FLOW,
            trips_count=trips_count
        )
        rail_with_flow = (rail_edges['flow'] > 0).sum() if 'flow' in rail_edges.columns else 0
        print(f"    Rail: {len(rail_edges)} edges, {rail_with_flow} with flow data")
        
        # Create bus edges with bus ridership (filtered to specific bus lines)
        # Note: Bus data already shows average_load (passengers per bus), so don't normalize
        print("  Creating bus edges...")
        bus_edges = create_edges_from_gtfs(
            stops, stop_times, trips, routes, [3],
            ridership_filepath=bus_ridership_file,
            day_type=DAY_TYPE.lower(),  # Bus data uses lowercase
            time_period=TIME_PERIOD,
            ridership_type='bus',
            specific_routes=BUS_LINES,
            normalize_flow=False,  # Bus data is already per-bus load
            trips_count=None
        )
        bus_with_flow = (bus_edges['flow'] > 0).sum() if 'flow' in bus_edges.columns else 0
        print(f"    Bus: {len(bus_edges)} edges, {bus_with_flow} with flow data")
        
        # Combine both edge dataframes
        edges_df = pd.concat([rail_edges, bus_edges], ignore_index=True)
        total_with_flow = (edges_df['flow'] > 0).sum()
        print(f"  Combined: {len(edges_df)} total edges, {total_with_flow} with flow data")
    
    # Handle line closure simulation
    if SIMULATE_LINE_CLOSURE:
        closure_value = str(SIMULATE_LINE_CLOSURE)
        result = filter_closed_lines(edges_df, closure_value)
        
        if result[0] is None:
            print("  ERROR: No edges remaining after line closure!")
            return
        
        edges_df = result[0]  # Update edges_df with filtered version
    
    # Create graph with geographic positions
    print("\nBuilding network graph...")
    G, pos, node_labels, node_routes = create_graph_with_positions(edges_df, stops)
    print(f"  {G.number_of_nodes()} stations, {G.number_of_edges()} connections")
    
    # Handle station/stop closure simulation
    station_closure_info = None
    if SIMULATE_STOP_STATION_CLOSURE:
        station_closure_info = close_station(G, pos, node_labels, SIMULATE_STOP_STATION_CLOSURE, edges_df)
        if station_closure_info:
            print(f"  After station closure: {G.number_of_nodes()} stations, {G.number_of_edges()} connections")
    
    # Debug: Show which routes serve each node (useful for verifying shared stops)
    if SIMULATE_LINE_CLOSURE:
        print(f"\n  Checking for nodes that serve multiple routes (including near-closure routes):")
        multi_route_nodes = [(node, routes) for node, routes in node_routes.items() if len(routes) > 1]
        if multi_route_nodes:
            print(f"  Found {len(multi_route_nodes)} nodes serving multiple routes")
            # Show a few examples
            for node, routes in multi_route_nodes[:5]:
                print(f"    - {node_labels.get(node, node)}: routes {sorted(routes)}")
        else:
            print(f"  No nodes serving multiple routes found")
    
    # Identify stations near key/campus locations
    print("\nIdentifying stations near key/campus locations...")
    key_stations = identify_key_stations(G, pos, KEY_LOCATIONS, CAMPUS_LOCATIONS, KEY_LOCATION_RADIUS)
    print(f"  Found {len(key_stations)} stations within {KEY_LOCATION_RADIUS} degrees of key/campus locations:")
    for station_id, info in key_stations.items():
        station_name = node_labels.get(station_id, station_id)
        print(f"    - {station_name} -> {info['location']} (distance: {info['distance']:.4f})")
    
    # Add key station attribute to graph nodes
    for node in G.nodes():
        if node in key_stations:
            G.nodes[node]['is_key_station'] = True
            G.nodes[node]['key_location'] = key_stations[node]['location']
        else:
            G.nodes[node]['is_key_station'] = False
    
    # Add travel time weights to edges (ALWAYS load for pathfinding, regardless of display)
    if MODE in ['rail_only', 'bus_and_rail']:
        print("\nAdding travel time data from GTFS schedules...")
        try:
            travel_times_df = load_travel_times_from_gtfs(gtfs_folder, stops_df=stops)
            add_travel_times_to_graph(G, travel_times_df)
        except Exception as e:
            print(f"  Warning: Could not load travel time data: {e}")
            print(f"  Continuing without travel times...")
    
    # Verify line closure was successful (if applicable) and REROUTE DEMAND
    if SIMULATE_LINE_CLOSURE:
        closure_value = str(SIMULATE_LINE_CLOSURE)
        def matches_closure(route_id):
            route_id_str = str(route_id)
            if route_id_str == closure_value:
                return True
            if route_id_str.startswith(closure_value + "-"):
                return True
            return False
        
        routes_in_graph = set()
        for u, v, data in G.edges(data=True):
            routes_in_graph.update(data.get('routes', set()))
        
        closed_routes_found = [r for r in routes_in_graph if matches_closure(r)]
        if closed_routes_found:
            print(f"\n  ERROR: Closed line variants still found in graph: {closed_routes_found}")
        else:
            print(f"\n  Verified: Closed line '{closure_value}' successfully removed from graph")
            print(f"  Active routes in graph: {sorted(routes_in_graph)}")
            
            # Perform Gravity-Based Rerouting
            try:
                # Load raw ridership data for the rerouting logic
                rail_data = pd.read_csv(rail_ridership_file)
                rail_data['parent_stop_id'] = rail_data['parent_station']
                
                bus_data = pd.read_csv(bus_ridership_file)
                stops_dict = stops.set_index('stop_id')['parent_station'].to_dict()
                def get_parent(stop_id):
                    sid = str(stop_id)
                    return stops_dict.get(sid, sid) if pd.notna(stops_dict.get(sid)) else sid
                bus_data['parent_stop_id'] = bus_data['stop_id'].apply(get_parent)
                
                # Concatenate relevant data
                all_ridership = pd.concat([
                    rail_data[['route_id', 'parent_stop_id', 'average_ons', 'average_offs']],
                    bus_data[['route_id', 'parent_stop_id', 'average_ons', 'average_offs']]
                ], ignore_index=True)
                
                # Create stops coordinate dictionary
                stops_coord_dict = stops.set_index('stop_id')[['stop_lat', 'stop_lon']].to_dict('index')
                
                # Calculate normalization factor for rerouting
                # Note: For multi-route networks, this is an approximation.
                # We use a representative route (Red Line) as the baseline.
                # The rerouted flows will be distributed proportionally.
                normalization_factor = 1.0
                if NORMALIZE_FLOW:
                    # Use Red Line as baseline for normalization (12 min headway)
                    normalization_factor = calculate_trips_from_headway(TIME_PERIOD, 'Red')
                    print(f"  Using baseline normalization factor: {normalization_factor} (Red Line)")
                
                # Reroute the demand
                reroute_line_closure(G, pos, stops, closure_value, all_ridership, 
                                   stops_coord_dict, normalization_factor)
                              
            except Exception as e:
                print(f"  Error during line closure rerouting: {e}")
                import traceback
                traceback.print_exc()
    
    # Handle station closure rerouting
    if station_closure_info and SIMULATE_STOP_STATION_CLOSURE:
        try:
            # Load ridership data if not already loaded
            if 'all_ridership' not in locals():
                rail_data = pd.read_csv(rail_ridership_file)
                rail_data['parent_stop_id'] = rail_data['parent_station']
                
                bus_data = pd.read_csv(bus_ridership_file)
                stops_dict_temp = stops.set_index('stop_id')['parent_station'].to_dict()
                def get_parent(stop_id):
                    sid = str(stop_id)
                    return stops_dict_temp.get(sid, sid) if pd.notna(stops_dict_temp.get(sid)) else sid
                bus_data['parent_stop_id'] = bus_data['stop_id'].apply(get_parent)
                
                all_ridership = pd.concat([
                    rail_data[['route_id', 'parent_stop_id', 'average_ons', 'average_offs']],
                    bus_data[['route_id', 'parent_stop_id', 'average_ons', 'average_offs']]
                ], ignore_index=True)
                
                stops_coord_dict = stops.set_index('stop_id')[['stop_lat', 'stop_lon']].to_dict('index')
            
            # Calculate normalization factor for station closure rerouting
            # Note: For multi-route networks, this is an approximation.
            # We use a representative route (Red Line) as the baseline.
            normalization_factor = 1.0
            if NORMALIZE_FLOW:
                # Use Red Line as baseline for normalization (12 min headway)
                normalization_factor = calculate_trips_from_headway(TIME_PERIOD, 'Red')
                print(f"  Using baseline normalization factor: {normalization_factor} (Red Line)")
            
            # Reroute the station closure demand
            reroute_station_closure(G, pos, station_closure_info, all_ridership, 
                                   stops_coord_dict, normalization_factor)
        
        except Exception as e:
            print(f"  Error during station closure rerouting: {e}")
            import traceback
            traceback.print_exc()
    
    # Compute shortest path if source and destination are specified
    # Note: transfer edges are added internally to a copy of the graph for pathfinding
    shortest_path = None
    transfer_edges = []
    if SHORTEST_PATH_SRC_DST:
        source_location, target_location = SHORTEST_PATH_SRC_DST
        all_locations = {}
        all_locations.update(KEY_LOCATIONS)
        all_locations.update(CAMPUS_LOCATIONS)
        
        shortest_path, path_stats, transfer_edges = find_and_compute_shortest_path(
            source_location, target_location, G, pos, node_labels, all_locations,
            transfer_radius=KEY_LOCATION_RADIUS, transfer_time=480.0, node_routes=node_routes
        )  # 60 seconds = ~1 minute walk between nearby stops
    
    # Generate interactive visualization
    print("\nGenerating visualization...")
    visualize_graph(G, pos, node_labels, node_routes, MODE, 
                   directed=DIRECTED_EDGES, show_flow_labels=SHOW_FLOW_LABELS, 
                   show_travel_time_labels=SHOW_TRAVEL_TIMES_LABELS,
                   shortest_path=shortest_path, transfer_edges=transfer_edges)


if __name__ == "__main__":
    main()