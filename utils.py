import pandas as pd

# MBTA Line Colors (official colors)
ROUTE_COLORS = {
    'Red': '#DA291C',
    'Orange': '#ED8B00',
    'Blue': '#003DA5',
    'Green-B': '#00843D',
    'Green-C': '#00843D',
    'Green-D': '#00843D',
    'Green-E': '#00843D',
    'Mattapan': '#DA291C',
    # Silver Line
    'SL1': '#7C878E',
    'SL2': '#7C878E',
    'SL3': '#7C878E',
    'SL4': '#7C878E',
    'SL5': '#7C878E',
    # Default for buses and other routes
    'default': '#999999'
}

def load_bus_data(filepath, time_period):
    """Load and filter bus ridership data."""
    df = pd.read_csv(filepath)
    df = df[df['time_period_name'] == time_period].copy()
    return df

def load_rail_data(filepath, time_period):
    """Load and filter rail ridership data."""
    df = pd.read_csv(filepath)
    df = df[df['time_period_name'] == time_period].copy()
    return df

def load_rail_stop_orders(filepath):
    """Load rail stop order data."""
    df = pd.read_csv(filepath)
    return df

def load_gtfs_data(gtfs_folder):
    """Load relevant GTFS files."""
    # Force stop_id to be string to avoid type mismatches between files
    stops = pd.read_csv(gtfs_folder / 'stops.txt', dtype={'stop_id': str, 'parent_station': str})
    stop_times = pd.read_csv(gtfs_folder / 'stop_times.txt', dtype={'stop_id': str}, low_memory=False)
    trips = pd.read_csv(gtfs_folder / 'trips.txt', dtype={'route_id': str})
    routes = pd.read_csv(gtfs_folder / 'routes.txt', dtype={'route_id': str})
    
    return stops, stop_times, trips, routes

def filter_routes_by_type(routes, route_types):
    """Filter routes by type (bus, rail, etc.)."""
    filtered = routes[routes['route_type'].isin(route_types)]
    return filtered