import pandas as pd

# Datatypes of each column
dtype = {
    'airline_name': str,
    'link': str,
    'title': str,
    'author': str,
    'author_country': str,
    'date': str,
    'content': str,
    'aircraft': str,
    'type_traveller': str,
    'cabin_flown': str,
    'route': str,
    'overall_rating': float,
    'seat_comfort_rating': float,
    'cabin_staff_rating': float,
    'food_beverages_rating': float,
    'inflight_entertainment_rating': float,
    'ground_service_rating': float,
    'wifi_connectivity_rating': float,
    'value_money_rating': float,
    'recommended': int}

# Read in data
data = pd.read_table('airline.csv', sep=',', dtype=dtype)

# Remove unused columns
data = data.drop(['link', 'title', 'author', 'date'], axis=1)
