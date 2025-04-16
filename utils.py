#utils
import pandas as pd
import requests

def load_planetary_data():
    url = "https://raw.githubusercontent.com/OpenExoplanetCatalogue/oec_tables/master/comma_separated/open_exoplanet_catalogue.txt"
    planetary_data = pd.read_csv(url)
    return planetary_data

def get_asteroid_data():
    url = "https://api.nasa.gov/neo/rest/v1/neo/browse"
    params = {'api_key': 'DEMO_KEY', 'size': 5}
    response = requests.get(url, params=params)
    data = response.json()
    return data['near_earth_objects']
