import requests

from shapely.geometry import mapping, shape
from shapely.prepared import prep
from shapely.geometry import Point


data = requests.get(
    "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson").json()

countries = {}
for feature in data["features"]:
    geom = feature["geometry"]
    country = feature["properties"]["ADMIN"]
    countries[country] = prep(shape(geom))

# print(len(countries))


def get_country(lon, lat):
    point = Point(lon, lat)
    for country, geom in countries.items():
        if geom.contains(point):
            return country

    return "unknown"


# print(get_country(-8.0, 53.0))
