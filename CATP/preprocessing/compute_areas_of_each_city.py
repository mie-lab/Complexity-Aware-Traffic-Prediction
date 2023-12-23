from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


## max_lat, min_lat computed using
## inside ProcessRaw.py
# if np.random.rand() < 0.5:
#     print(self.cityname, self.i_o_length, self.grid_size, "max_lon, max_lat", df["x"].max(), df["y"].max(),
#           "min_lon, min_lat", df["x"].min(), df["y"].min())

# City coordinates for scale 26 and scale 27 (they are the same)
cities = {
    "London": {"max_lon": 0.0670603, "max_lat": 51.69845774391412, "min_lon": -0.3688872, "min_lat": 51.25631772160935},
    "Melbourne": {"max_lon": 145.1928566, "max_lat": -37.6119873, "min_lon": 144.7571961, "min_lat": -38.1034605},
    "Madrid": {"max_lon": -3.5806081, "max_lat": 40.515610660745345, "min_lon": -3.836937105036116, "min_lat": 40.3323645}
}

# Calculate dimensions
city_dimensions = {}
for city, coords in cities.items():
    height = haversine(coords["min_lon"], coords["min_lat"], coords["min_lon"], coords["max_lat"])
    width = haversine(coords["min_lon"], coords["max_lat"], coords["max_lon"], coords["max_lat"])
    city_dimensions[city] = {"Height (km)": height, "Width (km)": width}
    print (city, city_dimensions[city])
    print ("Scale 25: ", height/25, width/25)
    print ("Scale 55: ", height/55, width/55)
    print ("Scale 105: ", height/105, width/105)

    # print (city_dimensions)


