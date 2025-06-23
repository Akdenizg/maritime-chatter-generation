import random
import geopandas as gpd
from shapely import LineString
from shapely.geometry import Point
from geopy.distance import geodesic
import pandas as pd
import json
import re
from shapely.ops import nearest_points
import warnings
import time
from datetime import datetime

class MaritimeAnalysis:
    def __init__(self, land_shapefile, geonames_data_path, vessel_data_path, task_name = None):
        # Load GSHHG land shapefile (shorelines)
        self.land = gpd.read_file(land_shapefile)
        self.task_name = task_name
        self.char_to_string = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
        'A': 'Alfa', 'B': 'Bravo', 'C': 'Charlie', 'D': 'Delta', 'E': 'Echo',
        'F': 'Foxtrot', 'G': 'Golf', 'H': 'Hotel', 'I': 'India', 'J': 'Juliet',
        'K': 'Kilo', 'L': 'Lima', 'M': 'Mike', 'N': 'November', 'O': 'Oscar',
        'P': 'Papa', 'Q': 'Quebec', 'R': 'Romeo', 'S': 'Sierra', 'T': 'Tango',
        'U': 'Uniform', 'V': 'Victor', 'W': 'Whisky', 'X': 'X-ray', 'Y': 'Yankee', 'Z': 'Zulu'
        }

        self.single_digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        self.tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        self.teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]

        # Load Geonames data
        columns = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 
                   'feature_class', 'feature_code', 'country_code', 'cc2', 'admin1_code', 
                   'admin2_code', 'admin3_code', 'admin4_code', 'population', 'elevation', 
                   'dem', 'timezone', 'modification_date']
        self.geonames_df = pd.read_csv(geonames_data_path, sep='\t', header=None, 
                                       names=columns, usecols=['name', 'latitude', 'longitude', 'feature_code', 'country_code','population'])
        
        self.vessel_df = pd.read_pickle(vessel_data_path)
        self.vessel_df['Vessel Name'] = self.vessel_df['Vessel Name'].apply(lambda x: re.sub(r'[^a-zA-Z0-9 ]', '', x))
        self.vessel_df['Vessel Name'] = self.vessel_df['Vessel Name'].apply(lambda x:' '.join(x.split()))
        self.vessel_df = self.vessel_df[self.vessel_df["Vessel Name"] != ""]
        self.vessel_df['Call Sign'] = self.vessel_df['Call Sign'].apply(
    lambda x: re.sub(r'[^a-zA-Z0-9 ]', '', x) if x is not None and x.lower() != "unknown" and pd.isna(x) == False else x
    )

    def generate_random_coordinate(self):
        # Generate random latitude and longitude
        lat = random.uniform(-60, 90)
        lon = random.uniform(-180, 180)
        return Point(lon, lat)  # Note: Shapely stores as (longitude, latitude)

    def is_maritime_area(self, point):
        # Re-project the point to match the projected CRS of the land data
        point_gdf = gpd.GeoDataFrame([point], columns=["geometry"], crs="EPSG:4326")
        point_projected = point_gdf.to_crs(epsg=4326).geometry[0]
        
        # Check if the point is within any land area
        return not self.land.contains(point_projected).any()

    def calculate_distance_to_nearest_land(self, point):
        warnings.filterwarnings("ignore")
        # Re-project the point to match the projected CRS of the land data
        point_gdf = gpd.GeoDataFrame([point], columns=["geometry"], crs="EPSG:4326")
        point_projected = point_gdf.to_crs(epsg=4326).geometry[0]
        
        # Calculate the nearest distance from the point to any land feature
        self.land["distance"] = self.land.distance(point_projected)
        nearest_geom = self.land["distance"].idxmin()  # Get the nearest geometry index
        nearest_polygon = self.land.geometry.iloc[nearest_geom]  # Get nearest land geometry

        # Find the closest point on the polygon to the point_projected
        _, nearest_point_on_polygon = nearest_points(point_projected, nearest_polygon)

        # Convert the nearest point back to geographic coordinates for distance calculation
        nearest_point = gpd.GeoSeries([nearest_point_on_polygon], crs="EPSG:4326").iloc[0]
        nearest_coord = (nearest_point.y, nearest_point.x)  # Convert to (latitude, longitude) format
        
        # Convert maritime point to (latitude, longitude) format
        maritime_coord = (point.y, point.x)  # Convert the random maritime point to (latitude, longitude)
        
        # Calculate geodesic distance
        distance_to_land = geodesic(maritime_coord, nearest_coord).nautical
        
        return distance_to_land, nearest_coord

    def calculate_compass_direction(self, closest_place, maritime_coord):
        lat_diff = maritime_coord.y - closest_place['latitude']
        lon_diff = maritime_coord.x - closest_place['longitude']

        vertical_direction = None
        horizontal_direction = None

        # Determine vertical direction
        if lat_diff > 0:
            vertical_direction = "north"
        else:
            vertical_direction = "south"

        # Determine horizontal direction
        if lon_diff > 0:
            horizontal_direction = "east"
        else:
            horizontal_direction = "west"

        # Return the primary direction
        if vertical_direction and horizontal_direction:
            return f"{vertical_direction} {horizontal_direction}"
        elif vertical_direction:
            return vertical_direction
        elif horizontal_direction:
            return horizontal_direction
        
    def find_closest_water_body(self, lat, lon):
        warnings.filterwarnings("ignore")
        # Define the coordinate for which we want to find the closest water
        target_coord = (lat, lon)
        
        max_distance = 10
        bounding_box_size = 5  # Initial bounding box size
        increment_size = 5  # Increment size for expanding the bounding box

        # Define a list of prioritized feature codes
        priority_features = ['BAY', 'BAYS', 'COVE', 'GULF', 'LGN']

        filtered_by_priority = pd.DataFrame()
        seas_or_oceans = pd.DataFrame()

        while len(filtered_by_priority) < 10 or len(seas_or_oceans) < 4:
            # Define a bounding box around the target coordinate
            bounding_box = (lon - bounding_box_size, lat - bounding_box_size,
                            lon + bounding_box_size, lat + bounding_box_size)
            
            # Filter geonames_df to include only water bodies within the bounding box
            filtered_df = self.geonames_df[(self.geonames_df['longitude'] >= bounding_box[0]) & 
                                            (self.geonames_df['longitude'] <= bounding_box[2]) &
                                            (self.geonames_df['latitude'] >= bounding_box[1]) & 
                                            (self.geonames_df['latitude'] <= bounding_box[3])]

            # Filter land polygons to include only those within the bounding box
            land_in_bbox = self.land.cx[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]]

            # Filter for priority features
            filtered_by_priority = filtered_df[filtered_df['feature_code'].isin(priority_features)]

            # If less than 10 priority features found, expand the bounding box
            if len(filtered_by_priority) < 10:
                bounding_box_size += increment_size
                continue

            # Filter for seas or oceans
            seas_or_oceans = filtered_df[filtered_df['feature_code'].isin(['OCN', 'SEA'])]

            # If less than 4 seas or oceans found, expand the bounding box
            if len(seas_or_oceans) < 4:
                bounding_box_size += increment_size
                continue

        # Calculate distances for priority features
        filtered_by_priority['distance'] = filtered_by_priority.apply(lambda row: geodesic(target_coord, (row['latitude'], row['longitude'])).nautical, axis=1)
        # Keep only those within max_distance and get the closest 10 entries
        filtered_within_distance = filtered_by_priority[filtered_by_priority['distance'] <= max_distance].nsmallest(10, 'distance')

        if not filtered_within_distance.empty:
            sorted_filtered = filtered_within_distance.sort_values(by='distance')
            for _, water_body in sorted_filtered.iterrows():
                water_body_point = Point(water_body['longitude'], water_body['latitude'])
                line_to_water_body = gpd.GeoSeries([LineString([Point(lon, lat), water_body_point])], crs="EPSG:4326")
                if not land_in_bbox.intersects(line_to_water_body.unary_union).any():
                    water_body["name"] = re.sub(r'[^a-zA-Z0-9 ]', '', water_body["name"])
                    return water_body

        # Calculate distances for seas and oceans and get the closest 4 entries
        seas_or_oceans['distance'] = seas_or_oceans.apply(lambda row: geodesic(target_coord, (row['latitude'], row['longitude'])).nautical, axis=1)
        sorted_seas_or_oceans = seas_or_oceans.nsmallest(4, 'distance')
        for _, water_body in sorted_seas_or_oceans.iterrows():
            water_body_point = Point(water_body['longitude'], water_body['latitude'])
            line_to_water_body = gpd.GeoSeries([LineString([Point(lon, lat), water_body_point])], crs="EPSG:4326")
            if not land_in_bbox.intersects(line_to_water_body.unary_union).any():
                return water_body

        return None

    def find_closest_place(self, lat, lon):
        warnings.filterwarnings("ignore")
        target_coord = (lat, lon)

        if lat > 60 or lat < -60:
            tolerance = 10
        elif lat > 30 or lat < -30:
            tolerance = 5
        else:
            tolerance = 3        

        bounding_box = (lon - tolerance, lat - 2, lon + tolerance, lat + 2)
        filtered_df = self.geonames_df[(self.geonames_df['longitude'] >= bounding_box[0]) & 
                                        (self.geonames_df['longitude'] <= bounding_box[2]) &
                                        (self.geonames_df['latitude'] >= bounding_box[1]) & 
                                        (self.geonames_df['latitude'] <= bounding_box[3])]

        filtered_gdf = gpd.GeoDataFrame(
            filtered_df,
            geometry=gpd.points_from_xy(filtered_df.longitude, filtered_df.latitude),
            crs="EPSG:4326"
        )

        priority_features = ['ISL', 'ISLF', 'ISLT', 'PPL', 'PPLL', 'PPLS', 'PPLC', 'ADM1', 'ADM2', 'ADM3', 'ADM4', 'ADM5', 'ADMD', 'CST', 'PRT', 'HBR', 'PEN', 'CAPE', 'BCH', 'BCHS']
        filtered_by_priority = filtered_gdf[filtered_gdf['feature_code'].isin(priority_features)]
        filtered_by_priority['country_code'] = filtered_by_priority['country_code'].fillna('None')

        if filtered_by_priority.empty:
            return None

        filtered_by_priority['distance'] = filtered_by_priority.apply(
            lambda row: geodesic(target_coord, (row['latitude'], row['longitude'])).nautical, axis=1
        )

        sorted_places = filtered_by_priority.sort_values(by='distance')

        for _, place in sorted_places.iterrows():
            place_point = Point(place['longitude'], place['latitude'])
            line_to_place = gpd.GeoSeries([LineString([Point(lon, lat), place_point])], crs="EPSG:4326")
            
            # Check intersections with land polygons
            land_in_bbox = self.land.cx[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]]
            intersecting_polygons_count = land_in_bbox.intersects(line_to_place.unary_union).sum()

            # Accept if the line intersects with at most one polygon
            if intersecting_polygons_count <= 1:
                for col in place.index:
                    if isinstance(place[col], str):
                        place[col] = re.sub(r'\s*\(.*?\)|\s*\[.*?\]', '', place[col]).strip()

                if place['feature_code'] == 'OCN':
                    place["name"] = place['name'].split(" Ocean")[0] + " Ocean"

                ports_df = self.geonames_df[self.geonames_df['feature_code'].isin(['PRT'])]
                if not ports_df.empty:
                    maritime_coord = (lat, lon)
                    ports_df['distance_to_port'] = ports_df.apply(
                        lambda row: geodesic(maritime_coord, (row['latitude'], row['longitude'])).nautical, axis=1
                    )
                    
                    closest_port_index = ports_df['distance_to_port'].idxmin()
                    closest_port = ports_df.loc[closest_port_index]
                    
                    place['distance_to_nearest_port'] = ports_df['distance_to_port'].min()
                    place['closest_port_name'] = re.sub(r'\s*\(.*?\)', '', closest_port['name'])
                    place['closest_port_coordinates'] = (closest_port['latitude'], closest_port['longitude'])

                harbors_df = self.geonames_df[self.geonames_df['feature_code'].isin(['HBR'])]
                if not harbors_df.empty:
                    maritime_coord = (lat, lon)
                    harbors_df['distance_to_harbor'] = harbors_df.apply(
                        lambda row: geodesic(maritime_coord, (row['latitude'], row['longitude'])).nautical, axis=1
                    )
                    
                    closest_harbor_index = harbors_df['distance_to_harbor'].idxmin()
                    closest_harbor = harbors_df.loc[closest_harbor_index]
                    
                    place['distance_to_nearest_harbor'] = harbors_df['distance_to_harbor'].min()
                    place['closest_harbor_name'] = re.sub(r'\s*\(.*?\)', '', closest_harbor['name'])
                    place['closest_harbor_coordinates'] = (closest_harbor['latitude'], closest_harbor['longitude'])    

                place['name'] = re.sub(r'\s*\(.*?\)', '', place['name'])
                return place

        return None
    
    # Helper function to convert a number to its word representation
    def number_to_words(self, num, digit_by_digit=False, is_decimal=False, decimal_connector=None):
        if is_decimal:
            int_part, decimal_part = str(num).split(".")
            int_words = " ".join(self.single_digits[int(digit)] for digit in int_part)
            decimal_words = " ".join(self.single_digits[int(digit)] for digit in decimal_part)
            return f"{int_words} {decimal_connector} {decimal_words}"
        
        if digit_by_digit:
            return " ".join(self.single_digits[int(digit)] for digit in str(num) if digit.isdigit())
        else:
            if num < 10:
                return self.single_digits[num]
            elif 10 <= num < 20:
                return self.teens[num - 10]
            elif 20 <= num < 100:
                return self.tens[num // 10] + (f"-{self.single_digits[num % 10]}" if num % 10 != 0 else "")
            elif 100 <= num < 1000:
                hundreds = self.single_digits[num // 100] + " hundred"
                remainder = num % 100
                if remainder != 0:
                    return hundreds + " " + self.number_to_words(remainder, digit_by_digit=False)
                else:
                    return hundreds
            else:
                return str(num)

    def convert_to_dms(self, latitude, longitude):
        digit_by_digit = random.choice([True, False])  

        # Randomly decide whether to use "decimal" or "point" for decimals
        decimal_connector = random.choice(["decimal", "point"])

        def decimal_to_dms(degree, is_latitude=True, include_minutes=True):
            is_negative = degree < 0
            degree = abs(degree)
            d = int(degree)
            
            if random.choice([True, False]):
                m = int((degree - d) * 60)
            else:    
                m = (degree - d) * 60

            if is_latitude:
                direction = "North" if not is_negative else "South"
            else:
                direction = "East" if not is_negative else "West"

            d_str = self.number_to_words(d, digit_by_digit)

            if include_minutes:
                if m % 1 == 0:
                    m_str = self.number_to_words(int(m), digit_by_digit)  
                else:
                    m_str = self.number_to_words(round(m, 2), digit_by_digit=digit_by_digit, is_decimal=True, decimal_connector=decimal_connector)  

                return f"{d_str} degrees {m_str} minutes {direction}"
            else:
                return f"{d_str} degrees {direction}"

        include_minutes = random.choice([True, False])  
        lat_dms = decimal_to_dms(latitude, is_latitude=True, include_minutes=include_minutes)
        lon_dms = decimal_to_dms(longitude, is_latitude=False, include_minutes=include_minutes)

        return lat_dms, lon_dms, digit_by_digit
    
    def mmsi_to_words(self, mmsi):
        # Convert MMSI to string and map each digit to its word representation
        return " ".join(self.single_digits[int(digit)] for digit in str(mmsi) if digit.isdigit())
    
    def convert_call_sign(self, s):
        result = []
        for char in s:
            result.append(self.char_to_string.get(char.upper(), char))
        return ' '.join(result)
    
    def get_random_vessel(self):
        random_vessel = self.vessel_df.sample(n=1)

        vessel = {
                    "Vessel Name": None,
                    "MMSI": None,
                    "Call Sign": None,
                    "Vessel Type": 'Motor Vessel',
                    "can_have_cargo": None,
               }

        vessel["Vessel Name"] = random_vessel["Vessel Name"].iloc[0]
        # vessel["Vessel Name"] = ' '.join(vessel["Vessel Name"].split())
        # vessel["Vessel Name"] = re.sub(r'[^a-zA-Z0-9 ]', '', vessel["Vessel Name"])
        # vessel["Vessel Name"] = re.sub(r'\s*\(.*?\)', '', vessel["Vessel Name"]).strip()
        # vessel["Vessel Name"] = re.sub(r'\s*\(.*', '', vessel["Vessel Name"]).strip()
        # # Remove all characters except letters and numbers
        # vessel["Vessel Name"] = re.sub(r'[^a-zA-Z0-9 ]', ' ', vessel["Vessel Name"])

        if random_vessel["MMSI"].iloc[0] != "" and random.uniform(0.0, 1.0) >= 0.6:
            vessel["MMSI"] = self.mmsi_to_words(random_vessel["MMSI"].iloc[0])

        if (random_vessel["Call Sign"].iloc[0] != "" and random_vessel["Call Sign"].iloc[0] != None and random_vessel["Call Sign"].iloc[0].lower() != 'unknown'
            and pd.notna(random_vessel["Call Sign"].iloc[0])) and random.uniform(0.0, 1.0) >= 0.5:
            
            #vessel["Call Sign"] = re.sub(r'[^a-zA-Z0-9 ]', '', random_vessel["Call Sign"].iloc[0])
            vessel["Call Sign"] = self.convert_call_sign(random_vessel["Call Sign"].iloc[0])
        
        if random_vessel["Vessel Type"].iloc[0] != "Other" and random.uniform(0.0, 1.0) >= 0.4:
            vessel['Vessel Type'] = random_vessel["Vessel Type"].iloc[0]

        allowed_to_have_cargo = ["Tanker", "Passenger Vessel", "Cargo Vessel"]

        if vessel['Vessel Type'] in allowed_to_have_cargo:
            vessel['can_have_cargo'] = "True"           

        return vessel

    def execute(self, used_mmsi_list):
        with open('/home/gakdeniz/Dev/ma-llm-tuning/all_countries/country_codes.json', 'r', encoding='utf-8') as json_file:
            country_codes = json.load(json_file)
        
        while True:
            vessel = self.get_random_vessel()
            if vessel["MMSI"] in used_mmsi_list:
                continue

            is_point_valid = False
            while is_point_valid == False:        
                point = self.generate_random_coordinate()

                if not self.is_maritime_area(point):
                    continue

                distance_to_land, nearest_land_coord = self.calculate_distance_to_nearest_land(point)
                
                if self.task_name == "reporting_grounding":
                    range_land = 1
                elif vessel["Vessel Type"] == "Port Tender":
                    range_land = 5
                else:
                    range_land = 60    

                if distance_to_land > range_land:
                    continue
                
                closest_place = self.find_closest_place(point.y, point.x)
                    
                if closest_place is None:
                    continue

                is_point_valid = True
                    
            closest_water_body = self.find_closest_water_body(point.y, point.x)
                
            compass_direction = self.calculate_compass_direction(closest_place, point)

            lat_dms, lon_dms, digit_by_digit = self.convert_to_dms(point.y, point.x)

            country_name = country_codes[closest_place['country_code']] if closest_place['country_code'] not in ['None', None] else None
            if country_name != None:
                country_name = re.sub(r'\s*\(.*?\)', '', country_name).strip() 

            scenario_inspect = {
                "vessel_name": vessel["Vessel Name"],
                "vessel_MMSI": vessel["MMSI"],
                "vessel_call_sign": vessel["Call Sign"],
                "vessel_type": vessel["Vessel Type"],
                "vessel_coordinate_lat": point.y,
                "vessel_coordinate_long": point.x,
                "vessel_coordinate_dms": (lat_dms, lon_dms),
                "nearest_land_point_lat": nearest_land_coord[0],
                "nearest_land_point_long": nearest_land_coord[1],
                "distance_to_nearest_land": self.number_to_words(round(distance_to_land), digit_by_digit),
                "compass_direction": compass_direction,
                "closest_place_name": closest_place["name"],
                "closest_place_lat": closest_place['latitude'],
                "closest_place_long": closest_place['longitude'],
                "closest_place_feature_code": closest_place['feature_code'],
                "distance_to_nearest_place": self.number_to_words(round(closest_place['distance']), digit_by_digit),
                "closest_place_country": country_name,
                "distance_to_nearest_port": self.number_to_words(round(closest_place['distance_to_nearest_port']), digit_by_digit),
                "nearest_port": closest_place['closest_port_name'],
                "nearest_port_coordinates": closest_place['closest_port_coordinates'],
                "distance_to_nearest_harbor": self.number_to_words(round(closest_place['distance_to_nearest_harbor']), digit_by_digit),
                "nearest_harbor": closest_place['closest_harbor_name'],
                "nearest_harbor_coordinates": closest_place['closest_harbor_coordinates'],
                "digit_by_digit": digit_by_digit,
                "can_have_cargo": vessel['can_have_cargo'],
            }

            scenario_input = {
                "vessel_name": vessel["Vessel Name"],
                "vessel_MMSI": vessel["MMSI"],
                "vessel_call_sign": vessel["Call Sign"],
                "vessel_type": vessel["Vessel Type"],
                "vessel_coordinate_dms": f"{lat_dms}, {lon_dms}",
                "compass_direction": compass_direction,
                "closest_place_name": closest_place["name"],
                "distance_to_nearest_place": self.number_to_words(round(closest_place['distance']) if round(closest_place['distance']) != 0 else 1, digit_by_digit),
                "closest_place_country": country_name,
                "distance_to_nearest_port": self.number_to_words(round(closest_place['distance_to_nearest_port']) if round(closest_place['distance_to_nearest_port']) != 0 else 1, digit_by_digit),
                "nearest_port": closest_place['closest_port_name'],
                "distance_to_nearest_harbor": self.number_to_words(round(closest_place['distance_to_nearest_harbor']) if round(closest_place['distance_to_nearest_harbor']) != 0 else 1, digit_by_digit),
                "nearest_harbor": closest_place['closest_harbor_name'],
                "digit_by_digit": digit_by_digit,
                "can_have_cargo": vessel['can_have_cargo'],
            }

            if closest_place['feature_code'] in ['ISL', 'PEN']:
                scenario_input['distance_to_nearest_place'] = scenario_inspect['distance_to_nearest_land'] if scenario_inspect['distance_to_nearest_land'] != 'zero' else 'one'

            if closest_water_body is not None and closest_water_body['distance'] <= 400:
                scenario_inspect["closest_water_body"] = closest_water_body['name'],
                scenario_inspect["closest_water_body_lat"] = closest_water_body['latitude'],
                scenario_inspect["closest_water_body_long"] = closest_water_body['longitude'],
                scenario_inspect["closest_water_body_distance"] = closest_water_body['distance']
                scenario_input["closest_water_body"] = closest_water_body['name']
            else:
                scenario_inspect["closest_water_body"] = None,
                scenario_inspect["closest_water_body_lat"] = None,
                scenario_inspect["closest_water_body_long"] = None,
                scenario_inspect["closest_water_body_distance"] = None
                scenario_input["closest_water_body"] = None  

            if closest_place['distance_to_nearest_port'] > 200 or closest_place["name"] == scenario_input["nearest_port"]:
                scenario_input["distance_to_nearest_port"] = None
                scenario_input["nearest_port"] = None

            if closest_place["distance_to_nearest_harbor"] > 200 or closest_place["name"] == scenario_input["nearest_harbor"]:
                scenario_input["distance_to_nearest_harbor"] = None
                scenario_input["nearest_harbor"] = None

            if self.task_name == "reporting_collision":
                collided_vessel = self.get_random_vessel()
                if random.uniform(0.0, 1.0) >= 0.5:
                    scenario_inspect["collided_vessel"] = collided_vessel
                    scenario_input["collided_vessel_name"] = collided_vessel["Vessel Name"]
                    scenario_input["collided_vessel_type"] = collided_vessel["Vessel Type"]
                else:
                    scenario_inspect["collided_vessel"] = None
                    scenario_input["collided_vessel_name"] = None
                    scenario_input["collided_vessel_type"] = None   
            break       
        
        return scenario_inspect, scenario_input 

if __name__ == "__main__": # For testing purposes only.
    land_shapefile = "./GSHHS_dataset/GSHHS_shp/f/GSHHS_f_L1.shp" # path of the GSHHS data
    geonames_data_path = './all_countries/allCountries.txt' # path of the geonames data
    num_scenarios = 100 # can be any number
    vessel_data_path = './data/ship_data_dk_us.pkl' # path of the ship data
    task_name = "task_name" # Choose from ["reporting_fire", "reporting_flooding", "reporting_collision", "reporting_grounding":
                            #              "reporting_list-danger_of_capsizing", "reporting_sinking", "reporting_attack",
                            #              "reporting_person_overboard", "reporting_drift", "reporting_undesignated_distress"]

    inspect = []
    input = []
    analysis = MaritimeAnalysis(land_shapefile, geonames_data_path, vessel_data_path)
    while True:
        scenario_inspect, scenario_input = analysis.execute([])
        inspect.append(scenario_inspect)
        input.append(scenario_input)
        a = 1