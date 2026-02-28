import pandas as pd
import numpy as np
import random

# Hackathon script to generate some dummy IoT sensor data for our cold chain
# Didn't have time to scrape the real API so making a realistic synthetic one

np.random.seed(42)

num_records = 850
categories = ['Dairy', 'Meat', 'Vegetables', 'Fruits', 'Seafood']
origins = ['Warehouse A', 'Port B', 'Farm C', 'Distribution Node 1']

data = []
for i in range(num_records):
    cat = random.choice(categories)
    origin = random.choice(origins)
    transit_hrs = np.random.normal(loc=24, scale=12)
    
    # baseline temps
    if cat in ['Meat', 'Seafood']:
        temp = np.random.normal(loc=2.0, scale=3.0)
    else:
        temp = np.random.normal(loc=6.0, scale=4.0)
        
    humidity = np.random.uniform(40, 95)
    
    # logic for spoilage
    spoiled = 0
    if temp > 8.0 and transit_hrs > 12:
        spoiled = 1
    elif temp > 4.0 and transit_hrs > 36:
        spoiled = 1
    elif cat in ['Seafood', 'Meat'] and temp > 5.0:
        spoiled = 1
        
    # introduce some noise
    if random.random() < 0.05:
        spoiled = 1 if spoiled == 0 else 0
        
    data.append([f"SHP_{i+1000}", origin, transit_hrs, temp, humidity, cat, spoiled])

df = pd.DataFrame(data, columns=['shipment_id', 'origin', 'transit_hours', 'avg_temp_c', 'humidity_pct', 'category', 'is_spoiled'])

# Let's add some missing values so we can show off our data cleaning skills to the judges lol
missing_indices = np.random.choice(df.index, size=25, replace=False)
df.loc[missing_indices, 'avg_temp_c'] = np.nan

df.to_csv('cold_chain_data.csv', index=False)
print("Saved dataset to cold_chain_data.csv!")
