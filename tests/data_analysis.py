import pandas as pd
import numpy as np

train = pd.read_csv('data/processed/train.csv')
train['date_time'] = pd.to_datetime(train['date_time'])
train['hour'] = train['date_time'].dt.hour
train['day_of_week'] = train['date_time'].dt.dayofweek
train['is_weekend'] = (train['day_of_week'] >= 5).astype(int)
train['is_holiday'] = (train['holiday'] != 'None').astype(int)

print('=== Dataset Stats ===')
print(f'Total rows: {len(train)}')
tv = train['traffic_volume']
print(f'Traffic: min={tv.min():.0f}, max={tv.max():.0f}, mean={tv.mean():.0f}')

print()
print('=== By Hour (weekday) ===')
wkday = train[train['is_weekend']==0]
print(wkday.groupby('hour')['traffic_volume'].mean().round(0).to_string())

print()
print('=== Holiday vs Non-Holiday ===')
print(train.groupby('is_holiday')['traffic_volume'].describe().round(0))

if 'weather_main' in train.columns:
    print()
    print('=== By Weather ===')
    print(train.groupby('weather_main')['traffic_volume'].mean().sort_values(ascending=False).round(0))

    print()
    print('=== Rush hour (7-9am weekday) Stats ===')
    rush = wkday[wkday['hour'].isin([7,8,9])]
    print(f'Rush hour traffic mean: {rush.traffic_volume.mean():.0f}')
    snow_rush = rush[rush['weather_main'] == 'Snow']['traffic_volume']
    if len(snow_rush) > 0:
        print(f'Rush hour Snow days: {snow_rush.mean():.0f}')
    else:
        print('No snow rush data')
