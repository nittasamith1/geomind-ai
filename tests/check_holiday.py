import pandas as pd

train = pd.read_csv('data/processed/train.csv')
val = pd.read_csv('data/processed/val.csv')
test = pd.read_csv('data/processed/test.csv')

for name, df in [('train', train), ('val', val), ('test', test)]:
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_holiday'] = (df['holiday'] != 'None').astype(int)
    print(f"=== {name} ===")
    print(f"  Shape: {df.shape}")
    print(f"  Holiday counts: {df['is_holiday'].value_counts().to_dict()}")
    print(f"  Unique holidays: {df['holiday'].unique()[:10]}")
    print(f"  Traffic by is_holiday:")
    print(f"    Non-holiday: {df[df['is_holiday']==0]['traffic_volume'].mean():.0f}")
    print(f"    Holiday: {df[df['is_holiday']==1]['traffic_volume'].mean():.0f}")
    print(f"  Traffic weekday rush (7-9am, non-holiday): {df[(df['hour'].isin([7,8,9])) & (df['is_weekend']==0) & (df['is_holiday']==0)]['traffic_volume'].mean():.0f}")
    print(f"  Traffic weekday rush (7-9am, holiday): {df[(df['hour'].isin([7,8,9])) & (df['is_weekend']==0) & (df['is_holiday']==1)]['traffic_volume'].mean():.0f}")
    print()
