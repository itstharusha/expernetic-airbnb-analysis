import pandas as pd
import gzip

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def profile_file(path, name, nrows=None, compression='infer'):
    print(f"\n{'='*80}")
    print(f"PROFILE: {name}")
    print(f"{'='*80}")
    df = pd.read_csv(path, compression=compression, nrows=nrows, low_memory=False)
    print(f"Shape: {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
    print(f"\nNull counts (top 20):\n{df.isnull().sum().sort_values(ascending=False).head(20)}")
    print(f"\nSample rows:\n{df.head(2)}")
    return df

if __name__ == "__main__":
    listings = profile_file("data/raw/listings.csv.gz", "LISTINGS (detailed)")
    calendar = profile_file("data/raw/calendar.csv.gz", "CALENDAR", nrows=100000)
    reviews = profile_file("data/raw/reviews.csv.gz", "REVIEWS (detailed)", nrows=100000)
    neighbourhoods = profile_file("data/raw/neighbourhoods.csv", "NEIGHBOURHOODS")
    print(f"\n{'='*80}")
    print("INVESTIGATING THE 15293 NULL PATTERN")
    print(f"{'='*80}")
    print(f"Total rows in listings: {len(listings)}")
    print(f"Rows missing host_since: {listings['host_since'].isnull().sum()}")
    
    # Check if these nulls always co-occur
    suspect_cols = ['host_since', 'host_response_time', 'instant_bookable', 'calendar_updated']
    null_mask = listings[suspect_cols].isnull()
    print(f"\nDo all 4 columns go null together?")
    print(null_mask.all(axis=1).sum(), "rows have ALL 4 null")
    print(null_mask.any(axis=1).sum(), "rows have AT LEAST 1 null")
    
    # Check the 'source' column - often explains this
    if 'source' in listings.columns:
        print(f"\nSource value counts overall:\n{listings['source'].value_counts()}")
        print(f"\nSource value counts WHERE host_since is null:\n{listings[listings['host_since'].isnull()]['source'].value_counts()}")
        print(f"\n{'='*80}")
    print("FULL NULL-RATE AUDIT — LISTINGS")
    print(f"{'='*80}")
    null_pct = (listings.isnull().sum() / len(listings) * 100).sort_values(ascending=False)
    print(null_pct.to_string())