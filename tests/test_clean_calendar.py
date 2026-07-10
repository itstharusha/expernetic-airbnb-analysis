
import pandas as pd

from src.clean_calendar import clean_calendar


def test_clean_calendar_parsing(tmp_path):
    """Test that the calendar cleaning script parses data correctly."""
    # Create mock raw data
    raw_file = tmp_path / "raw_calendar.csv.gz"
    
    mock_data = pd.DataFrame({
        'listing_id': [1, 1, 2],
        'date': ['2024-01-01', 'invalid_date', '2024-01-03'],
        'available': ['t', 'f', 'f'],
        'price': ['$100.00', '$100.00', '$200.00'],
        'adjusted_price': ['$90.00', '$90.00', '$190.00'],
        'minimum_nights': [2, -1, 3], # Includes a negative value
        'maximum_nights': [30, 30, 30]
    })
    mock_data.to_csv(raw_file, compression='gzip', index=False)
    
    out_file = tmp_path / "clean_calendar.parquet"
    
    df = clean_calendar(str(raw_file), str(out_file))
    
    assert len(df) == 3
    # Check boolean mapping
    assert df['available'].iloc[0] == True
    assert df['available'].iloc[1] == False
    
    # Check date parsing (invalid coerced to NaT)
    assert not pd.isna(df['date'].iloc[0])
    assert pd.isna(df['date'].iloc[1])
