import pandas as pd

from src.profile_data import profile_file


def test_profile_file(tmp_path, capsys):
    """Test the profile_file function loads and prints stats."""
    raw_file = tmp_path / "sample.csv"
    mock_data = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['A', 'B', None]
    })
    mock_data.to_csv(raw_file, index=False)
    
    df = profile_file(str(raw_file), "TEST_SAMPLE", compression=None)
    
    assert len(df) == 3
    assert 'name' in df.columns
    
    # Check if logging was printed (will check stdout or log depending on implementation)
    # Just ensuring it doesn't crash is a good start.
