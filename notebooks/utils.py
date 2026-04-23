import pandas as pd
def custom_feature_engineering(df):
    df = df.copy()
    
    # Binning continuous variables
    df['Age Group'] = pd.cut(df['Age'], bins=[0, 18, 30, 50, 100], labels=[0, 1, 2, 3]).astype(float)
    df['Flight Distance Group'] = pd.cut(df['Flight Distance'], bins=[0, 500, 1500, 3000, 5000], labels=[0, 1, 2, 3]).astype(float)
    
    # Total Delay (Handling NaNs is crucial for Logistic Regression)
    df['Total Delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes'].fillna(0)
    
    # Average Service Score
    service_cols = ['On-board service', 'Checkin service', 'Inflight service', 'Inflight wifi service']
    df['Service Score Average'] = df[service_cols].mean(axis=1)
    
    # Drop columns that are no longer needed for modeling
    # Note: We keep Gender, Customer Type, etc. here because they will be encoded in the next step
    cols_to_drop = [
        'id', 'Unnamed: 0', 'Age', 'Flight Distance', 
        'Departure Delay in Minutes', 'Arrival Delay in Minutes',
        'On-board service', 'Checkin service', 'Inflight service', 'Inflight wifi service',
        'Departure/Arrival time convenient', 'Gate location'
    ]
    return df.drop(columns=cols_to_drop, errors='ignore')