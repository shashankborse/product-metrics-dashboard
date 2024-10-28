import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Constants for data simulation
num_users = 1000  # Total number of users
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

# Generate user IDs
user_ids = [f'user_{i+1}' for i in range(num_users)]

# Generate random login timestamps for DAU and MAU metrics
login_data = {
    "user_id": np.random.choice(user_ids, size=5000, replace=True),
    "login_timestamp": [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(5000)]
}

login_df = pd.DataFrame(login_data)

# Generate churn data (last active date for each user)
churn_data = {
    "user_id": user_ids,
    "last_active_date": [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(num_users)]
}
churn_df = pd.DataFrame(churn_data)

# Generate retention data (first active and return dates)
retention_data = {
    "user_id": user_ids,
    "first_active_date": [start_date + timedelta(days=np.random.randint(0, 30)) for _ in range(num_users)],
    "return_dates": [start_date + timedelta(days=np.random.randint(30, 365)) for _ in range(num_users)]
}
retention_df = pd.DataFrame(retention_data)

# Generate purchase data for LTV calculation
purchase_data = {
    "user_id": np.random.choice(user_ids, size=2000, replace=True),
    "purchase_amount": [round(np.random.uniform(20, 200), 2) for _ in range(2000)],
    "subscription_date": [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(2000)]
}
purchase_df = pd.DataFrame(purchase_data)

# Generate NPS data
np.random.seed(42)  # For reproducible results

nps_data = {
    "user_id": np.random.choice(user_ids, size=1000, replace=True),
    "nps_score": np.random.choice(
        range(0, 11), 
        size=1000, 
        p=[0.1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1]  # Increased probability for higher scores
    )
}
nps_df = pd.DataFrame(nps_data)

# Preview the data
print("Login Data Sample:\n", login_df.head())
print("Churn Data Sample:\n", churn_df.head())
print("Retention Data Sample:\n", retention_df.head())
print("Purchase Data Sample:\n", purchase_df.head())
print("NPS Data Sample:\n", nps_df.head())

def calculate_dau(login_df):
    # Group by date only (extract date from timestamp) and count unique user IDs per day
    login_df['login_date'] = login_df['login_timestamp'].dt.date
    dau = login_df.groupby('login_date')['user_id'].nunique().reset_index()
    dau.columns = ['date', 'dau']
    return dau

# Calculate and display DAU
dau_df = calculate_dau(login_df)
print("Daily Active Users (DAU) Sample:\n", dau_df.head())

def calculate_mau(login_df):
    # Ensure login_timestamp is in datetime format
    login_df['login_timestamp'] = pd.to_datetime(login_df['login_timestamp'], errors='coerce')
    
    # Group by year and month and count unique user IDs per month
    login_df['login_month'] = login_df['login_timestamp'].dt.to_period('M')
    mau = login_df.groupby('login_month')['user_id'].nunique().reset_index()
    mau.columns = ['month', 'mau']
    
    # Convert 'month' to string format for plotting compatibility
    mau['month'] = mau['month'].astype(str)
    
    return mau

# Calculate and display MAU
mau_df = calculate_mau(login_df)
print("Monthly Active Users (MAU) Sample:\n", mau_df.head())

def calculate_churn_rate(churn_df, cutoff_days=180):
    # Define the cutoff date for churn based on the last known date in the dataset
    last_date = churn_df['last_active_date'].max()
    churn_cutoff_date = last_date - timedelta(days=cutoff_days)
    
    # Identify users as 'churned' if their last activity date is before the cutoff date
    churned_users = churn_df[churn_df['last_active_date'] < churn_cutoff_date]
    churn_rate = (len(churned_users) / len(churn_df)) * 100
    
    return churn_rate

# Calculate and print Churn Rate
churn_rate = calculate_churn_rate(churn_df)
print(f"Churn Rate: {churn_rate:.2f}%")

def calculate_retention_rate(retention_df, retention_days=60):
    # Define cutoff date for retention based on each user's first active date + retention_days
    retention_df['retention_cutoff_date'] = retention_df['first_active_date'] + timedelta(days=retention_days)
    
    # Calculate retained users: those with a return date on or after the cutoff date
    retained_users = retention_df[retention_df['return_dates'] >= retention_df['retention_cutoff_date']]
    retention_rate = (len(retained_users) / len(retention_df)) * 100
    
    return retention_rate

# Calculate and print Retention Rate
retention_rate = calculate_retention_rate(retention_df)
print(f"Retention Rate: {retention_rate:.2f}%")

def calculate_ltv(purchase_df):
    # Calculate Total Revenue and Average Revenue Per User (ARPU)
    total_revenue = purchase_df['purchase_amount'].sum()
    unique_users = purchase_df['user_id'].nunique()
    arpu = total_revenue / unique_users if unique_users else 0
    
    # Estimate Average User Lifetime in months
    purchase_df['subscription_month'] = purchase_df['subscription_date'].dt.to_period('M')
    user_lifetimes = purchase_df.groupby('user_id')['subscription_month'].nunique()
    avg_user_lifetime = user_lifetimes.mean() if not user_lifetimes.empty else 0
    
    # Calculate LTV
    ltv = arpu * avg_user_lifetime
    return ltv

# Calculate and print LTV
ltv = calculate_ltv(purchase_df)
print(f"Customer Lifetime Value (LTV): ${ltv:.2f}")

def calculate_cac(total_marketing_cost, new_customers_count):
    # Avoid division by zero
    if new_customers_count == 0:
        return 0
    # Calculate CAC
    cac = total_marketing_cost / new_customers_count
    return cac

# Simulate total marketing cost and new customers count
total_marketing_cost = 50000  # Example marketing spend in dollars
new_customers_count = len(purchase_df['user_id'].unique())  # Assuming each unique user is a new customer

# Calculate and print CAC
cac = calculate_cac(total_marketing_cost, new_customers_count)
print(f"Customer Acquisition Cost (CAC): ${cac:.2f}")

# Simulate feature usage data for testing
feature_usage_data = {
    "user_id": np.random.choice(user_ids, size=int(num_users * 0.5), replace=True),  # 50% of users
    "feature_used": [True] * int(num_users * 0.5),
    "usage_timestamp": [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(int(num_users * 0.5))]
}
feature_usage_df = pd.DataFrame(feature_usage_data)
feature_usage_df['usage_date'] = feature_usage_df['usage_timestamp'].dt.date

def calculate_feature_adoption_rate(login_df, feature_usage_df):
    # Calculate the total number of active users (using DAU calculation approach)
    active_users = login_df['user_id'].nunique()
    
    # Calculate the number of users who used the feature
    feature_users = feature_usage_df['user_id'].nunique()
    
    # Calculate Feature Adoption Rate
    feature_adoption_rate = (feature_users / active_users) * 100 if active_users else 0
    return feature_adoption_rate

# Calculate and print Feature Adoption Rate
feature_adoption_rate = calculate_feature_adoption_rate(login_df, feature_usage_df)
print(f"Feature Adoption Rate: {feature_adoption_rate:.2f}%")

def calculate_nps(nps_df):
    # Classify users based on their NPS score
    promoters = nps_df[nps_df['nps_score'] >= 9].shape[0]
    passives = nps_df[(nps_df['nps_score'] >= 7) & (nps_df['nps_score'] <= 8)].shape[0]
    detractors = nps_df[nps_df['nps_score'] <= 6].shape[0]
    
    # Calculate total responses
    total_responses = promoters + passives + detractors
    
    # Avoid division by zero
    if total_responses == 0:
        return 0
    
    # Calculate NPS
    nps = ((promoters - detractors) / total_responses) * 100
    return nps

# Calculate and print NPS
nps_score = calculate_nps(nps_df)
print(f"Net Promoter Score (NPS): {nps_score:.2f}")

# Export each DataFrame as a CSV file in the data folder
login_df.to_csv("data/login_data.csv", index=False)
churn_df.to_csv("data/churn_data.csv", index=False)
retention_df.to_csv("data/retention_data.csv", index=False)
purchase_df.to_csv("data/purchase_data.csv", index=False)
nps_df.to_csv("data/nps_data.csv", index=False)
feature_usage_df.to_csv("data/feature_usage.csv", index=False)

print("DataFrames have been successfully saved as CSV files.")