import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data files
login_df = pd.read_csv("data/login_data.csv")
churn_df = pd.read_csv("data/churn_data.csv")
retention_df = pd.read_csv("data/retention_data.csv")
purchase_df = pd.read_csv("data/purchase_data.csv")
nps_df = pd.read_csv("data/nps_data.csv")
feature_usage_df = pd.read_csv("data/feature_usage.csv")

# Define functions for metrics
def calculate_dau(login_df):
    login_df['login_date'] = pd.to_datetime(login_df['login_timestamp']).dt.date
    dau = login_df.groupby('login_date')['user_id'].nunique().reset_index()
    dau.columns = ['date', 'dau']
    return dau

def calculate_mau(login_df):
    login_df['login_timestamp'] = pd.to_datetime(login_df['login_timestamp'], errors='coerce')
    login_df['login_month'] = login_df['login_timestamp'].dt.to_period('M')
    mau = login_df.groupby('login_month')['user_id'].nunique().reset_index()
    mau.columns = ['month', 'mau']
    mau['month'] = mau['month'].dt.to_timestamp()
    return mau

def calculate_churn_rate(churn_df, cutoff_days=90):
    last_date = pd.to_datetime(churn_df['last_active_date']).max()
    churn_cutoff_date = last_date - pd.Timedelta(days=cutoff_days)
    churned_users = churn_df[pd.to_datetime(churn_df['last_active_date']) < churn_cutoff_date]
    churn_rate = (len(churned_users) / len(churn_df)) * 100
    return churn_rate

def calculate_retention_rate(retention_df, retention_days=30):
    retention_df['retention_cutoff_date'] = pd.to_datetime(retention_df['first_active_date']) + pd.Timedelta(days=retention_days)
    retained_users = retention_df[pd.to_datetime(retention_df['return_dates']) >= retention_df['retention_cutoff_date']]
    retention_rate = (len(retained_users) / len(retention_df)) * 100
    return retention_rate

def calculate_ltv(purchase_df):
    total_revenue = purchase_df['purchase_amount'].sum()
    unique_users = purchase_df['user_id'].nunique()
    arpu = total_revenue / unique_users if unique_users else 0
    purchase_df['subscription_month'] = pd.to_datetime(purchase_df['subscription_date']).dt.to_period('M')
    user_lifetimes = purchase_df.groupby('user_id')['subscription_month'].nunique()
    avg_user_lifetime = user_lifetimes.mean() if not user_lifetimes.empty else 0
    ltv = arpu * avg_user_lifetime
    return ltv

def calculate_cac(total_marketing_cost, new_customers_count):
    return total_marketing_cost / new_customers_count if new_customers_count > 0 else 0

def calculate_feature_adoption_rate(login_df, feature_usage_df):
    active_users = login_df['user_id'].nunique()
    feature_users = feature_usage_df['user_id'].nunique()
    feature_adoption_rate = (feature_users / active_users) * 100 if active_users else 0
    return feature_adoption_rate

def calculate_nps(nps_df):
    promoters = nps_df[nps_df['nps_score'] >= 9].shape[0]
    detractors = nps_df[nps_df['nps_score'] <= 6].shape[0]
    total_responses = nps_df.shape[0]
    nps = ((promoters - detractors) / total_responses) * 100 if total_responses > 0 else 0
    return nps

# Streamlit Dashboard Layout
st.title("Product Metrics Dashboard")
st.write("This dashboard presents key product metrics, including DAU, MAU, churn rate, retention rate, LTV, CAC, feature adoption, and NPS.")

# Sidebar: Date Range Selection for DAU and MAU
dau_df = calculate_dau(login_df)
mau_df = calculate_mau(login_df)

st.sidebar.subheader("Date Range Filter")
date_min = pd.to_datetime(dau_df['date']).min()
date_max = pd.to_datetime(dau_df['date']).max()
start_date, end_date = st.sidebar.date_input("Select Date Range:", [date_min, date_max])

# Convert start_date and end_date to datetime64[ns]
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter DAU and MAU data based on selected date range
dau_filtered = dau_df[(pd.to_datetime(dau_df['date']) >= start_date) & (pd.to_datetime(dau_df['date']) <= end_date)]
mau_filtered = mau_df[(pd.to_datetime(mau_df['month']) >= start_date) & (pd.to_datetime(mau_df['month']) <= end_date)]

# Sidebar: Adjustable Parameters for Churn and Retention
st.sidebar.subheader("Adjustable Parameters")
cutoff_days = st.sidebar.slider("Churn Cutoff (Days)", min_value=30, max_value=180, value=90, step=10)
retention_days = st.sidebar.slider("Retention Period (Days)", min_value=7, max_value=90, value=30, step=7)

# Recalculate churn and retention with updated parameters
churn_rate = calculate_churn_rate(churn_df, cutoff_days)
retention_rate = calculate_retention_rate(retention_df, retention_days)

# DAU Plot with Date Filter
st.subheader("Daily Active Users (DAU)")
fig, ax = plt.subplots()
ax.plot(dau_filtered['date'], dau_filtered['dau'], linestyle='solid', marker=None)
ax.set_xlabel("Date")
ax.set_ylabel("DAU")
plt.xticks(rotation=45)
st.pyplot(fig)

# MAU Plot with Date Filter
st.subheader("Monthly Active Users (MAU)")
fig, ax = plt.subplots()
ax.plot_date(mau_filtered['month'], mau_filtered['mau'], linestyle='solid', marker=None)
ax.set_xlabel("Month")
ax.set_ylabel("MAU")
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
st.pyplot(fig)

# Display Updated Churn Rate and Retention Rate
st.subheader("Churn and Retention Rates")
st.metric(label="Churn Rate", value=f"{churn_rate:.2f}%")
st.metric(label="Retention Rate", value=f"{retention_rate:.2f}%")

# Display LTV and CAC
st.subheader("Customer Lifetime Value (LTV) and Customer Acquisition Cost (CAC)")
ltv = calculate_ltv(purchase_df)
cac = calculate_cac(50000, len(purchase_df['user_id'].unique()))
st.metric(label="Customer Lifetime Value (LTV)", value=f"${ltv:.2f}")
st.metric(label="Customer Acquisition Cost (CAC)", value=f"${cac:.2f}")

# Display Feature Adoption Rate and NPS
st.subheader("Feature Adoption Rate and Net Promoter Score (NPS)")
feature_adoption_rate = calculate_feature_adoption_rate(login_df, feature_usage_df)
st.metric(label="Feature Adoption Rate", value=f"{feature_adoption_rate:.2f}%")
nps_score = calculate_nps(nps_df)
st.metric(label="Net Promoter Score (NPS)", value=f"{nps_score:.2f}")