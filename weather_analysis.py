# weather_analysis.py
"""
Weather Data Analysis & Visualization
--------------------------------------
Performs exploratory data analysis (EDA) on weatherHistory.csv:
- Data cleaning
- Resampling (daily, monthly)
- 10+ visualizations (time series, boxplots, heatmaps, scatterplots, etc.)
- Outputs summary CSVs and PNG charts
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot, register_matplotlib_converters
from statsmodels.tsa.seasonal import seasonal_decompose

register_matplotlib_converters()
sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv("weatherHistory.csv")
print("Dataset loaded with", len(df), "rows and", len(df.columns), "columns.")

# Cleaning
df.rename(columns=lambda x: x.strip(), inplace=True)
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True, errors='coerce')
df.dropna(subset=['Formatted Date'], inplace=True)
if 'Precip Type' in df.columns:
    df['Precip Type'] = df['Precip Type'].fillna('unknown')
if 'Loud Cover' in df.columns:
    df.drop(columns=['Loud Cover'], inplace=True)

df.set_index('Formatted Date', inplace=True)
df.sort_index(inplace=True)

# Resampling
monthly_avg = df.resample('M').mean(numeric_only=True)
daily = df.resample('D').mean(numeric_only=True)

# --- Visualizations ---
print("Generating visualizations...")

# 1) Monthly Avg Temperature
plt.figure(figsize=(12,5))
plt.plot(monthly_avg.index, monthly_avg['Temperature (C)'])
plt.title("Monthly Avg Temperature")
plt.xlabel("Year"); plt.ylabel("Temp (°C)")
plt.savefig("viz_monthly_avg_temp.png"); plt.close()

# 2) Rolling Mean
plt.figure(figsize=(12,5))
plt.plot(daily.index, daily['Temperature (C)'], alpha=0.5, label="Daily")
plt.plot(daily.index, daily['Temperature (C)'].rolling(30).mean(), color='red', label="30-day Rolling Mean")
plt.legend(); plt.title("Temperature Rolling Mean")
plt.savefig("viz_temp_rolling_mean.png"); plt.close()

# 3) Monthly Boxplot
df['month'] = df.index.month
plt.figure(figsize=(12,6))
sns.boxplot(x='month', y='Temperature (C)', data=df)
plt.title("Monthly Temperature Distribution")
plt.savefig("viz_temp_boxplot.png"); plt.close()

# 4) Heatmap (Year vs Month)
pivot = df['Temperature (C)'].resample('M').mean().to_frame()
pivot['year'] = pivot.index.year; pivot['month'] = pivot.index.month
table = pivot.pivot_table(values='Temperature (C)', index='year', columns='month')
plt.figure(figsize=(12,6))
sns.heatmap(table, annot=True, fmt=".1f", cmap="coolwarm")
plt.title("Monthly Temperature Heatmap")
plt.savefig("viz_temp_heatmap.png"); plt.close()

# 5) Scatter Temp vs Humidity
plt.figure(figsize=(10,6))
sns.regplot(x='Temperature (C)', y='Humidity', data=df.sample(5000), scatter_kws={'alpha':0.3,'s':8})
plt.title("Temperature vs Humidity")
plt.savefig("viz_temp_vs_humidity.png"); plt.close()

# 6) Pairplot
pair_cols = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']
sns.pairplot(df[pair_cols].dropna().sample(1000), diag_kind="kde")
plt.savefig("viz_pairplot.png"); plt.close()

# 7) KDE Distribution
plt.figure(figsize=(10,6))
sns.kdeplot(df['Temperature (C)'], fill=True, label="Temp")
sns.kdeplot(df['Humidity'], fill=True, label="Humidity")
plt.legend(); plt.title("KDE Distributions")
plt.savefig("viz_kde.png"); plt.close()

# 8) Yearly Precipitation
yearly_precip = df.groupby(df.index.year)['Precip Type'].value_counts().unstack().fillna(0)
yearly_precip.plot(kind='bar', stacked=True, figsize=(12,6))
plt.title("Yearly Precipitation Types")
plt.savefig("viz_precip_types.png"); plt.close()

# 9) Autocorrelation
plt.figure(figsize=(10,5))
autocorrelation_plot(monthly_avg['Temperature (C)'])
plt.title("Autocorrelation: Monthly Avg Temp")
plt.savefig("viz_autocorr.png"); plt.close()

# 10) Seasonal Decomposition
decomp = seasonal_decompose(monthly_avg['Temperature (C)'].dropna(), period=12, model="additive")
decomp.plot(); plt.suptitle("Seasonal Decomposition")
plt.savefig("viz_seasonal.png"); plt.close()

# Save summary
monthly_avg.to_csv("monthly_avg.csv")
df.describe().to_csv("describe.csv")

print("✅ Analysis complete. Visualizations & CSVs saved in folder.")
