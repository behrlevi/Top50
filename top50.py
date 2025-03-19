import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from google.colab import files

# CSV beolvasása
df = pd.read_csv('Top50musicalityglobal.csv')

# Display the first few rows to understand the structure
print("Dataset preview:")
display(df.head())

# Check basic information
print("\nDataset info:")
display(df.info())

# Check summary statistics
print("\nSummary statistics:")
display(df.describe())

# Check for missing values
print("\nMissing values:")
display(df.isnull().sum())

# Create a 'Position' column if it doesn't exist
# Since we don't have an explicit position column, we'll use the 'Unnamed: 0' column 
# as it might represent the original row number, or we'll create a position based on popularity
if 'Position' not in df.columns:
    if 'Unnamed: 0' in df.columns:
        df['Position'] = df['Unnamed: 0'] + 1  # Adding 1 to make it 1-based indexing
    else:
        # If we don't have any position info, we'll create one based on popularity
        # Higher popularity means better position (lower number)
        df['Position'] = df['Popularity'].rank(ascending=False)

# Create a copy of the dataframe with just the numerical features
numerical_df = df.select_dtypes(include=['float64', 'int64'])
numerical_df = numerical_df.drop(['Unnamed: 0'], axis=1, errors='ignore')  # Drop index if it exists

# 1. Distribution of each numerical feature
print("\nAnalyzing distributions of numerical features:")
features = numerical_df.columns.tolist()
features = [f for f in features if f != 'Position']  # Remove position from features to analyze

# Create histograms for each feature
plt.figure(figsize=(20, 15))
for i, feature in enumerate(features):
    plt.subplot(4, 5, i+1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.show()

# 2. Box plots to identify outliers
plt.figure(figsize=(20, 10))
sns.boxplot(data=df[features])
plt.title('Box Plots of Features')
plt.xticks(rotation=90)
plt.savefig('feature_boxplots.png')
plt.show()

# 3. Correlation analysis
print("\nCorrelation with position:")
for feature in features:
    correlation = df['Position'].corr(df[feature])
    print(f"Correlation between Position and {feature}: {correlation:.4f}")

# Correlation heatmap
plt.figure(figsize=(14, 12))
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

# 4. Analyze top vs bottom songs
print("\nComparing top 50 vs bottom 50 songs:")
# Sort by position
df_sorted = df.sort_values('Position')
top_50 = df_sorted.head(50)
bottom_50 = df_sorted.tail(50)

for feature in features:
    print(f"{feature} - Top 50 avg: {top_50[feature].mean():.2f}, Bottom 50 avg: {bottom_50[feature].mean():.2f}")

# 5. Scatter plots of Position vs each feature
plt.figure(figsize=(20, 15))
for i, feature in enumerate(features):
    plt.subplot(4, 5, i+1)
    sns.scatterplot(x='Position', y=feature, data=df)
    plt.title(f'Position vs {feature}')
plt.tight_layout()
plt.savefig('position_vs_features.png')
plt.show()

# 6. Advanced visualization with Plotly
# Create an interactive bubble chart with Plotly
fig = px.scatter(df, x='Energy', y='Danceability', size='Popularity', 
                color='Position', hover_name='Track Name',
                size_max=15, color_continuous_scale='Viridis')
fig.update_layout(title='Energy vs Danceability by Popularity')
fig.write_html('energy_danceability_popularity.html')
fig.show()

# 7. Feature importance using Random Forest
X = df[features]
y = df['Position']

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Plot feature importance
importance = model.feature_importances_
indices = np.argsort(importance)

plt.figure(figsize=(12, 8))
plt.barh(range(len(indices)), importance[indices])
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Position in Ranking')
plt.savefig('feature_importance.png')
plt.show()

# 8. Country analysis
if 'Country' in df.columns:
    plt.figure(figsize=(14, 8))
    country_counts = df['Country'].value_counts()
    sns.barplot(x=country_counts.index[:15], y=country_counts.values[:15])
    plt.title('Count of Songs by Top 15 Countries')
    plt.xticks(rotation=90)
    plt.savefig('songs_by_country.png')
    plt.show()
    
    # Average position by country
    country_position = df.groupby('Country')['Position'].mean().sort_values()
    plt.figure(figsize=(14, 8))
    sns.barplot(x=country_position.index[:15], y=country_position.values[:15])
    plt.title('Average Position by Top 15 Countries')
    plt.xticks(rotation=90)
    plt.ylabel('Average Position (lower is better)')
    plt.savefig('position_by_country.png')
    plt.show()

# 9. Popularity vs other features
popular_features = ['Danceability', 'Energy', 'Acousticness', 'Loudness', 'Speechiness', 'Tempo', 'Positiveness']
plt.figure(figsize=(20, 15))
for i, feature in enumerate(popular_features):
    if feature in df.columns:
        plt.subplot(3, 3, i+1)
        sns.scatterplot(x='Popularity', y=feature, data=df)
        plt.title(f'Popularity vs {feature}')
        
        # Add trend line
        z = np.polyfit(df['Popularity'], df[feature], 1)
        p = np.poly1d(z)
        plt.plot(df['Popularity'], p(df['Popularity']), "r--")
plt.tight_layout()
plt.savefig('popularity_vs_features.png')
plt.show()

# 10. Analyze distribution of key musical attributes
key_attributes = ['Key', 'Mode', 'TSignature']
for attr in key_attributes:
    if attr in df.columns:
        plt.figure(figsize=(10, 6))
        counts = df[attr].value_counts().sort_index()
        sns.barplot(x=counts.index, y=counts.values)
        plt.title(f'Distribution of {attr}')
        plt.savefig(f'{attr}_distribution.png')
        plt.show()

# 11. Analyze artist diversity
if 'Artist Name' in df.columns:
    artist_counts = df['Artist Name'].value_counts()
    top_artists = artist_counts.head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_artists.values, y=top_artists.index)
    plt.title('Top 15 Artists by Number of Songs')
    plt.savefig('top_artists.png')
    plt.show()
    
    # Average popularity by top artists
    top_artist_names = top_artists.index.tolist()
    top_artists_df = df[df['Artist Name'].isin(top_artist_names)]
    artist_popularity = top_artists_df.groupby('Artist Name')['Popularity'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=artist_popularity.values, y=artist_popularity.index)
    plt.title('Average Popularity of Top 15 Artists')
    plt.savefig('artist_popularity.png')
    plt.show()

# 12. Summary of findings
print("\nSummary of Key Findings:")
print("1. Features with strongest correlation to position:")
correlations = [(feature, abs(df['Position'].corr(df[feature]))) for feature in features]
correlations.sort(key=lambda x: x[1], reverse=True)
for feature, corr in correlations[:5]:
    print(f"   - {feature}: {corr:.4f}")

print("\n2. Most important features according to Random Forest:")
top_features = [(features[i], importance[i]) for i in indices[-5:]]
for feature, imp in reversed(top_features):
    print(f"   - {feature}: {imp:.4f}")

print("\n3. Key differences between top and bottom songs:")
for feature in features:
    diff = top_50[feature].mean() - bottom_50[feature].mean()
    if abs(diff) > 0.1:  # Only show meaningful differences
        print(f"   - {feature}: Difference of {diff:.2f}")
