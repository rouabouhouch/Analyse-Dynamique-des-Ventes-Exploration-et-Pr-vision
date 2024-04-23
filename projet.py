import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the synthetic dataset
sales_data = pd.read_csv('synthetic_sales_data.csv')

# 1. Exploratory Data Analysis (EDA)

# Exclude non-numeric columns
numeric_columns = sales_data.select_dtypes(include=[np.number])

# Calculate correlation matrix
correlation_matrix = numeric_columns.corr()

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Plot distribution of sales by product category and gender
plt.figure(figsize=(12, 6))
sns.boxplot(x='Product Category', y='Unit Price', hue='Gender', data=sales_data)
plt.title('Sales Distribution by Product Category and Gender')
plt.xlabel('Product Category')
plt.ylabel('Unit Price')
plt.show()

# 2. Time Series Analysis

# Monthly sales trend
monthly_sales = sales_data.groupby(pd.Grouper(key='Date', freq='M')).sum()['Unit Price']
plt.figure(figsize=(12, 6))
monthly_sales.plot(marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()

# 3. Customer Segmentation

# Select features for segmentation
segmentation_data = sales_data[['Age', 'Unit Price', 'Quantity']]

# Standardize the data
scaler = StandardScaler()
segmentation_data_scaled = scaler.fit_transform(segmentation_data)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
segmentation_data_pca = pca.fit_transform(segmentation_data_scaled)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
segmentation_data['Cluster'] = kmeans.fit_predict(segmentation_data_pca)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=segmentation_data_pca[:,0], y=segmentation_data_pca[:,1], hue=segmentation_data['Cluster'], palette='viridis')
plt.title('Customer Segmentation')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# 4. Predictive Modeling

# Prepare data for prediction
X = sales_data[['Age', 'Gender', 'Product Category', 'Quantity']]
X = pd.get_dummies(X, columns=['Gender', 'Product Category'], drop_first=True)
y = sales_data['Unit Price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Feature importance
feature_importance = pd.Series(rf_regressor.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.show()
