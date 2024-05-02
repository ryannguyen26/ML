import pandas as pd                 # Import pandas for data manipulation
import matplotlib.pyplot as plt     # Import matplotlib for plotting
import seaborn as sns               # Import seaborn for enhanced visualization features

# Load the dataset
data_path = 'C:/Users/RyanPC/OneDrive/Desktop/Stockton/Spring 2024/Machine Learning/assignment 3/Advertising.csv'

data = pd.read_csv(data_path)

### Exploratory Data Analysis

# Display the first few rows of the dataset
data.head(), data.columns

# Creating histograms for each feature including the target variable to identify any outliers
plt.figure(figsize=(12, 8))        # Set the size of the figure for the plots
for i, column in enumerate(data.columns[1:], 1):  # Loop over each column, starting from the second column
    plt.subplot(2, 2, i)           # Create a subplot for each histogram
    sns.histplot(data[column], kde=True)  # Generate a histogram with a kernel density estimate
    plt.title(f'Histogram of {column}')  # Set the title for each histogram
plt.tight_layout()                 # Adjust subplots to fit into the figure area
plt.show()                         # Display the histograms

# Creating boxplots for each feature including the target variable
plt.figure(figsize=(12, 8))        # Set the size of the figure for the plots
for i, column in enumerate(data.columns[1:], 1):  # Loop over each column, starting from the second column
    plt.subplot(2, 2, i)           # Create a subplot for each boxplot
    sns.boxplot(y=data[column])    # Generate a boxplot
    plt.title(f'Boxplot of {column}')  # Set the title for each boxplot
plt.tight_layout()                 # Adjust subplots to fit into the figure area
plt.show()                         # Display the boxplots

# Newspapers appears to have two outliers, let's identify them

# Calculate the first (Q1) and third (Q3) quartiles of the 'Newspaper' column
Q1 = data['Newspaper'].quantile(0.25)  # Calculate the 25th percentile
Q3 = data['Newspaper'].quantile(0.75)  # Calculate the 75th percentile

# Calculate the Interquartile Range (IQR) by subtracting Q1 from Q3
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR  # Outliers are below this
upper_bound = Q3 + 1.5 * IQR  # Outliers are above this

# Filter the data to find points that fall outside of the lower and upper bounds
outliers = data[(data['Newspaper'] < lower_bound) | (data['Newspaper'] > upper_bound)]

# Print the outliers
print("Outliers in the Newspaper data:")
print(outliers)
print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

#Build the Linear Regression Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Prepare the features and target variable
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the sales on the testing data
y_pred = model.predict(X_test)

# Calculate the coefficients, intercept, Mean Squared Error (MSE), and R-squared value
coefficients = model.coef_
intercept = model.intercept_
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

(coefficients, intercept, mse, r2)

# Print the results
print("Coefficients:", coefficients)
print("Intercept:", intercept)
print("Mean Squared Error (MSE):", mse)
print("R-squared Value:", r2)
