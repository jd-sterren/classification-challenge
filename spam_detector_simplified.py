import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Import the data
data = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv")
data.head()

# Create the labels set `y` and features DataFrame `X`
X = data.copy()
X = X.drop(columns="spam")
y = data['spam']

# Split the data into X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create the StandardScaler instance and fit it
X_scaler = StandardScaler().fit(X_train)

# Scale the training data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# LOGISTIC REGRESSION MODEL ---------------------------------
# Train a Logistic Regression model and print the model score
logistic_regression_model = LogisticRegression(random_state=1).fit(X_train_scaled, y_train)

# Print the model score
print(f'Logistic Regression Testing Score: {logistic_regression_model.score(X_test_scaled, y_test)}')

# Make and save testing predictions with the saved logistic regression model using the test data
test_predictions = logistic_regression_model.predict(X_test_scaled)

# Calculate the accuracy score by evaluating `y_test` vs. `testing_predictions`.
accuracy_score(y_test, test_predictions)

# RANDOM FOREST CLASSIFIER ---------------------------------------
# Train a Random Forest Classifier model and print the model score
# n_estimators improve potential accuracy but the higher the number, the longer
# it'll take to fit so think of data size over accuracy.
rfc = RandomForestClassifier(random_state=1, n_estimators=500).fit(X_train_scaled, y_train)

# Make and save testing predictions with the saved logistic regression model using the test data
forest_prediction = rfc.predict(X_test_scaled)


# Calculate the accuracy score by evaluating `y_test` vs. `testing_predictions`.
print(f'Random Forest Classifier Testing Score: {accuracy_score(y_test, forest_prediction)}')