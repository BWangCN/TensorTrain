import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load training data
train_data = pd.read_csv('train_data.csv')

# Separate features and labels for training data
X_train = train_data[['Bx0', 'By0', 'Bz0', 'Bx1', 'By1', 'Bz1', 'Bx2', 'By2', 'Bz2', 'Bx3', 'By3', 'Bz3', 'Bx4', 'By4', 'Bz4']]
y_train = train_data['label']

# Load testing data
test_data = pd.read_csv('test_data.csv')

# Separate features and labels for testing data
X_test = test_data[['Bx0', 'By0', 'Bz0', 'Bx1', 'By1', 'Bz1', 'Bx2', 'By2', 'Bz2', 'Bx3', 'By3', 'Bz3', 'Bx4', 'By4', 'Bz4']]
y_test = test_data['label']

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Predict
y_pred = clf.predict(X_test_scaled)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
