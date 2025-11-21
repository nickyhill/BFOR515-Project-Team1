import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import randint
import matplotlib.pyplot as plt

#load dataset
df = pd.read_csv('cybersecurity_intrusion_data.csv')

#identify columns we are working with
required_cols = ['network_packet_size', 'session_duration']

df_clean = df.dropna(subset=required_cols).copy()

#print the number of columms in the dataframe
print(df_clean.describe())
print(df_clean.value_counts())

#create variables called network_packet_size and session_duration to be used for x and y values in the scatterplot
network_packet_size = df_clean['network_packet_size']
session_duration = df_clean['session_duration']

#create a scatterplot that compares network packet size to session duration
duration_plot = sns.scatterplot(x=network_packet_size, y=session_duration)
plt.title('Network Packet Size vs Session Duration')
plt.xlabel('Network Packet Size')
plt.ylabel('Session Duration')
plt.show()

#create a pairplot that creates a scatter of all of the columns so that we can identify what variables to hone in on
#required_cols = ['session_id', 'network_packet_size', 'protocol_type', 'login_attempts', 'session_duration', 'encryption_used', 'ip_reputation_score', 'failed_logins', 'browser_type', 'unusual_time_access', 'attack_detected']
#scat = sns.pairplot(df_clean[required_cols], diag_kind='kde', kind='kde')
#plt.show()


############### Models ####################

# Set column variables
predictor_cols = ['login_attempts', 'ip_reputation_score']
cat_cols = ['protocol_type', 'encryption_used', 'browser_type']
target = 'attack_detected'


# Drop Session ID
df = df.drop(columns=['session_id'])


X = df[predictor_cols]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# Train initial Random Forest model
model2 = RandomForestClassifier()
model2.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y2_pred = model2.predict(X_test)


## Evaluate the Logistic model
print("###### Logistic Regression Results ######", end="\n\n")
print("Coefficients:", dict(zip(X.columns, model.coef_[0])))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
plt.figure(1, figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

print("Classification Report:\n", classification_report(y_test, y_pred))
print("\n\n\n\n")

## Evaluate the Random Forest model (Hyperparamter tuning was used from
## https://www.datacamp.com/tutorial/random-forests-classifier-python)
print("###### Random Forest Results ######", end="\n\n")
cm = confusion_matrix(y_test, y2_pred)
print("Confusion Matrix:\n", cm)
plt.figure(2, figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

print("Classification Report:\n", classification_report(y_test, y2_pred))
print("\n\n")


# Hyperparameter tuning
param_dist = {
  'n_estimators': randint(100, 500),
  'max_depth': randint(3, 15),
  'min_samples_split': randint(2, 10),
  'min_samples_leaf': randint(1, 5)
}

# Train Random Forest model
model3 = RandomForestClassifier(random_state=42, n_jobs=-1)
# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(
  model3, param_distributions=param_dist,
  n_iter=10, cv=5, scoring='accuracy',
  n_jobs=-1, random_state=42)

# Create a variable for the best model
best_model3 = rand_search.fit(X_train, y_train)

# Make predictions
y3_pred = best_model3.predict(X_test)

print("###### Tuned Random Forest Results ######\n")
# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)
cm = confusion_matrix(y_test, y3_pred)
print("Confusion Matrix:\n", cm)
plt.figure(3, figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Tuned Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
print("Classification Report:\n", classification_report(y_test, y3_pred))
print("\n\n\n\n")
