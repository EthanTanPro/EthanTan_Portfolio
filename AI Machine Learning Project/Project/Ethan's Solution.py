#Step 1: Import dataset
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())


#Step 2a: Remove invalid data

# Use .apply() with a lambda function to filter out non-integer ages
df = df[df['Age'].apply(lambda x: x.is_integer())]

# Display the result to verify
df.head()


#Step 2b: Remove null data
df = df.dropna()

df.replace('Unknown', pd.NA, inplace=True)
df = df.dropna()


#Step 2c: Remove duplicates
df = df.drop_duplicates(subset='Name', keep='first')

df.head()

df = df[df['Age'].apply(lambda x: x.is_integer())]

df.head()


from sklearn.preprocessing import LabelEncoder

# Step 3: Convert categorical values to numeric
le = LabelEncoder()

# Assuming 'Sex' and 'Embarked' are categorical features, convert them
df['Sex'] = le.fit_transform(df['Sex'])
if 'Embarked' in df.columns:
    df['Embarked'] = le.fit_transform(df['Embarked'])

# For multiple categorical columns, use pd.get_dummies()
# df = pd.get_dummies(df, drop_first=True)

df.head()


data = df.drop(columns=['Name'])



# Step 4: Choose features and target
# 'Survived' is the target variable

features = df.drop('Survived', axis=1)  # Drop the target column
target = df['Survived']  # Target variable

features.head(), target.head()

le = LabelEncoder()
features['Pclass'] = le.fit_transform(features['Pclass'])

ordinal_map = {'1st': 1, '2nd': 2, '3rd': 3}
features['Pclass'] = features['Pclass'].map(ordinal_map)



from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Step 5: Define models
model1 = RandomForestClassifier(random_state=42)
model2 = LogisticRegression(random_state=42)

from sklearn.preprocessing import StandardScaler


# Step 6: Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Convert the scaled features back to a DataFrame for easier manipulation later
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

features_scaled_df.head()

from sklearn.model_selection import train_test_split


# Step 7: Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(features_scaled_df, target, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Verify shapes
X_train.shape, X_val.shape, X_test.shape


# Step 8: Train Random Forest
model1.fit(X_train, y_train)

# Step 8: Train Logistic Regression
model2.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 9: Evaluate Random Forest
y_pred1 = model1.predict(X_val)
print("Random Forest Model Evaluation:")
print(confusion_matrix(y_val, y_pred1))
print(classification_report(y_val, y_pred1))
print("Accuracy:", accuracy_score(y_val, y_pred1))

# Step 9: Evaluate Logistic Regression
y_pred2 = model2.predict(X_val)
print("\nLogistic Regression Model Evaluation:")
print(confusion_matrix(y_val, y_pred2))
print(classification_report(y_val, y_pred2))
print("Accuracy:", accuracy_score(y_val, y_pred2))


# Step 10: Determine the most suitable model based on accuracy
best_model = "Random Forest" if accuracy_score(y_val, y_pred1) > accuracy_score(y_val, y_pred2) else "Logistic Regression"
print(f"The most suitable model is: {best_model}")