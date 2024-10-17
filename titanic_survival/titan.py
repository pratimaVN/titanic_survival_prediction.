import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
titanic_dataset = pd.read_csv('E:/titanic_survival/tested.csv')
print(titanic_dataset.head())
# Fill missing 'Age' values with the median
titanic_dataset['Age'].fillna(titanic_dataset['Age'].median(), inplace=True)

# Fill missing 'Fare' values with the median
titanic_dataset['Fare'].fillna(titanic_dataset['Fare'].median(), inplace=True)

# Drop rows with missing 'Embarked'
titanic_dataset.dropna(subset=['Embarked'], inplace=True)

# Encode 'Sex' (male = 1, female = 0) and 'Embarked' (C, Q, S)
labelencoder = LabelEncoder()
titanic_dataset['Sex'] = labelencoder.fit_transform(titanic_dataset['Sex'])
titanic_dataset['Embarked'] = labelencoder.fit_transform(titanic_dataset['Embarked'])

# Select important features for prediction
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = titanic_dataset[features]  # Features for prediction
y = titanic_dataset['Survived']  # Target variable (Survived)
print(X.head(10))
# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model on the training data
model.fit(X_train, y_train)
LogisticRegression
LogisticRegression(max_iter=1000)
# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Survived', 'Survived'], 
            yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
