import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

# Load and display dataset
df = pd.read_csv('weather_forecast.csv')
print(df.head())

# Preprocess the data
encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(df.drop('Play', axis=1)).toarray()
y = df['Play']
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Define and evaluate the decision tree classifiers
for criterion, name in [('entropy', 'ID3'), ('gini', 'CART')]:
    clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
    clf.fit(X_train, y_train)
    
    # Visualize the decision tree
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=encoder.get_feature_names_out(), class_names=['No', 'Yes'])
    plt.title(f'{name} Decision Tree')
    plt.show()
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    print(f"{name} Algorithm Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X_encoded, y, cv=5)
    print(f"Cross-Validation Scores ({name}):", cv_scores)
    print(f"Mean CV Accuracy ({name}):", cv_scores.mean())
