import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# Load the dataset
df = pd.read_csv(r"C:\Users\tmabu\OneDrive\Desktop\Bank_Churners_Credit_Cards.csv")

# ===============================================
# 1. Feature Selection and Target Encoding
# ===============================================

# Drop irrelevant columns (CLIENTNUM) and the Naive Bayes columns, 
# which are likely the result of a pre-existing model.
cols_to_drop = ['CLIENTNUM', 
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']
df = df.drop(columns=cols_to_drop)

# Create the binary target variable 'Churn' (1 for Attrited Customer, 0 for Existing Customer)
df['Churn'] = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)
df = df.drop(columns=['Attrition_Flag'])

# ===============================================
# 2. Data Preprocessing (One-Hot Encoding)
# ===============================================

# Identify categorical columns (object types)
categorical_cols = df.select_dtypes(include=['object']).columns

# One-Hot Encode categorical features, dropping the first category to avoid multicollinearity
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

# ===============================================
# 3. Model Training
# ===============================================

# Separate Features (X) and Target (y)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Split data into training and test sets (70% train, 30% test, stratified by 'Churn')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize and train a Random Forest Classifier
# 'class_weight="balanced"' helps mitigate the class imbalance in the target variable
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# ===============================================
# 4. Prediction and Evaluation
# ===============================================

y_pred = rf_model.predict(X_test)

# Classification Report
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['No Churn (0)', 'Churn (1)']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n--- Confusion Matrix (Actual vs. Predicted) ---")
print(cm)

# Feature Importance Plot (Saved as 'top_20_feature_importances.png')
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_20_features = feature_importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 8))
top_20_features.plot(kind='barh')
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('top_20_feature_importances.png')
model_filename = 'bank_churn_model.joblib'
joblib.dump(rf_model, model_filename)