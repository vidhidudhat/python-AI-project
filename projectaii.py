import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_csv("water_potabilty.csv")


# Split the dataset into features (X) and target (y)
X = data.drop(columns=["potability"])
y = data["potability"]

# Impute missing values using the mean strategy
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Apply SMOTE oversampling to the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train a Logistic Regression model on the oversampled training set
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_smote, y_train_smote)

# Make predictions on the test set
y_pred = model.predict(X_test)



# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy)

#Alternatively, you can use the following code to calculate the accuracy manually:

# Calculate the number of true positives, true negatives, false positives, and false negatives
tp = sum((y_test == 1) & (y_pred == 1))
tn = sum((y_test == 0) & (y_pred == 0))
fp = sum((y_test == 0) & (y_pred == 1))
fn = sum((y_test == 1) & (y_pred == 0))

# Calculate the accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)




# def predict_from_dataset():
    # Ask the user for input features values




pH = float(input("Enter pH value: "))
Hardness = float(input("Enter Hardness value: "))
Solids = float(input("Enter Solids value: "))
Chloramines = float(input("Enter Chloramines value: "))
Sulfate = float(input("Enter Sulfate value: "))
Conductivity = float(input("Enter Conductivity value: "))
organic_carbon = float(input("Enter Organic_carbon value: "))
Trihalomethanes = float(input("Enter Trihalomethanes value: "))
Turbidity = float(input("Enter Turbidity value: "))
    # Create a test instance with the user input values
test_instance = [[pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, organic_carbon, Trihalomethanes, Turbidity]]

    # Scale the test instance using the StandardScaler
scaled_test_instance = scaler.transform(test_instance)

    # Make a prediction
prediction = model.predict(scaled_test_instance)
output = "Potable" if prediction[0] == 1 else "non-potable"

 # Print the prediction results
print("\nPrediction Results:")
print("Input Features:")
print(f" - pH: {pH}")
print(f" - Hardness: {Hardness}")
print(f" - Solids: {Solids}")
print(f" - Chloramines: {Chloramines}")
print(f" - Sulfate: {Sulfate}")
print(f" - Conductivity: {Conductivity}")
print(f" - Organic_carbon: {organic_carbon}")
print(f" - Trihalomethanes: {Trihalomethanes}")
print(f" - Turbidity: {Turbidity}")
print("\nOutput:")
print("Accuracy:", accuracy)
print(f" - Predicted Water Quality: {output}")



    
    
input_values = {
    "pH": pH,
    "Hardness": Hardness,
    "Solids": Solids,
    "Chloramines": Chloramines,
    "Sulfate": Sulfate,
    "Conductivity": Conductivity,
    "Organic Carbon": organic_carbon,
    "Trihalomethanes": Trihalomethanes,
    "Turbidity": Turbidity
}

# Create a bar chart to display the input values
plt.bar(input_values.keys(), input_values.values(),color="red")
plt.xlabel("Water Quality Parameters")
plt.ylabel("Values")
plt.title("Water Quality Checker")
plt.show()

# Call the function to predict water quality
# predict_from_dataset()
