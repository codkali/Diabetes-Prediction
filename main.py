import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
# Ignore UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Check for missing values
print("Missing values in the dataset:")
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Preprocessing using Ordinal Encoder
enc = OrdinalEncoder()
df["smoking_history"] = enc.fit_transform(df[["smoking_history"]])
df["gender"] = enc.fit_transform(df[["gender"]])

# Define Independent and Dependent Variables
x = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Check data types
print("\nData types of features:")
print(x.dtypes)

# 70% data - Train and 30% data - Test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Check shapes
print("\nShapes of training and testing data:")
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

# DecisionTree Algorithm
model = DecisionTreeClassifier(random_state=42).fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

# Print accuracy
print(f"\nModel Accuracy: {accuracy:.2f}")

# Function to get user input and make predictions
def predict_diabetes():
    print("\nEnter the following details to predict diabetes:")
    gender = input("Gender (Male/Female/Other): ").title()
    age = input("Age: ")
    hypertension = input("Hypertension (No/Yes): ")
    heart_disease = input("Heart Disease (No/Yes): ")
    smoking_history = input("Smoking History (Never/Current/Former/Ever/Not Current/No Info): ")

    while True:
        try:
            bmi = float(input("BMI (10-50): "))
            if 10 <= bmi <= 50:
                break
            else:
                print("Please enter a BMI value between 10 and 50.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    while True:
        try:
            hba1c_level = float(input("HbA1c Level (4-15): "))
            if 4 <= hba1c_level <= 15:
                break
            else:
                print("Please enter an HbA1c Level value between 4 and 15.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    while True:
        try:
            blood_glucose_level = float(input("Blood Glucose Level (70-200): "))
            if 70 <= blood_glucose_level <= 200:
                break
            else:
                print("Please enter a Blood Glucose Level value between 70 and 200.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    # Dictionaries for encoding user input
    gender_dict = {'Female': 0.0, 'Male': 1.0, 'Other': 2.0}
    hypertension_dict = {'No': 0, 'Yes': 1}
    heart_disease_dict = {'No': 0, 'Yes': 1}
    smoking_history_dict = {'Never': 4.0, 'No Info': 0.0, 'Current': 1.0,
                            'Former': 3.0, 'Ever': 2.0, 'Not Current': 5.0}

    try:
        # Convert user input to a numpy array
        user_data = np.array([[gender_dict[gender], float(age), hypertension_dict[hypertension],
                               heart_disease_dict[heart_disease], smoking_history_dict[smoking_history],
                               bmi, hba1c_level, blood_glucose_level]])

        # Make prediction
        test_result = model.predict(user_data)
        probabilities = model.predict_proba(user_data)[0]

        # Display result
        if test_result[0] == 0:
            print("\nDiabetes Result: Negative")
        else:
            print("\nDiabetes Result: Positive (Please Consult with Doctor)")

        # Plot probabilities
        labels = ['Negative', 'Positive']
        plt.bar(labels, probabilities, color=['blue', 'red'])
        plt.xlabel('Diabetes Result')
        plt.ylabel('Probability')
        plt.title('Diabetes Prediction Probability')
        plt.show()

    except Exception as e:
        print(f"\nAn error occurred: {e}. Please check your input and try again.")

# Run the prediction function
if __name__ == "__main__":
    while True:
        predict_diabetes()
        another = input("\nWould you like to make another prediction? (yes/no): ").strip().lower()
        if another != 'yes':
            print("Thank you for using the Diabetes Prediction tool. Goodbye!")
            break