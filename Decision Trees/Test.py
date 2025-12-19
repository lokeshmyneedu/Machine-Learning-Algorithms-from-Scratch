import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import sys

# 1. THE DATASET (The "Memory" of the Model)
# ------------------------------------------
# We use more data here to make it smarter
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 
                'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 
                    'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 
             'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 
                   'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# 2. PREPROCESSING (Converting Words to Numbers)
# ----------------------------------------------
# Decision Trees in sklearn require numbers, not strings.
inputs = df.drop('PlayTennis', axis='columns')
target = df['PlayTennis']

le_outlook = LabelEncoder()
le_temp = LabelEncoder()
le_humidity = LabelEncoder()
le_wind = LabelEncoder()

# Create new columns with encoded numbers
inputs['outlook_n'] = le_outlook.fit_transform(inputs['Outlook'])
inputs['temp_n'] = le_temp.fit_transform(inputs['Temperature'])
inputs['humidity_n'] = le_humidity.fit_transform(inputs['Humidity'])
inputs['wind_n'] = le_wind.fit_transform(inputs['Wind'])

# Drop the original text columns for training
inputs_n = inputs.drop(['Outlook', 'Temperature', 'Humidity', 'Wind'], axis='columns')

# 3. TRAIN THE MODEL
# ------------------
model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(inputs_n, target)

print("\n=== 🎾 TENNIS DECISION TREE TRAINED ===")
print("Mappings for your reference:")
print(f"Outlook: {dict(zip(le_outlook.classes_, le_outlook.transform(le_outlook.classes_)))}")
print(f"Temp:    {dict(zip(le_temp.classes_, le_temp.transform(le_temp.classes_)))}")
print(f"Humidity:{dict(zip(le_humidity.classes_, le_humidity.transform(le_humidity.classes_)))}")
print(f"Wind:    {dict(zip(le_wind.classes_, le_wind.transform(le_wind.classes_)))}")
print("=======================================\n")

# 4. INTERACTIVE TESTING
# ----------------------
def predict_play():
    print("Enter current weather conditions:")
    
    # Simple input handling
    try:
        user_outlook = input("Outlook (Sunny/Overcast/Rain): ").strip().capitalize()
        user_temp = input("Temperature (Hot/Mild/Cool): ").strip().capitalize()
        user_humidity = input("Humidity (High/Normal): ").strip().capitalize()
        user_wind = input("Wind (Weak/Strong): ").strip().capitalize()

        # Convert user input to model's numbers
        out_n = le_outlook.transform([user_outlook])[0]
        temp_n = le_temp.transform([user_temp])[0]
        hum_n = le_humidity.transform([user_humidity])[0]
        wnd_n = le_wind.transform([user_wind])[0]

        # Predict
        prediction = model.predict([[out_n, temp_n, hum_n, wnd_n]])
        
        print("\n------------------------------------------------")
        print(f"🤖 AI Decision: Should you play? -> {prediction[0].upper()}")
        print("------------------------------------------------\n")
        
    except ValueError as e:
        print("\n[!] Error: Please ensure you type the options exactly as shown (e.g., 'Sunny').")
    except Exception as e:
        print(f"\n[!] Error: {e}")

# Run the test loop
while True:
    predict_play()
    again = input("Test another scenario? (y/n): ")
    if again.lower() != 'y':
        print("Exiting...")
        break