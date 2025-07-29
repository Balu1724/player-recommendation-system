import pickle

def main():
    # Load the trained model
    with open("player_model.pkl", "rb") as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully!")

    # Ask user for the 5 input values
    print("Enter the following player details:")
    age = int(input("Age: "))
    height = int(input("Height (cm): "))
    weight = int(input("Weight (kg): "))
    overall = int(input("Overall rating (0–100): "))
    potential = int(input("Potential rating (0–100): "))

    # Create input in correct format
    input_data = [[age, height, weight, overall, potential]]

    # Predict
    prediction = model.predict(input_data)[0]

    # Output result
    if prediction == 1:
        print("✅ This player is recommended!")
    else:
        print("❌ This player is not recommended.")

if __name__ == "__main__":
    main()
