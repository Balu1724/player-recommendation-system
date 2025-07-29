import pickle

def load_model():
    print("✅ Model loaded successfully!")
    with open("player_model.pkl", "rb") as f:
        return pickle.load(f)

def main():
    model = load_model()

    print("Enter the following player details:")
    age = int(input("Age: "))
    height = int(input("Height (cm): "))
    weight = int(input("Weight (kg): "))
    overall = int(input("Overall rating (0–100): "))
    potential = int(input("Potential rating (0–100): "))

    # These are unused for now (kept for future use or full feature models)
    input("Market value in EUR: ")
    input("Wage in EUR: ")
    input("International reputation (1–5): ")
    input("Skill moves (1–5): ")
    input("Weak foot rating (1–5): ")

    # Only use the 5 features model was trained with
    input_data = [[age, height, weight, overall, potential]]

    prediction = model.predict(input_data)[0]

    result = "✅ Recommended" if prediction == 1 else "❌ Not Recommended"
    print(f"\n👉 {result}")

if __name__ == "__main__":
    main()
