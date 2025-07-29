import pandas as pd

def load_and_clean_data():
    print("Loading dataset...")  # Checkpoint 1
    df = pd.read_csv("players_21.csv")

    # Keep only useful columns
    df = df[[
        "short_name", "age", "height_cm", "weight_kg", "overall", "potential",
        "value_eur", "wage_eur", "international_reputation", "skill_moves",
        "weak_foot", "work_rate", "body_type", "player_traits"
    ]]

    print("Initial data shape:", df.shape)  # Checkpoint 2

    # Drop rows with missing data
    df_clean = df.dropna()

    print("Cleaned data shape:", df_clean.shape)  # Checkpoint 3
    print(df_clean.head())  # Print first few rows

    return df_clean

if __name__ == "__main__":
    print("Script started...")  # Checkpoint 0
    cleaned_df = load_and_clean_data()

    # Save to CSV so model_training.py can use it
    cleaned_df.to_csv("cleaned_players.csv", index=False)
    print("Cleaned data saved to 'cleaned_players.csv'")
