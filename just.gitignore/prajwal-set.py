import pandas as pd
import numpy as np
import os

NUM_SAMPLES = 10000
DATASET_FOLDER = 'datasets'
OUTPUT_CSV_PATH = os.path.join(DATASET_FOLDER, 'simulated_behavior_data(1).csv')

CATEGORY_PROFILES = {
    "Work/Academic": {"base_dist": 0.1, "avg_dwell": 240, "dwell_std": 120},
    "Social Media":  {"base_dist": 0.8, "avg_dwell": 300, "dwell_std": 180},
    "Entertainment": {"base_dist": 0.9, "avg_dwell": 600, "dwell_std": 400},
    "News":          {"base_dist": 0.6, "avg_dwell": 180, "dwell_std": 90},
    "Shopping":      {"base_dist": 0.7, "avg_dwell": 250, "dwell_std": 150},
    "Other":         {"base_dist": 0.4, "avg_dwell": 120, "dwell_std": 60}
}
TIME_OF_DAY_MULTIPLIERS = {"Morning": 0.8, "Afternoon": 1.1, "Evening": 1.3, "Night": 1.2}

print("Starting enhanced behavioral data simulation...")

data = []
for _ in range(NUM_SAMPLES):
    category = np.random.choice(list(CATEGORY_PROFILES.keys()))
    time_of_day = np.random.choice(list(TIME_OF_DAY_MULTIPLIERS.keys()))

    profile = CATEGORY_PROFILES[category]
    base_dist = profile["base_dist"]
    time_multiplier = TIME_OF_DAY_MULTIPLIERS[time_of_day]

    # --- Simulate more realistic behavioral features ---
    dwell_time = max(5, np.random.normal(loc=profile["avg_dwell"], scale=profile["dwell_std"]))

    # Tab switches are higher if base distraction is high OR if dwell time on a productive site is low (indicating searching)
    is_searching = category == "Work/Academic" and dwell_time < 60
    switch_lamda = 2 + (base_dist * 8) + (5 if is_searching else 0)
    tab_switches = np.random.poisson(lam=switch_lamda)

    domain_score = max(0, min(100, int(base_dist * 100 + np.random.normal(0, 10))))

    # --- Calculate Final Distraction Score & Label using heuristics ---
    dwell_factor = (dwell_time / profile["avg_dwell"]) - 1.0  # How much longer/shorter than average?
    distraction_score = base_dist + (dwell_factor * 0.2)  # Long dwell times modify the base score
    distraction_score *= time_multiplier # Time of day has a big impact

    # High tab switching is always a sign of distraction
    if tab_switches > 10:
        distraction_score += 0.2

    # Assign label based on final score
    if distraction_score > 0.9:
        label = "High"
    elif distraction_score > 0.5:
        label = "Medium"
    else:
        label = "Low"

    data.append([category, time_of_day, int(dwell_time), tab_switches, domain_score, label])

# Create DataFrame and save
columns = ['tab_category', 'time_of_day', 'dwell_time_seconds', 'recent_tab_switches', 'domain_reputation_score', 'distraction_label']
df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"Simulation complete. Dataset saved to '{OUTPUT_CSV_PATH}' with {len(df)} rows.")
print("\nSample of the new, more realistic data:")
print(df.sample(5))