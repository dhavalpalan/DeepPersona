import numpy as np
import pandas as pd

def generate_bot():
    sophisticated = np.random.random() < 0.3  

    if sophisticated:
        click_interval = max(10, np.random.normal(600, 150))
        scroll_variance = max(0.01, np.random.normal(0.25, 0.08))
        mouse_score = max(0.0, np.random.normal(0.35, 0.1))
        session_duration = max(0.1, np.random.normal(8, 3))
        skip_rate = min(1.0, max(0.0, np.random.normal(0.35, 0.1)))        
        typing_variance = max(0.0, np.random.normal(0.25, 0.08))           
        browser_headless = np.random.choice([0, 1], p=[0.6, 0.4])         
        action_count = max(1, int(np.random.normal(700, 100)))               
    else:
        click_interval = max(10, np.random.normal(150, 60))
        scroll_variance = max(0.0, np.random.normal(0.005, 0.002))
        mouse_score = max(0.0, np.random.normal(0.1, 0.07))
        session_duration = max(0.1, np.random.normal(2.5, 1.2))
        skip_rate = min(1.0, max(0.0, np.random.normal(0.85, 0.1)))
        typing_variance = max(0.0, np.random.normal(0.04, 0.02))
        browser_headless = np.random.choice([0, 1], p=[0.2, 0.8])
        action_count = max(1, int(np.random.normal(1200, 200)))
    return {
        "session_duration_min": session_duration,
        "action_count": action_count,
        "click_interval_ms": click_interval,
        "scroll_variance": scroll_variance,
        "skip_rate": skip_rate,
        "mouse_movement_score": mouse_score,
        "typing_rhythm_variance": typing_variance,
        "browser_headless": browser_headless,
        "is_bot": 1
    }

def generate_human():
    return{
        "session_duration_min": max(0.1, np.random.normal(5, 3)),
        "action_count": max(1, int(np.random.normal(400, 150))),
        "click_interval_ms":max(200,np.random.normal(1400,500)),
        "scroll_variance": min(1.0, max(0.0, np.random.normal(0.05, 0.02))),
        "skip_rate": min(1.0, max(0.0, np.random.normal(0.2, 0.1))),
        "mouse_movement_score": min(1.0, max(0.0, np.random.normal(0.35, 0.1))),
        "typing_rhythm_variance": min(1.0, max(0.0, np.random.normal(0.38, 0.12))),
        "browser_headless" : np.random.choice([0,1], p=[0.97,0.03]),
        "is_bot" : 0
    }

rows = []

for _ in range(900):
    rows.append(generate_human())

for _ in range(400):
    rows.append(generate_bot())

df = pd.DataFrame(rows)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("sessions.csv", index=False)

print(f"Dataset saved: {len(df)} rows")
print(df.head())
print(df['is_bot'].value_counts())

noise_cols = [
    "session_duration_min", "action_count", "click_interval_ms",
    "scroll_variance", "skip_rate", "mouse_movement_score",
    "typing_rhythm_variance"
]

for col in noise_cols:
    noise = np.random.normal(0, df[col].std() * 0.15, size=len(df))
    df[col] = df[col] + noise

df["session_duration_min"] = df["session_duration_min"].clip(lower=0.1)
df["action_count"] = df["action_count"].clip(lower=1).astype(int)
df["click_interval_ms"] = df["click_interval_ms"].clip(lower=10)
df["scroll_variance"] = df["scroll_variance"].clip(0.01, 1.0)
df["skip_rate"] = df["skip_rate"].clip(0.0, 1.0)
df["mouse_movement_score"] = df["mouse_movement_score"].clip(0.0, 1.0)
df["typing_rhythm_variance"] = df["typing_rhythm_variance"].clip(0.0, 1.0)