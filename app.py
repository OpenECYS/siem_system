import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Log line example
# 2025-03-06 08:00:00, INFO, User login success, user: admin

logs = [
    "2025-03-06 08:00:00, INFO, User login success, user: admin",
    "2025-03-06 08:01:23, INFO, User login success, user: alice",
    "2025-03-06 08:02:45, ERROR, Failed login attempt, user: alice",
    # More log lines...
]

parsed_logs = []

for line in logs:
    parts = [p.strip() for p in line.split(",")]
    timestamp = parts[0]
    level = parts[1]
    message = parts[2]
    user = parts[3].split(":")[1].strip() if "user:" in parts[3] else None
    parsed_logs.append({"timestamp": timestamp, "level": level, "message": message, "user": user})

# Convert to DataFrame for easier analysis
df_logs = pd.DataFrame(parsed_logs)
print(df_logs.head())

np.random.seed(42)
normal_counts = np.random.poisson(lam=5, size=50)

anomalous_counts = np.array([30, 40, 50])

login_attempts = np.concatenate([normal_counts, anomalous_counts])
print("Login attempts:", login_attempts)

X = login_attempts.reshape(-1, 1)
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

# Use the model to predict anomalies
label = model.predict(X)

# Extract the anomaly indices and values
anomaly_indices = np.where(label == -1)[0]
anomaly_values = login_attempts[anomaly_indices]

if len(anomaly_indices) > 0:
    print(f"Alert! Detected {len(anomaly_indices)} anomalous events. Initiating response procedures...")

plt.plot(login_attempts, label="Login attempts per minute")
plt.scatter(anomaly_indices, anomaly_values, color='red', label="Anomalies")
plt.xlabel("Time (minute index)")
plt.ylabel("Login attempts")
plt.legend()
plt.show()