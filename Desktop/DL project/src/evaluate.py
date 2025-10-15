import re
import pandas as pd
import matplotlib.pyplot as plt

# Path to your training log file
LOG_FILE = "src/training.txt"

# Read the log file
with open(LOG_FILE, "r") as f:
    lines = f.readlines()

# Extract epoch, D_loss, G_loss
data = []
for line in lines:
    # Match format like: [Epoch 1/1000] D loss: -36.6943 | G loss: 12.2987
    match = re.search(r"\[Epoch\s+(\d+)/\d+\]\s+D\s+loss:\s*([-.\d]+)\s*\|\s*G\s+loss:\s*([-.\d]+)", line)
    if match:
        epoch = int(match.group(1))
        d_loss = float(match.group(2))
        g_loss = float(match.group(3))
        data.append((epoch, d_loss, g_loss))

# Create a DataFrame
df = pd.DataFrame(data, columns=["Epoch", "D_loss", "G_loss"])

# Add derived metrics
df["D_loss_change_%"] = df["D_loss"].pct_change() * 100
df["G_loss_change_%"] = df["G_loss"].pct_change() * 100
df["D_to_G_ratio"] = df["D_loss"].abs() / df["G_loss"]

# Display summary statistics
print("\n📊 Training Summary:")
print(df.describe())

# Plot losses over epochs
plt.figure(figsize=(10, 5))
plt.plot(df["Epoch"], df["D_loss"], label="D_loss")
plt.plot(df["Epoch"], df["G_loss"], label="G_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Losses per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save to CSV for further analysis
df.to_csv("training_stats.csv", index=False)
print("\n✅ Saved training statistics to training_stats.csv")
