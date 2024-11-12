from pyspark.sql import functions as F

df = spark.read.format("delta").table("cleanedhistorical")

# Calculate Home Goals Per Game (HGPG) and Away Goals Per Game (AGPG)
goals_per_game = df.groupBy("Season").agg(
    (F.sum("Full_Time_Home_Goals") / F.count("*")).alias("HGPG"),  # Count all rows for matches played
    (F.sum("Full_Time_Away_Goals") / F.count("*")).alias("AGPG")   # Count all rows for matches played
)

goals_per_game_sorted = goals_per_game.orderBy("Season")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Convert to Pandas DataFrame for visualization
pdf = goals_per_game_sorted.toPandas()

# Prepare data for trend lines
x = np.arange(len(pdf['Season']))  # Numeric x-values for fitting
y_hgpg = pdf['HGPG']
y_agpg = pdf['AGPG']

# Fit polynomial trend lines (1st degree polynomial = linear fit)
coeff_hgpg = np.polyfit(x, y_hgpg, 1)
coeff_agpg = np.polyfit(x, y_agpg, 1)

# Create polynomial functions from the coefficients
poly_hgpg = np.poly1d(coeff_hgpg)
poly_agpg = np.poly1d(coeff_agpg)

# Generate y-values for the trend lines
trend_hgpg = poly_hgpg(x)
trend_agpg = poly_agpg(x)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(pdf['Season'], pdf['HGPG'], marker='o', label='Home Goals per Game')
plt.plot(pdf['Season'], pdf['AGPG'], marker='o', label='Away Goals per Game')
plt.plot(pdf['Season'], trend_hgpg, linestyle='--', color='blue', label='Trend Line for HGPG')
plt.plot(pdf['Season'], trend_agpg, linestyle='--', color='orange', label='Trend Line for AGPG')

plt.title('Home Goals per Game vs. Away Goals per Game by Season')
plt.xlabel('Season')
plt.ylabel('Goals per Game')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()