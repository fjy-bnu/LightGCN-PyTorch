import csv
import matplotlib.pyplot as plt

input_csv = "../results/gowalla_item_popularity.csv"
output_png = "../results/gowalla_item_popularity_curve.png"

ranks = []
counts = []

with open(input_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ranks.append(int(row["popularity_rank"]))
        counts.append(int(row["interaction_count"]))

plt.figure(figsize=(8, 5))
plt.plot(ranks, counts)
plt.xlabel("Item popularity rank")
plt.ylabel("Interaction count")
plt.title("Gowalla Item Popularity Distribution")
plt.yscale("log")
plt.tight_layout()
plt.savefig(output_png, dpi=300)

print("Popularity curve saved to:", output_png)