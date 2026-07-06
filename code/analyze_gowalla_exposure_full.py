import csv

popularity_csv = "../results/gowalla_item_popularity.csv"
recommendation_csv = "../results/gowalla_top20_recommendations_full.csv"
output_path = "../results/gowalla_exposure_full_summary.txt"

item_rank = {}

with open(popularity_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        item_id = int(row["item_id"])
        rank = int(row["popularity_rank"])
        item_rank[item_id] = rank

num_items = len(item_rank)
head_threshold = int(num_items * 0.2)

head_exposure = 0
tail_exposure = 0
total_exposure = 0
unique_recommended_items = set()
num_users = 0

with open(recommendation_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        num_users += 1

        for i in range(1, 21):
            item_id = int(row[f"top{i}"])
            total_exposure += 1
            unique_recommended_items.add(item_id)

            rank = item_rank.get(item_id)

            if rank is not None and rank <= head_threshold:
                head_exposure += 1
            else:
                tail_exposure += 1

head_ratio = head_exposure / total_exposure
tail_ratio = tail_exposure / total_exposure
coverage = len(unique_recommended_items) / num_items

with open(output_path, "w", encoding="utf-8") as f:
    f.write("Gowalla LightGCN Top-20 Exposure Full Analysis\n")
    f.write("============================================\n\n")

    f.write("Note: This file analyzes the full exported Top-20 recommendation list.\n\n")

    f.write(f"Number of test users: {num_users}\n")
    f.write(f"Number of items: {num_items}\n")
    f.write(f"Head item threshold: top 20%, rank <= {head_threshold}\n\n")

    f.write(f"Total exposure count: {total_exposure}\n")
    f.write(f"Head item exposure count: {head_exposure}\n")
    f.write(f"Tail item exposure count: {tail_exposure}\n\n")

    f.write(f"Head item exposure ratio: {head_ratio:.4f}\n")
    f.write(f"Tail item exposure ratio: {tail_ratio:.4f}\n")
    f.write(f"Coverage on full test users: {coverage:.6f}\n")

print("Full exposure analysis finished.")
print("Saved to:", output_path)