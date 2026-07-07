import csv

popularity_csv = "../results/gowalla_item_popularity.csv"
recommendation_csv = "../results/gowalla_top20_recommendations_sample.csv"
output_path = "../results/gowalla_exposure_metrics_sample.txt"

TOPK = 20

# 读取物品流行度排名
item_rank = {}
item_count = {}

with open(popularity_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        item_id = int(row["item_id"])
        interaction_count = int(row["interaction_count"])
        rank = int(row["popularity_rank"])

        item_rank[item_id] = rank
        item_count[item_id] = interaction_count

num_items = len(item_rank)
head_threshold = int(num_items * 0.2)

# 曝光统计
total_exposure = 0
head_exposure = 0
tail_exposure = 0
unique_recommended_items = set()
recommended_popularity_ranks = []
recommended_interaction_counts = []

position_head_count = [0 for _ in range(TOPK)]
position_total_count = [0 for _ in range(TOPK)]

item_exposure_counter = {}

num_users = 0

with open(recommendation_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        num_users += 1

        for i in range(1, TOPK + 1):
            item_id = int(row[f"top{i}"])
            pos_index = i - 1

            total_exposure += 1
            position_total_count[pos_index] += 1
            unique_recommended_items.add(item_id)

            item_exposure_counter[item_id] = item_exposure_counter.get(item_id, 0) + 1

            rank = item_rank.get(item_id)
            count = item_count.get(item_id)

            if rank is not None:
                recommended_popularity_ranks.append(rank)

                if rank <= head_threshold:
                    head_exposure += 1
                    position_head_count[pos_index] += 1
                else:
                    tail_exposure += 1
            else:
                tail_exposure += 1

            if count is not None:
                recommended_interaction_counts.append(count)

head_ratio = head_exposure / total_exposure
tail_ratio = tail_exposure / total_exposure
coverage = len(unique_recommended_items) / num_items

avg_popularity_rank = sum(recommended_popularity_ranks) / len(recommended_popularity_ranks)
avg_interaction_count = sum(recommended_interaction_counts) / len(recommended_interaction_counts)

# 计算曝光 Gini：衡量推荐曝光是否集中在少数物品上
exposure_values = []

for item_id in item_rank.keys():
    exposure_values.append(item_exposure_counter.get(item_id, 0))

exposure_values.sort()
n = len(exposure_values)
total = sum(exposure_values)

if total == 0:
    gini = 0
else:
    weighted_sum = 0
    for idx, value in enumerate(exposure_values, start=1):
        weighted_sum += idx * value
    gini = (2 * weighted_sum) / (n * total) - (n + 1) / n

top_recommended_items = sorted(
    item_exposure_counter.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

with open(output_path, "w", encoding="utf-8") as f:
    f.write("Gowalla LightGCN Exposure Metrics Sample\n")
    f.write("======================================\n\n")

    f.write("Note: This result is based on 200 sampled test users.\n")
    f.write("It is used to verify the metric calculation pipeline.\n\n")

    f.write(f"Number of sampled users: {num_users}\n")
    f.write(f"TopK: {TOPK}\n")
    f.write(f"Number of items: {num_items}\n")
    f.write(f"Head item threshold: top 20%, rank <= {head_threshold}\n\n")

    f.write(f"Total exposure count: {total_exposure}\n")
    f.write(f"Head exposure count: {head_exposure}\n")
    f.write(f"Tail exposure count: {tail_exposure}\n")
    f.write(f"Head exposure ratio: {head_ratio:.4f}\n")
    f.write(f"Tail exposure ratio: {tail_ratio:.4f}\n\n")

    f.write(f"Unique recommended items: {len(unique_recommended_items)}\n")
    f.write(f"Coverage: {coverage:.6f}\n")
    f.write(f"Exposure Gini: {gini:.6f}\n")
    f.write(f"Average popularity rank of recommended items: {avg_popularity_rank:.2f}\n")
    f.write(f"Average training interaction count of recommended items: {avg_interaction_count:.2f}\n\n")

    f.write("Position-wise head exposure ratio:\n")
    for i in range(TOPK):
        ratio = position_head_count[i] / position_total_count[i]
        f.write(f"Top-{i + 1}: {ratio:.4f}\n")

    f.write("\nMost exposed recommended items:\n")
    for item_id, exposure_count in top_recommended_items:
        f.write(
            f"item {item_id}: exposure={exposure_count}, "
            f"popularity_rank={item_rank[item_id]}, "
            f"train_interactions={item_count[item_id]}\n"
        )

print("Exposure metrics sample analysis finished.")
print("Saved to:", output_path)