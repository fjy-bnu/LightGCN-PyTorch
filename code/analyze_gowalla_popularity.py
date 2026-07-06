from collections import Counter
import csv

train_path = "../data/gowalla/train.txt"
output_csv = "../results/gowalla_item_popularity.csv"
output_summary = "../results/gowalla_popularity_summary.txt"

item_counter = Counter()
num_users = 0
num_interactions = 0

with open(train_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue

        user_id = int(parts[0])
        item_ids = [int(x) for x in parts[1:]]

        num_users += 1
        num_interactions += len(item_ids)
        item_counter.update(item_ids)

items_sorted = item_counter.most_common()
num_items = len(items_sorted)

# 保存每个物品的流行度表
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["item_id", "interaction_count", "popularity_rank"])

    for rank, (item_id, count) in enumerate(items_sorted, start=1):
        writer.writerow([item_id, count, rank])

# 做一个简单总结
top_20_percent = max(1, int(num_items * 0.2))
head_items = items_sorted[:top_20_percent]
tail_items = items_sorted[top_20_percent:]

head_interactions = sum(count for _, count in head_items)
tail_interactions = sum(count for _, count in tail_items)

with open(output_summary, "w", encoding="utf-8") as f:
    f.write("Gowalla 物品流行度分析\n")
    f.write("======================\n\n")
    f.write(f"用户数: {num_users}\n")
    f.write(f"物品数: {num_items}\n")
    f.write(f"训练交互数: {num_interactions}\n\n")

    f.write(f"前20%热门物品数量: {len(head_items)}\n")
    f.write(f"后80%长尾物品数量: {len(tail_items)}\n")
    f.write(f"前20%热门物品交互数: {head_interactions}\n")
    f.write(f"后80%长尾物品交互数: {tail_interactions}\n")
    f.write(f"前20%热门物品交互占比: {head_interactions / num_interactions:.4f}\n")
    f.write(f"后80%长尾物品交互占比: {tail_interactions / num_interactions:.4f}\n\n")

    f.write("交互次数最多的前10个物品:\n")
    for item_id, count in items_sorted[:10]:
        f.write(f"item {item_id}: {count}\n")

    f.write("\n交互次数最少的后10个物品:\n")
    for item_id, count in items_sorted[-10:]:
        f.write(f"item {item_id}: {count}\n")

print("流行度分析完成")
print("已保存:", output_csv)
print("已保存:", output_summary)