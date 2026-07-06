from collections import Counter

train_path = "../data/gowalla/train.txt"

num_users = 0
num_interactions = 0
item_counter = Counter()

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

num_items = len(item_counter)

print("Gowalla 训练集基础统计")
print("----------------------")
print("用户数:", num_users)
print("物品数:", num_items)
print("训练交互数:", num_interactions)

print()
print("交互次数最多的前 10 个物品:")
for item_id, count in item_counter.most_common(10):
    print("item", item_id, "交互次数:", count)

print()
print("交互次数最少的后 10 个物品:")
for item_id, count in item_counter.most_common()[-10:]:
    print("item", item_id, "交互次数:", count)