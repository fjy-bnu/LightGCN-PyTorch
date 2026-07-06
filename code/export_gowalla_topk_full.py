import csv
import os
import torch

import world
import utils
import register
from register import dataset

TOPK = 20
output_path = "../results/gowalla_top20_recommendations_full.csv"

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)

weight_file = utils.getFileName()
print("Try to load checkpoint:", weight_file)

if os.path.exists(weight_file):
    Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))
    print("Checkpoint loaded.")
else:
    print("Checkpoint not found. Use current model parameters.")

Recmodel.eval()

users = list(dataset.testDict.keys())
u_batch_size = world.config["test_u_batch_size"]

print("Total test users:", len(users))
print("Batch size:", u_batch_size)
print("TopK:", TOPK)

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    header = ["user_id"] + [f"top{i}" for i in range(1, TOPK + 1)]
    writer.writerow(header)

    with torch.no_grad():
        batch_id = 0

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            batch_id += 1

            allPos = dataset.getUserPosItems(batch_users)

            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)

            exclude_index = []
            exclude_items = []

            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating[exclude_index, exclude_items] = -(1 << 10)

            _, rating_K = torch.topk(rating, k=TOPK)
            rating_K = rating_K.cpu().numpy().tolist()

            for user_id, rec_items in zip(batch_users, rating_K):
                writer.writerow([user_id] + rec_items)

            if batch_id % 20 == 0:
                print("Finished batch:", batch_id)

print("Full Top-20 recommendation export finished.")
print("Saved to:", output_path)