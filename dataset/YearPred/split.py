import random

path = "YearPredictionMSD.txt"
train_out = "train.csv"
train_cnt = 463715
test_out = "test.csv"

lines = open(path).readlines()
total_cnt = len(lines)
random.shuffle(lines)
ptr = 0
with open(train_out, "w") as f:
    while ptr < train_cnt:
        f.write(lines[ptr])
        ptr += 1
with open(test_out, "w") as f:
    while ptr < total_cnt:
        f.write(lines[ptr])
        ptr += 1
