import random

def x(l):
    ret = l.replace("Iris-versicolor", "1")
    ret = ret.replace("Iris-virginica", "2")
    ret = ret.replace("Iris-setosa", "3")
    return ret

ratio = 0.2
fn = "bezdekIris.data.txt"
datas = None
with open(fn, "r") as f:
    datas = f.readlines()
datas = [l for l in datas if l.strip("\n") != ""]

random.shuffle(datas)

test = int(len(datas) * ratio)

cnt = 0
with open("test.txt", "w") as f:
    while cnt < test:
        f.write(x(datas[cnt]))
        cnt += 1

with open("train.txt", "w") as f:
    while cnt < len(datas):
        f.write(x(datas[cnt]))
        cnt += 1