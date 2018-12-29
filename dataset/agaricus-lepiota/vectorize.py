d = []
with open("agaricus-lepiota.data.txt") as f:
    d = [_.split(",") for _ in f.readlines()]

l = len(d[0])
for attr in range(l):
    s = set([x[attr] for x in d])
    dic = {}
    cnt = 0
    for each in s:
        dic[each] = cnt
        cnt += 1
    for _ in range(len(d)):
        d[_][attr] = dic[d[_][attr]]

with open("vectorized.txt", "w") as f:
    for each in d:
        f.write(
            ",".join([str(_) for _ in each]) + "\n"
        )