import json

correct = 0
total = 0

with open('results.txt', encoding='utf-8') as f:
    for line in f:
        a, b = json.loads(line)
        a = a.rsplit('\n', 1)[1]
        b = b.rsplit('\n', 1)[1]
        correct += (a == b)
        total += 1
        if a != b:
            print(a, b)

print(f'Correct {correct}, total {total}, acc {correct / total}')
