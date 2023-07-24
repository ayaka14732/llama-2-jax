import json

correct = 0
total = 0

with open('results.txt', encoding='utf-8') as f:
    try:
        while True:
            a = json.loads(next(f)).rsplit('\n', 1)[1]
            b = json.loads(next(f)).rsplit('\n', 1)[1]
            correct += (a == b)
            total += 1
            if a != b:
                print(a, b)
    except StopIteration:
        pass

print(f'Correct {correct}, total {total}, acc {correct / total}')
