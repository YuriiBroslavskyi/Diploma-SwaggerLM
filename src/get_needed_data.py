import json
from collections import defaultdict
import random

random.seed(42)

records = [json.loads(l) for l in open('apis_guru_20260521_170953.jsonl')]

# групуємо по провайдеру
by_provider = defaultdict(list)
for r in records:
    by_provider[r['_meta']['provider']].append(r)

# максимум N записів на провайдера
MAX_PER_PROVIDER = 15  # 656 провайдерів × 15 ≈ 9,840 записів

balanced = []
for provider, recs in by_provider.items():
    sample = random.sample(recs, min(MAX_PER_PROVIDER, len(recs)))
    balanced.extend(sample)

random.shuffle(balanced)

print(f'Total balanced records: {len(balanced):,}')
print(f'Providers covered: {len(by_provider):,}')

with open('apis_guru_balanced.jsonl', 'w') as f:
    for r in balanced:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print('✅ Saved → apis_guru_balanced.jsonl')