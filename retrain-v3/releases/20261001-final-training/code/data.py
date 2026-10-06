"""Fixed pair exposure, stateless corruption, explicit chain and residue metadata."""
import numpy as np
import torch


class PairData:
    def __init__(self, path, partition, split, cap=None):
        assert partition in ['official', 'fold-0', 'fold-1', 'fold-2']
        assert split in ['train', 'val'], 'Production trainer must not load a test split'
        self.rows = np.load(path / partition / f'{split}.npy')
        self.tokens = np.load(path / 'tokens.npy', mmap_mode='r')
        self.offsets = np.load(path / 'offsets.npy', mmap_mode='r')
        lengths = np.diff(self.offsets)
        self.lengths = lengths[self.rows[:, 0]] + lengths[self.rows[:, 1]] + 3
        if cap is not None:
            assert split == 'train', 'Validation coverage must remain unchanged'
            keep = self.lengths <= cap + 3
            self.rows, self.lengths = self.rows[keep], self.lengths[keep]
        assert len(self.rows) and set(np.unique(self.rows[:, 2])) == {0, 1}
        self._plans = {}

    def __len__(self): return len(self.rows)

    def sequence(self, index):
        return self.tokens[self.offsets[index]:self.offsets[index + 1]].astype(np.int64)

    def plan(self, cycle, seed, batch_size):
        key = (cycle, seed, batch_size)
        if key not in self._plans:
            rng = np.random.default_rng(np.random.SeedSequence([seed, cycle, 712]))
            ids = rng.permutation(len(self))
            chunks = []
            for start in range(0, len(ids), batch_size * 50):
                pool = ids[start:start + batch_size * 50]
                pool = pool[np.argsort(self.lengths[pool], kind='stable')]
                chunks.extend(pool[i:i + batch_size] for i in range(0, len(pool), batch_size))
            rng.shuffle(chunks)
            self._plans = {key: np.concatenate(chunks)}
        return self._plans[key]

    def next_batch(self, cursor, seed, batch_size):
        cycle, offset = cursor['cycle'], cursor['offset']
        indices, cycles = [], []
        while len(indices) < batch_size:
            plan = self.plan(cycle, seed, batch_size)
            take = min(batch_size - len(indices), len(plan) - offset)
            indices.extend(plan[offset:offset + take].tolist())
            cycles.extend([cycle] * take)
            offset += take
            if offset == len(plan): cycle, offset = cycle + 1, 0
        return np.array(indices), np.array(cycles), {'cycle': cycle, 'offset': offset}

    def microbatches(self, indices, token_budget, max_pairs):
        # Positions keep each example's masking cycle aligned with its row.
        current, maximum = [], 0
        for position, idx in enumerate(indices):
            length = int(self.lengths[idx])
            new_max = max(maximum, length)
            if current and (len(current) >= max_pairs or 2 * (len(current) + 1) * new_max > token_budget):
                yield np.array(current)
                current, maximum = [], 0
            current.append(position)
            maximum = max(maximum, length)
        if current: yield np.array(current)

    def batch(self, indices, cycles, seed, mask_any, device):
        clean, masked, targets, chain_ids, residue_masks, labels = [], [], [], [], [], []
        for idx, cycle in zip(indices, cycles):
            a, b, label, source_row = self.rows[idx]
            originals = [self.sequence(a), self.sequence(b)]
            altered, supervised = [], []
            for side, sequence in enumerate(originals):
                changed = sequence.copy()
                target = np.full(len(sequence), -100, dtype=np.int64)
                if mask_any:
                    rng = np.random.default_rng(np.random.SeedSequence([seed, int(cycle), int(source_row), side, 981]))
                    selected = rng.random(len(sequence)) < .15
                    if not selected.any(): selected[rng.integers(len(sequence))] = True
                    target[selected] = sequence[selected]
                    chance = rng.random(len(sequence))
                    changed[selected & (chance < .8)] = 32
                    replace = selected & (chance >= .8) & (chance < .9)
                    changed[replace] = rng.integers(4, 24, replace.sum())
                altered.append(changed)
                supervised.append(target)
            for x, z in [(0, 1), (1, 0)]:
                clean.append(np.concatenate(([0], originals[x], [2], originals[z], [2])))
                masked.append(np.concatenate(([0], altered[x], [2], altered[z], [2])))
                targets.append(np.concatenate(([-100], supervised[x], [-100], supervised[z], [-100])))
                chain_ids.append(np.concatenate((np.zeros(len(originals[x]) + 2, dtype=np.int64),
                                                 np.ones(len(originals[z]) + 1, dtype=np.int64))))
                residue_masks.append(np.concatenate(([False], np.ones(len(originals[x]), dtype=bool), [False],
                                                       np.ones(len(originals[z]), dtype=bool), [False])))
            labels.append(label)
        length = max(map(len, clean))
        def pad(values, fill, dtype):
            result = np.full((len(values), length), fill, dtype=dtype)
            for i, value in enumerate(values): result[i, :len(value)] = value
            return torch.from_numpy(result).to(device)
        valid = [np.ones(len(x), dtype=np.int64) for x in clean]
        return {'clean_ids': pad(clean, 1, np.int64), 'masked_ids': pad(masked, 1, np.int64),
                'attention_mask': pad(valid, 0, np.int64), 'mlm_labels': pad(targets, -100, np.int64),
                'chain_ids': pad(chain_ids, -1, np.int64), 'residue_mask': pad(residue_masks, False, bool),
                'labels': torch.tensor(labels, dtype=torch.float32, device=device)}
