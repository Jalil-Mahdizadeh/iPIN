"""Native CLS-ReLU-linear PPI head and active pretrained MLM decoder."""
import torch
import torch.nn.functional as F


class NativePairModel(torch.nn.Module):
    def classify(self, hidden):
        return self.classifier(F.relu(hidden[:, 0])).view(-1, 2)

    def forward(self, clean_ids, masked_ids, attention_mask, chain_ids=None,
                residue_mask=None, labels=None, mlm_labels=None, compute_loss=True,
                pair_normalizer=None, mlm_normalizer=None):
        # Classification and MLM share the same corrupted-input encoder pass.
        hidden = self.encode(masked_ids if compute_loss else clean_ids, attention_mask)
        logits = self.classify(hidden)
        if not compute_loss:
            return logits
        cls = F.binary_cross_entropy_with_logits(
            logits.float(), labels[:, None].expand_as(logits).float(), reduction='none').mean(1).sum()
        selected = mlm_labels.ne(-100)
        assert selected.any() and torch.all(mlm_labels[selected] >= 0)
        mlm = F.cross_entropy(self.decode(hidden[selected]).float(), mlm_labels[selected], reduction='sum')
        pair_normalizer = len(labels) if pair_normalizer is None else pair_normalizer
        mlm_normalizer = selected.sum() if mlm_normalizer is None else mlm_normalizer
        assert pair_normalizer > 0 and mlm_normalizer > 0
        loss = self.cfg['classification_weight']*cls/pair_normalizer + self.cfg['mlm_weight']*mlm/mlm_normalizer
        return loss, cls.detach(), mlm.detach(), logits.detach()
