"""Original frozen layer-15 trunk, exact original summaries and released C-beta head."""
import numpy as np
import torch
from study import config
from encoder import Encoder


def forbidden(module, args):
    raise RuntimeError('Unrequested LM/ConFind head or later trunk layer was evaluated')


class ContactEncoder(Encoder):
    def __init__(self):
        super().__init__()
        if self.model.contact_layer != 15:
            raise ValueError('Unexpected contact-head layer')
        self.model.lm_head.register_forward_pre_hook(forbidden)
        self.model.confind_contact_head.register_forward_pre_hook(forbidden)
        for layer in self.model.core_stack.layers[16:]:
            for module in layer:
                module.register_forward_pre_hook(forbidden)
        self.calls = 0
        self.model.contact_head.register_forward_hook(self.count_head)

    def count_head(self, module, args, output):
        self.calls += 1

    @torch.inference_mode()
    def one_contact(self, tokens, breakpoint):
        from MSA_Pairformer.dataset import prepare_msa_masks
        t = torch.as_tensor(tokens.astype(np.int64), device=self.device).unsqueeze(0)
        masks = prepare_msa_masks(t, device=self.device)
        good = t[0, 0] != 26; a, b = good[:breakpoint], good[breakpoint:]
        if not a.any() or not b.any():
            raise ValueError('No valid chain positions')
        before = self.calls
        with torch.autocast('cuda', dtype=torch.bfloat16):
            msa, pair = self.model.init_representations(torch.nn.functional.one_hot(t, 28).to(torch.bfloat16), [[breakpoint]])
            result = self.model.core_stack(msa=msa, pairwise_repr=pair, mask=masks[0], msa_mask=masks[1],
                full_mask=masks[2], pairwise_mask=masks[3], return_repr_after_layer_idx=15,
                return_pairwise_repr_only=True, store_pairwise_repr_cpu=False)
            z = result['final_pairwise_repr'][0]
            cross = (z[:breakpoint, breakpoint:] + z[breakpoint:, :breakpoint].transpose(0, 1)) / 2
            projected = (cross @ self.projection)[a][:, b].reshape(-1, 32).float()
            contact = self.model.contact_head(z.unsqueeze(0))[0, :breakpoint, breakpoint:].float()
        old = torch.cat([projected.mean(0), projected.std(0, unbiased=False), projected.amax(0),
                         torch.quantile(projected, .95, dim=0)])
        if self.calls != before+1 or not torch.isfinite(contact).all() or not torch.isfinite(old).all():
            raise ValueError('Invalid frozen contact computation')
        return contact.cpu().numpy(), old.cpu().numpy().astype(np.float64)

    def symmetric_contacts(self, tokens, breakpoint):
        first, x = self.one_contact(tokens, breakpoint)
        reverse = np.concatenate([tokens[:, breakpoint:], tokens[:, :breakpoint]], axis=1)
        second, y = self.one_contact(reverse, tokens.shape[1]-breakpoint)
        probability = (first + second.T) / np.float32(2)
        valid = (tokens[0, :breakpoint] != 26)[:, None] & (tokens[0, breakpoint:] != 26)[None, :]
        disagreement = float(np.abs(first-second.T)[valid].mean())
        return probability, (x+y)/2, disagreement

    @torch.inference_mode()
    def official(self, tokens, breakpoint):
        from MSA_Pairformer.dataset import prepare_msa_masks
        t = torch.as_tensor(tokens.astype(np.int64), device=self.device).unsqueeze(0)
        m = prepare_msa_masks(t, device=self.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            value = self.model.predict_cb_contacts(msa=torch.nn.functional.one_hot(t, 28).to(torch.bfloat16),
                mask=m[0], msa_mask=m[1], full_mask=m[2], pairwise_mask=m[3],
                complex_chain_break_indices=[[breakpoint]])
        return value['predicted_cb_contacts'][0, :breakpoint, breakpoint:].float().cpu().numpy()
