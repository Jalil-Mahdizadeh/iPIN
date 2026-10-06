"""V5 dispatch: native CLS classifier plus active MLM decoder for both backbones."""
from model_esm2 import PairModel as ESM2PairModel


def PairModel(base, cfg, tiny=False):
    if cfg['backbone'] == 'esm2':
        return ESM2PairModel(base, cfg, tiny=tiny)
    assert cfg['backbone'] == 'esmc'
    # ESM2 continues to run in its original image, which does not install esm.
    from model_esmc import ESMCPairModel
    return ESMCPairModel(base, cfg, tiny=tiny)
