"""Declared once, before candidate test pairs or test truth are opened."""
SCORERS = ('sprint_native_train_positive_graph',)
REFERENCES = ('ipin_baseline', 'ipin_optimized')
ALL_SCORERS = SCORERS + REFERENCES
REVISION = 'b6272c76e1a943e6812b1b815607691819e6202e'
SPRINT_SIF_SHA = '699112807a7829b1b4268fa83d358a206d62de5a5a0f7c7d199beb18f9f47f09'
MODEL_SIF_SHA = 'c4bddf5f7b40cf7c5bbfba82f47ef2b1bbc5786c7bb36d98b020ca09761aad91'
SEQUENCES_SHA = 'bc7a91661ea05cbfdcf3551b9a4d18c149ac077549ebbcf0be7e7ef5d94ace77'
TRAINING_SHA = '43ae252277820ffe527dad1ed4839d5673cff279b6c8c943ff4b5350a9283ca6'
HSP_ARGS = ['-Thit', '15', '-Tsim', '35', '-M', '1']
PREDICT_ARGS = ['-Thc', '40']
HSP_THREADS = 64
POLICY = {
    'model': 'Original unmodified SPRINT algorithm with the iPIN TRAIN-positive graph',
    'upstream_revision': REVISION,
    'training': 'All 16799 frozen TRAIN positives in frozen input order; no U, development/test interactions, or external PPI graph',
    'sequence_features': 'All 17000 frozen full-length sequences; no truncation or new downloads',
    'sequence_exposure': 'Sequence-only transductive HSP/high-count preprocessing on the fixed corpus, including development/test endpoints; not strict inductive feature construction',
    'hsp_arguments': HSP_ARGS, 'hsp_threads': HSP_THREADS,
    'hsp_order': 'Canonical protein-pair block ordering of native HSP output; HSP records unaltered',
    'prediction_arguments': PREDICT_ARGS,
    'prediction_binary': 'Unmodified serial build; parallel score accumulation has unsynchronized shared matrix updates',
    'score': 'Native nonnegative score printed at upstream default precision; higher is better; retain true zeros and ties',
    'candidate_routing': 'All unlabeled candidate identities passed through -pos; -neg is empty. Native label 1 is a routing marker, never ground truth.',
    'primary_cell': 'C3_test', 'primary_contrast': 'SPRINT minus optimized iPIN',
    'secondary_cells': ['C1_test', 'C2_test'], 'additional_reference': 'baseline iPIN',
    'metric': 'Historical design-weighted P-versus-U concordance; exact half ties; 2000 paired two-endpoint component bootstrap draws',
    'native_qualification_relative_tolerance': 1e-5,
    'no_neural_pretrained_checkpoint': True, 'no_hyperparameter_search': True,
    'test_previously_examined': True, 'test_pairs_read': False, 'test_truth_read': False,
}
