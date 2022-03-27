{
    type: 'basic',
    layer_pooler: import 'layer_pooler/linear.libsonnet',
    seq_encoder: import 'seq_encoder/pass.libsonnet',
    vec_encoder: import 'vec_encoder/boe.libsonnet',
    feedforward: null,
    layer_norm: null,
    intra_slice_dim: 0,
    norm_unit_slice: true,
}
