# The aligner model.
local model_type = 'cycle_aligner';
local vector_similarity = import '../similarity/dot_product_similarity.libsonnet';
local cycle = [
    {
        input_dim: 1024,
        num_layers: 1,
        hidden_dims: [1024,],
        activations: ['linear',],
        dropout: [0.1,],
    },
    {
        input_dim: 1024,
        num_layers: 1,
        hidden_dims: [1024,],
        activations: ['linear',],
        dropout: [0.1,],
    },
    ];

# The data loading configuration.
local batch_size = 32;


{
    trainer: {
        type: 'gradient_descent',
        optimizer: {
            type: 'adam',
            lr: 0.001,
            amsgrad: false,
        },
        num_epochs: 32,
        patience: 1,
        cuda_device: null,
        use_amp: true,
    },
    model: {
        type: model_type,
        text_field_embedder: import '../xlmr/text_field_embedder.libsonnet',
        lens: null,
        vector_similarity: vector_similarity,
        criterion: null,
        cycle: cycle,
    },
    data_loader: {
        batch_size: batch_size,
        shuffle: false,
    },
    validation_data_loader: {
        batch_size: batch_size,
        shuffle: false,
    },
    dataset_reader: {
        type: 'xlm_para',
        tokenizer: import '../xlmr/tokenizer.libsonnet',
        token_indexers: import '../xlmr/token_indexers.libsonnet',
        pair_names: null,
        lazy: true,
        lazy_shuffle: true,
    },
    train_data_path: 'data/xlm/para/train',
    validation_data_path: 'data/xlm/para/valid',
    test_data_path: 'data/xlm/para/test',
    datasets_for_vocab_creation: ['validation',],
}
