# The aligner model.
local model_type = 'gan_aligner';
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
local critic = {
    input_dim: 1024,
    num_layers: 1,
    hidden_dims: [1,],
    activations: ['linear',],
    dropout: [0.1,],
};
local critic_criterion = import '../criterion/max_difference_criterion.libsonnet';
local aggregate_criterion = {
    type: 'sum',
    lamda_weights: [1,],
};

# The training configuration
local critic_num_steps = 1;

# The data loading configuration.
local batch_size = 32;


{
    trainer: {
        type: 'multiplayer',
        optimizer :{
            type: 'multiplayer',
            player_names: ['critic', 'generator',],
            nums_steps: {
                critic: critic_num_steps,
                generator: 1,
            },
            optimizers: {
                critic: {
                    type: 'adam',
                    lr: 0.001,
                    amsgrad: false,
                },
                generator: {
                    type: 'adam',
                    lr: 0.001,
                    amsgrad: false,
                },
            },
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
        critic: critic,
        critic_criterion: critic_criterion,
        aggregate_criterion: aggregate_criterion,
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
        lazy_shuffle: false,
    },
    train_data_path: 'data/xlm/para/train',
    validation_data_path: 'data/xlm/para/valid',
    test_data_path: 'data/xlm/para/test',
    datasets_for_vocab_creation: ['validation',],
}
