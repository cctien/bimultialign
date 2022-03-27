# The aligner model.
local model_type = 'sentence_encoder';
local vector_similarity = import '../similarity/dot_product_similarity.libsonnet';

# The data loading configuration.
local batch_size = 1;


{
    trainer: {
        type: 'gradient_descent',
        optimizer: {
            type: 'adam',
            lr: 0.001,
            amsgrad: false,
        },
        num_epochs: 1,
        patience: 1,
        cuda_device: null,
    },
    model: {
        type: model_type,
        text_field_embedder: import '../xlmr/text_field_embedder.libsonnet',
        lens: null,
        vector_similarity: vector_similarity,
        feedforward: {
            input_dim: 1024,
            num_layers: 1,
            hidden_dims: [1,],
            activations: ['linear',],
        },
        criterion: null,
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
        type: 'sentence_encoder',
        tokenizer: import '../xlmr/tokenizer.libsonnet',
        token_indexers: import '../xlmr/token_indexers.libsonnet',
    },
    train_data_path: 'data/tatoeba/tatoeba.deu-eng.eng',
    validation_data_path: 'data/tatoeba/tatoeba.deu-eng.eng',
    test_data_path: 'data/tatoeba/tatoeba.deu-eng.eng',
    datasets_for_vocab_creation: ['validation',],
}
