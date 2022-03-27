# Retriever
local retriever_type = 'basic';
local use_shift_embeds = false;
local similarity_metric = 'cosine';
local neighborhood_size = 4;

{
    retriever: {
        type: 'basic',
        task_name: 'bucc',
        use_shift_embeds: use_shift_embeds,
        similarity_metric: similarity_metric,
        neighborhood_size: neighborhood_size,
    },
    dataset_reader: {
        type: 'sentence_encoder',
        tokenizer: import '../xlmr/tokenizer.libsonnet',
        token_indexers: import '../xlmr/token_indexers.libsonnet',
        data_dir: 'data/bucc2018',
        all_lang_names: {
            src: ['de', 'fr', 'ru', 'zh',],
            trg: ['en',],
        },
        lazy: false,
    },
    batch_size: 128,
    splits: ['dev',],
}
