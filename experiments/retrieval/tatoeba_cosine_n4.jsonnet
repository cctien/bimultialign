# Retriever
local retriever_type = 'basic';
local use_shift_embeds = false;
local similarity_metric = 'cosine';
local neighborhood_size = 4;

{
    retriever: {
        type: 'basic',
        task_name: 'tatoeba',
        use_shift_embeds: use_shift_embeds,
        similarity_metric: similarity_metric,
        neighborhood_size: neighborhood_size,
    },
    dataset_reader: {
        type: 'sentence_encoder',
        tokenizer: import '../xlmr/tokenizer.libsonnet',
        token_indexers: import '../xlmr/token_indexers.libsonnet',
        data_dir: 'data/tatoeba',
        all_lang_names: {
            src: ['afr', 'amh', 'ang', 'ara', 'arq', 'arz', 'ast', 'awa', 'aze', 'bel', 'ben', 'ber',
                  'bos', 'bre', 'bul', 'cat', 'cbk', 'ceb', 'ces', 'cha', 'cmn', 'cor', 'csb', 'cym',
                  'dan', 'deu', 'dsb', 'dtp', 'ell', 'epo', 'est', 'eus', 'fao', 'fin', 'fra', 'fry',
                  'gla', 'gle', 'glg', 'gsw', 'heb', 'hin', 'hrv', 'hsb', 'hun', 'hye', 'ido', 'ile',
                  'ina', 'ind', 'isl', 'ita', 'jav', 'jpn', 'kab', 'kat', 'kaz', 'khm', 'kor', 'kur',
                  'kzj', 'lat', 'lfn', 'lit', 'lvs', 'mal', 'mar', 'max', 'mhr', 'mkd', 'mon', 'nds',
                  'nld', 'nno', 'nob', 'nov', 'oci', 'orv', 'pam', 'pes', 'pms', 'pol', 'por', 'ron',
                  'rus', 'slk', 'slv', 'spa', 'sqi', 'srp', 'swe', 'swg', 'swh', 'tam', 'tat', 'tel',
                  'tgl', 'tha', 'tuk', 'tur', 'tzl', 'uig', 'ukr', 'urd', 'uzb', 'vie', 'war', 'wuu',
                  'xho', 'yid', 'yue', 'zsm',],
            trg: ['eng',],
        },
        lazy: false,
    },
    zero_threshold: true,
    batch_size: 128,
    splits: ['',],
}
