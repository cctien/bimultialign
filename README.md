# Bilingual alignment transfers to multilingual alignment for unsupervised parallel text mining

## Paper

Implementation of the paper of Chih-chan Tien and Shane Steinert-Threlkeld. 2022. ‘Bilingual alignment transfers to multilingual alignment for unsupervised parallel text mining’.

## Libraries

Python libraries listed in `requirements.txt` are used.

## Data

### Training corpora

The bilingual corpora may be downloaded through scripts in [the repository of XLM](https://github.com/facebookresearch/XLM), and then moved to `data/xlm/para`.
For example, the command below can be used to get parallel texts in English and German with scripts from the [the repository of XLM](https://github.com/facebookresearch/XLM).

```shell
# Use the script `get-data-para.sh` of XLM
./get-data-para.sh de-en

```

### Evaluation datasets

The evaluation dataset may be downloaded with `scripts/get_data.sh`.

```shell
./get_data.sh
```

## Training and evaluation

The shell commands below can be used to replicate the main results in Table 2 and Table 3.

### Unsupervised model (with adversarial and cycle losses)

```shell
python interlens pipeline \
    --param_path "aligner/gancycle_aligner.jsonnet" \
    --lens "softmaxlinear pass boe linear" \
    --criterion_variants "triplet_ranking" \
    --margins "0.2" \
    --nums_random_samples "1" \
    --batch_size "32" --epoch_size "524288" \
    --critic_criterion "max_difference" \
    --cycle_loss_lambda "10" \
    --critic_num_steps "2" \
    --pivot_languages "de"

```

### Bilingually-supervised model

```shell
python interlens pipeline \
    --param_path "aligner/para_aligner.jsonnet" \
    --lens "softmaxlinear pass boe linear" \
    --criterion_variants "triplet_ranking" \
    --margins "0.0" \
    --nums_random_samples "0" \
    --batch_size "32" --epoch_size "524288" \
    --pivot_languages "de"

```
