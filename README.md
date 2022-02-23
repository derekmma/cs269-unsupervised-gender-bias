# Demo for "Unsupervised Discovery of Implicit Gender Bias"

This is a demo repo for the paper "Unsupervised Discovery of Implicit Gender Bias" developed based on the [source code from the author](https://github.com/anjalief/unsupervised_gender_bias).

```
@inproceedings{field-tsvetkov-2020-unsupervised,
    title = "Unsupervised Discovery of Implicit Gender Bias",
    author = "Field, Anjalie  and
      Tsvetkov, Yulia",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.44",
    doi = "10.18653/v1/2020.emnlp-main.44",
    pages = "596--608",
    abstract = "Despite their prevalence in society, social biases are difficult to identify, primarily because human judgements in this domain can be unreliable. We take an unsupervised approach to identifying gender bias against women at a comment level and present a model that can surface text likely to contain bias. Our main challenge is forcing the model to focus on signs of implicit bias, rather than other artifacts in the data. Thus, our methodology involves reducing the influence of confounds through propensity matching and adversarial learning. Our analysis shows how biased comments directed towards female politicians contain mixed criticisms, while comments directed towards other female public figures focus on appearance and sexualization. Ultimately, our work offers a way to capture subtle biases in various domains without relying on subjective human judgements.",
}
```

## Environment Setup

```
# Install conda environment from environment file
conda env create -f env.yml
conda activate cs269
python -m spacy download en_core_web_sm
```

## Training from scratch

1. Download the RtGender dataset from https://nlp.stanford.edu/robvoigt/rtgender/

2. Run `src/run_from_scratch.sh` to create data splits and train models

## Evaluation and Analysis on Saved Models

1. Download models and the training data from [this link](https://drive.google.com/file/d/1FvVTk-FIW__oEl3Nz2ruifhpfIQ_JCsO/view?usp=sharing).

2. Run the following script

```
TOP_DIR="/home/ma/fairness/unsupervised_gender_bias_models" # the path to unzipped tarball of saved models
SUFFIX="facebook_wiki"
DATA_DIR="${TOP_DIR}/${SUFFIX}"
MATCHED_SUFFIX="matched_${SUFFIX}"
SUBS_SUFFIX="subs_name2"
EXTRA_SUFFIX="withtopics"
NO_SUFFIX="notopics"

# Matched, no demotion
python train.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${MATCHED_SUFFIX}.${NO_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.${SUBS_SUFFIX}.${SUFFIX}.txt --save_dir ${DATA_DIR}/matched_notopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_matched_notopics.model --gpu 0 --batch_size 32  --write_attention --epochs 5 --lr 0.0001 --load
python analysis_scripts/write_metrics.py --attention_file ${DATA_DIR}/matched_notopics_${SUBS_SUFFIX}/test.${SUBS_SUFFIX}.${SUFFIX}.rt_gender_${SUFFIX}_matched_notopics.model_attention.txt

No matching, no demotion
python train.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${SUFFIX}.${NO_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.${SUBS_SUFFIX}.${SUFFIX}.txt --save_dir ${DATA_DIR}/baseline_notopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_baseline_notopics.model --gpu 0 --batch_size 32 --write_attention --epochs 5 --lr 0.0001 --load
python analysis_scripts/write_metrics.py --attention_file ${DATA_DIR}/baseline_notopics_${SUBS_SUFFIX}/test.${SUBS_SUFFIX}.${SUFFIX}.rt_gender_${SUFFIX}_baseline_notopics.model_attention.txt

# Matching, demotion
python train_ganlike_multiple_decoders.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${MATCHED_SUFFIX}.${EXTRA_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.${SUBS_SUFFIX}.${SUFFIX}.txt --save_dir ${DATA_DIR}/matched_withtopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_matched_withtopics.model --gpu 0 --batch_size 32  --write_attention --c_steps 3 --t_steps 10 --epochs 3 --lr 0.0001 --load
python analysis_scripts/write_metrics.py --attention_file ${DATA_DIR}/matched_withtopics_${SUBS_SUFFIX}/test.${SUBS_SUFFIX}.${SUFFIX}.rt_gender_${SUFFIX}_matched_withtopics.model_attention.txt

# No matching, demotion
# This command will not run for the congress data. We have excluded the file train.subs_name2.facebook_congress.withtopics.txt from this directory because of its size. We have provided the outputted _attention files in baseline_withtopics_subs_name2
python train_ganlike_multiple_decoders.py --data RT_GENDER --base_path ${DATA_DIR} --train_file train.${SUBS_SUFFIX}.${SUFFIX}.${EXTRA_SUFFIX}.txt --valid_file valid.${SUBS_SUFFIX}.${SUFFIX}.txt --test_file test.${SUBS_SUFFIX}.${SUFFIX}.txt --save_dir ${DATA_DIR}/baseline_withtopics_${SUBS_SUFFIX} --model RNN --model_name rt_gender_${SUFFIX}_baseline_withtopics.model --gpu 0 --batch_size 32  --write_attention --c_steps 3 --t_steps 10 --epochs 3 --lr 0.00099999 --load
python analysis_scripts/write_metrics.py --attention_file ${DATA_DIR}/baseline_withtopics_${SUBS_SUFFIX}/test.${SUBS_SUFFIX}.${SUFFIX}.rt_gender_${SUFFIX}_baseline_withtopics.model_attention.txt
```
