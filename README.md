Long-Span Summarization
=====================================================
Code for ACL 2021 paper "[Long-Span Summarization via Local Attention and Content Selection](https://arxiv.org/abs/2105.03801)" (previously the title was "Long-Span Dependencies in Transformer-based Summarization Systems").


Requirements
--------------------------------------
- python 3.7
- torch 1.2.0
- transformers (HuggingFace) 2.11.0

Overview
--------------------------------------
1. train/ = training scripts for BART, LoBART, HierarchicalModel (MCS)
2. decode/ = running decoding (inference) for BART, LoBART, MCS-extractive, MCS-attention, MCS-combined
3. data/ = data modules, pre-processing, and sub-directories containing train/dev/test data
4. models/ = defined LoBART & HierarchicalModel
5. traintime_select/ = scripts for processing data for trainining (aka ORACLE methods, pad-rand, pad-lead, no-pad)
6. conf/ = configuration files for training

Pipeline (before training starts)
--------------------------------------
- Download data, e.g. Spotify Podcast, arXiv, PubMed & put in data/
- Basic pre-processing (train/dev/test) & put in data/
- ORACLE processing (train/dev/) & put in data/
- Train HierModel (aka MCS) using data with basic pre-processing
- MCS processing & put in data/
- Train BART or LoBART using data above
- Decode BART or LoBART (note that if MCS is applied, run MCS first i.e. save your data from MCS somewhere and load it)

Data Preparation
--------------------------------------
**Spotify Podcast**
- Download link: https://podcastsdataset.byspotify.com/
- See ```data/podcast_processor.py```
- We recommend splitting the data into chunks such that each chuck contains 10k instance, e.g. id0-id9999 in podcast_set0

**arXiv & PubMed**
- Download link: https://github.com/armancohan/long-summarization
- See ```data/arxiv_processor.py``` (very minimal pre-processing done)

Training BART & LoBART
--------------------------------------
**Training**:

    python train/train_abssum.py conf.txt

**Configurations**:

Setting in conf.txt, e.g. conf/bart_podcast_v0.txt
- **bart_weights** - pre-trained BART weights, e.g. facebook/bart-large-cnn
- **bart_tokenizer** - pre-trained tokenizer, e.g. facebook/bart-large
- **model_name** - model name to be saved
- **selfattn** - full | local
- **multiple_input_span** - maximum input span (multiple of 1024)
- **window_width** - local self-attention width
- **save_dir** - directory to save checkpoints
- **dataset** - podcast
- **data_dir** -  path to data
- **optimizer** - optimzer (currently only adam supported)
- **max_target_len** - maximum target length
- **lr0**  - lr0
- **warmup** - warmup
- **batch_size** - batch_size
- **gradient_accum** - gradient_accum
- **valid_step** - save a checkpoint every ...
- **total_step** - maximum training steps
- **early_stop** - stop training if validaation loss stops improving for ... times
- **random_seed** - random_seed
- **use_gpu** - True | False

Decoding (Inference) BART & LoBART
--------------------------------------
**decoding**:

    python decode/decode_abssum.py \
        --load model_checkpoint
        --selfattn [full|local]
        --multiple_input_span INT
        --window_width INT
        --decode_dir output_dir
        --dataset [podcast|arxiv|pubmed]
        --datapath path_to_dataset
        --start_id 0
        --end_id 1000
        [--num_beams NUM_BEAMS]
        [--max_length MAX_LENGTH]
        [--min_length MIN_LENGTH]
        [--no_repeat_ngram_size NO_REPEAT_NGRAM_SIZE]
        [--length_penalty LENGTH_PENALTY]
        [--random_order [RANDOM_ORDER]]
        [--use_gpu [True|False]]

Training Hierarchical Model
--------------------------------------
    python train/train_hiermodel.py conf.txt

 see conf/hiermodel_v1.txt for an example of config file

Training-time Content Selection
--------------------------------------
 **step1**: running oracle selection {pad|nopad} per sample/instance

    python traintime_select/oracle_select_{pad|nopad}.py

**step2**: combine all test samples into one file

    python traintime_select/oracle_select_combine.py

See traintime_select/README.md for more information about arguments.

Test-time Content Selection (Running MCS)
--------------------------------------
 **step1**: running decoding for get attention & extractive labelling predictions (per sample)

    python decode/inference_hiermodel.py

**step2**: combine all test samples into one file

    python decode/inference_hiermodel_combine.py

See decode/README.md for more information about arguments.

Analysis
-----------------------------------------
Requires package ```pytorch_memlab```. Args: localattn = True if LoBART, False if BART, X = max. input length, Y = max. target length, W = local attention width (if localattn == True), B = batch size.

**Memeory (BART & LoBART)**

    python analysis/memory_inspect.py localattn X Y W B

**Time (BART & LoBART)**

    python analysis/speed_inspect.py localattn X Y W B num_iterations


Results using this repository
-----------------------------------------
- Simple fine-tuning vanilla BART(1k) on truncated data

|   Data  | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:-------:|:-------:|:-------:|:-------:|
| Podcast |  26.43  |   9.22  |   18.35 |
|  arXiv  |  44.96  |  17.25  |  39.76  |
|  PubMed |  45.06  |  18.27  |  40.84  |

- Our best results using LoBART(N=4096,W=1024) + MCS

|   Data  | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:-------:|:-------:|:-------:|:-------:|
| Podcast |  27.81  |  10.30  |   19.61 |
|  arXiv  |  48.79  |  20.55  |  43.31  |
|  PubMed |  48.06  |  20.96  |  43.56  |

Trained Weights
-----------------------------------------
Links to Google Drive to be added!
