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

Test-time Content Selection (e.g. MCS inference)
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
The outputs of our systems are available -- click the dataset in the table to download (note that after the unzipped files are id_decoded.txt). Note that podcast IDs are according to the order in metadata, and arxiv/pubmed IDs are according to the order in text file in the original data download. If you need to convert these IDs into article_id, refer to [id_lists](https://drive.google.com/file/d/116Hw7aWp13AU3K65Bu0jzxgOMOSplT4B/view?usp=sharing). 

- BART(1k,truncate)

|   Data  | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:-------:|:-------:|:-------:|:-------:|
| [Podcast](https://drive.google.com/file/d/1-jCanm14LUIeozU5GwIYcptiUtxPtUK2/view?usp=sharing) |  26.43  |   9.22  |   18.35 |
|  [arXiv](https://drive.google.com/file/d/1-LzzOsMshwf4NUK-RDLqVa0M81ul4CwR/view?usp=sharing)  |  44.96  |  17.25  |  39.76  |
|  [PubMed](https://drive.google.com/file/d/1-EvvQQ8ijk9cjua8vRMetXzJqorrsxqC/view?usp=sharing) |  45.06  |  18.27  |  40.84  |

- BART(1k,ORC-padrand) + ContentSelection

|   Data  | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:-------:|:-------:|:-------:|:-------:|
| [Podcast](https://drive.google.com/file/d/1-klu-SZV_3JGuk-TORhZRaDGIP1BxalS/view?usp=sharing) |  27.28  |  9.82   |  19.00  |
|  [arXiv](https://drive.google.com/file/d/1-XfjvTFNVP3JszlLP2WNRneP106d8ftQ/view?usp=sharing)  |  47.68  |  19.77  |  42.25  |
|  [PubMed](https://drive.google.com/file/d/1-ACWhTSI2NQJIoTXo05Rcm5q3L39L9m-/view?usp=sharing) |  46.49  |  19.45  |  42.04  |

- LoBART(N=4096,W=1024,ORC-padrand)

|   Data  | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:-------:|:-------:|:-------:|:-------:|
| [Podcast](https://drive.google.com/file/d/1Y85RUahLn0wuwks1w7Fp7BrhASpfVIO3/view?usp=sharing) |  27.36  |  10.04  |  19.33  |
|  [arXiv](https://drive.google.com/file/d/1-K7oEBwIXMybqfPgM6jbn9cMnx5WXfIy/view?usp=sharing)  |  46.59  |  18.72  |  41.24  |
|  [PubMed](https://drive.google.com/file/d/1-Gbqxc4zkxd3CGX84LtqloZjjMIlHF3H/view?usp=sharing) |  47.47  |  20.47  |  43.02  |

- LoBART(N=4096,W=1024,ORC-padrand) + ContentSelection. This is the best configuration reported in the paper.

|   Data  | ROUGE-1 | ROUGE-2 | ROUGE-L |
|:-------:|:-------:|:-------:|:-------:|
| [Podcast](https://drive.google.com/file/d/1f2IpAjrhLU_z5uImB1jmaHHwXkaIPRDR/view?usp=sharing) |  27.81  |  10.30  |   19.61 |
|  [arXiv](https://drive.google.com/file/d/1b1JHD5VkBhhvYjkEKT0YLDsq5CHtviHJ/view?usp=sharing)  |  48.79  |  20.55  |  43.31  |
|  [PubMed](https://drive.google.com/file/d/1pM7SH6UL5HZozhJxzqJrFKxnkiaVYfvq/view?usp=sharing) |  48.06  |  20.96  |  43.56  |

Trained Weights
-----------------------------------------
TRC=Truncate-training, ORC=Oracle-training

|   Model  | Trained on Data |
|:--------:|:------------:|
|LoBART(N=4096,W=1024)\_TRC|[Podcast](https://drive.google.com/file/d/1ZXQ0KP3CHJxWZdK88ebNslV6hlcLgTvA/view?usp=sharing), [arXiv](https://drive.google.com/file/d/1gwX-FCXib5WF9p-dTx-mWIQ3lPL7Psn8/view?usp=sharing), [PubMed](https://drive.google.com/file/d/18TtN-jwW4WadBAUA7P8vcgB6x4BlFHJf/view?usp=sharing)|
|LoBART(N=4096,W=1024)\_ORC|[Podcast](https://drive.google.com/file/d/1JdZpJsgrvjTqA1NqPbiKteNL3CuTjMRC/view?usp=sharing), [arXiv](https://drive.google.com/file/d/1H9Bw2ighKT8LJe-iNK2iLk7lwiQAEli0/view?usp=sharing), [PubMed](https://drive.google.com/file/d/1vvJHKmPI1E284RugWuW_ZaFJO1taG-pb/view?usp=sharing)|
|Hierarchical-Model|[Podcast](https://drive.google.com/file/d/1jF7ydOXVNBj01-aWi18_60H2_D7sFsFo/view?usp=sharing), [arXiv](https://drive.google.com/file/d/1EDZ-XfhDxQUwtbb3y_bH5T7zL_rnUklS/view?usp=sharing), [PubMed](https://drive.google.com/file/d/1yUfY7hEZTQfInYM9BeAdTGsdhz0KRiBa/view?usp=sharing)|

Citation
-----------------------------------------

	@inproceedings{manakul-gales-2021-long,
	    title = "Long-Span Summarization via Local Attention and Content Selection",
	    author = "Manakul, Potsawee  and Gales, Mark",
	    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
	    month = aug,
	    year = "2021",
	    address = "Online",
	    publisher = "Association for Computational Linguistics",
	    url = "https://aclanthology.org/2021.acl-long.470",
	    doi = "10.18653/v1/2021.acl-long.470",
	    pages = "6026--6041",
	}
    
