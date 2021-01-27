Long-Span Summarization
=====================================================
Requirements
--------------------------------------
- python 3.7
- PyTorch 1.2.0
- transformers (HuggingFace) 2.11.0

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

Decoding (Inference)
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
