Training-time Content Selection
========================================
- **Oracle (ORC)**: Filter (pre-process) data such that the filtered input maximises the ROUGE score. As the target reference is required, if used at test time, it's cheating. Because some of the filtered inputs will be aggressviely pruned, e.g. shorter than ABSSUM\_MAX\_LEN, our work propose:
	- **ORC\_nopad**: no padding is performed.
	- **ORC\_padrand**: pad by random unselected sentences.
	- **ORC\_padlead**: pad by lead unselected sentences.

Note that after padding, we keep the original sentence order.

### Step1: Oracle filtering per sample

- **Usage (no-padding)**:

		python traintime_select/oracle_select_nopad.py \
			--dataset [podcast|arxiv|pubmed]
			--output_dir temp_dir_to_store_filtered_inputs
			--datapath path_to_data
			--max_abssum_len max_filtered_input_length
			--start_id INT
			--end_id INT
			--random_order BOOL

	e.g. to filter podcast training partition 0 with ORC\_nopad:

		python traintime_select/oracle_select_nopad.py --dataset podcast --output_dir data/traintime_0/ --datapath data/podcast_set0.bin --max_abssum_len 1024 --start_id 0 --end_id 10000 --random_order False

- **Usage (padding)**:

		python traintime_select/oracle_select_pad.py \
			--dataset [podcast|arxiv|pubmed]
			--output_dir temp_dir_to_store_filtered_inputs
			--datapath path_to_data
			--oracle_type [padrand|padlead]
			--max_abssum_len max_filtered_input_length
			--start_id INT
			--end_id INT
			--random_order BOOL

	e.g. to filter podcast training parition 0 with ORC\_padrand:

		python traintime_select/oracle_select_pad.py --dataset podcast --output_dir data/traintime_1/ --datapath data/podcast_set0.bin --oracle_type padrand --max_abssum_len 4096 --start_id 0 --end_id 10000 --random_order False

### Step2: Combine all samples into one file

		python traintime_select/oracle_select_combine.py \
			--dataset [podcast|arxiv|pubmed]
			--output_dir temp_dir_in_step1
			--original_datapath datapath_in_step1
			--filtered_datapath filtered_datapath

e.g.

		python traintime_select/oracle_select_combine.py --dataset podcast --output_dir data/traintime_1/ --original_datapath data/podcast_set0.bin --filtered_datapath data/podcast_set0.padrand.4096.bin
