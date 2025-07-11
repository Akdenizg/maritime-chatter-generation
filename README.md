# maritime-chatter-generation

The maritime industry is the backbone of global trade, with more than 80% of trade volume transported by sea. The safety and efficiency of maritime trade suffer from disasters, mainly due to human errors. Automatic Speech Recognition (ASR) systems can help, but there is no open-source ASR dataset for maritime radio communication. Creating such a dataset manually is time-consuming, so this project uses a Large Language Model (LLM) to generate maritime chatters and a Text-to-Speech model to create audio data for scalable dataset creation.

Llama 3.1 8B is used to augment a manually created dataset of seed radio chatter instances, using an adaptation of the [Self-Instruct method](https://doi.org/10.48550/arXiv.2212.10560). The model is then fine-tuned with the augmented dataset using Low-Rank Adaptation (LoRA) and prompt-tuning. Evaluation metrics include format correctness, information accuracy, and uniqueness.

See `Methodology.md` for details.

---

## Getting Started

- Download the [GSHHG dataset](https://www.soest.hawaii.edu/pwessel/gshhg/). Put `"GSHHS_shp/f/GSHHS_f_L1.shp"` in the `GSHHS_dataset` folder.
- Download [`allCountries.zip`](https://download.geonames.org/export/dump/) from GeoNames. Extract and put `allCountries.txt` in the `all_countries` folder.
- Install dependencies with requirements.txt

---

## Folder Structure

- **all_countries/**: Should contain `allCountries.txt` from GeoNames.
- **data/**: Vessel dataset and filter keywords.
- **evaluation/**: Stores evaluation metrics/results.
  - `evaluation/{task_name}`: Evaluation of chatters generated by LoRA adapters.
  - `evaluation/prompt_tuning/{task_name}`: For prompt tuning adapters.
- **experiments/**: Synthetic training chatters. Includes example chatters generated by the generates_instances.py
- **GSHHS_dataset/**: GSHHS shapefiles.
- **models/**: Trained LoRA and prompt-tuning adapters.
  - `models/{task_name}`: LoRA adapters.
  - `models/prompt_tuning/{task_name}`: Prompt tuning adapters.
- **prompts/**: Prompt templates for Self-Instruct.
- **scripts/**: All scripts and notebooks.
- **seed_outputs/**: Seed instances and inspection files (per SMCP category).
- **synthetic_chatters/**: 100 synthetic chatters generated by adapters. Includes example chatters generated by the run_models.py and run_models_prompt_tuning.py
  - `synthetic_chatters/{task_name}`: LoRA adapters.
  - `synthetic_chatters/prompt_tuning/{task_name}`: Prompt tuning adapters.

---

## Script Overview and Usage

Below is the recommended order. In each script, **edit the file paths and parameters at the top before running**.

1. **geo_test.py**  
   - *Purpose*: Utility for context generation and scenario creation.
   - *Edit*: Paths to shapefile, GeoNames, vessel data.
   - *Run order*: Used as a utility, called by other scripts.

2. **generate_instances.py**  
   - *Purpose*: Self-Instruct pipeline. Generates and filters synthetic chatters.
   - *Edit*: Paths (`land_shapefile`, `geonames_data_path`, `vessel_data_path`), `task_name`, `prompt_file`, LLM path, seed output path, number of instances, etc.
   - *Run order*: **First main script to run** (after data/seed prep).
   - *Output*: Synthetic chatters in `experiments/{task_name}`.

3. **lora_finetune.ipynb**  
   - *Purpose*: Fine-tune Llama 3.1 8B using LoRA. Modification of the unsloth colab notebook.
   - *Edit*: `model_dir` (base LLM), `chatter_path` (from step 2), `task_name`, hyperparameters.
   - *Run order*: After `generate_instances.py`.
   - *Output*: LoRA adapters in `models/{task_name}`.

4. **prompt_tuning.ipynb**  
   - *Purpose*: Fine-tune base model with prompt-tuning.
   - *Edit*: `model_dir`, `chatter_path`, `task_name`, `prompt_file`, hyperparameters.
   - *Run order*: After `generate_instances.py` (alternative to LoRA).
   - *Output*: Prompt-tuning adapters in `models/prompt_tuning/{task_name}`.

5. **run_models.py**  
   - *Purpose*: Generate 100 new chatters using LoRA adapters.
   - *Edit*: `hyperparameters_path` (from LoRA output), file paths.
   - *Run order*: After LoRA fine-tune.
   - *Output*: Results in `synthetic_chatters/{task_name}`.

6. **run_models_prompt_tuning.py**  
   - *Purpose*: Generate 100 new chatters using prompt-tuning adapters.
   - *Edit*: `hyperparameters_path` (from prompt-tuning output), `prompt_file`, file paths.
   - *Run order*: After prompt-tuning.
   - *Output*: Results in `synthetic_chatters/prompt_tuning/{task_name}`.

7. **inspect_model.py**  
   - *Purpose*: Evaluate chatters generated by LoRA adapters.
   - *Edit*: `keywords_path`, `ship_data_path`, `land_shapefile`, `geonames_data_path`, `seed_path`, `synthetic_chatter_path`, `save_dir`.
   - *Run order*: After generating synthetic chatters with LoRA.
   - *Output*: Evaluation results in `evaluation/{task_name}`.

8. **inspect_model_prompt_tuning.py**  
   - *Purpose*: Evaluate chatters generated by prompt-tuned adapters.
   - *Edit*: (Same as above, use prompt-tuned outputs).
   - *Run order*: After generating synthetic chatters with prompt-tuning.
   - *Output*: Evaluation results in `evaluation/prompt_tuning/{task_name}`.

---

## Order of Execution

1. **Prepare data** (GSHHG, GeoNames, seed chatters).
2. **Run** `generate_instances.py` for each SMCP category needed.
3. **Fine-tune**:  
   - LoRA: `lora_finetune.ipynb`  
   - Prompt-tuning: `prompt_tuning.ipynb`
4. **Generate Chatters**:  
   - LoRA: `run_models.py`  
   - Prompt-tuning: `run_models_prompt_tuning.py`
5. **Evaluate**:  
   - LoRA: `inspect_model.py`  
   - Prompt-tuning: `inspect_model_prompt_tuning.py`
