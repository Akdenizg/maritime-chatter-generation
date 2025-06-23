# Methodology

This section outlines the methodology employed in the master thesis for generating and evaluating realistic maritime radio chatter using Large Language Models (LLMs). The core approach involves leveraging the Self-Instruct method to augment existing datasets, followed by fine-tuning LLMs using Low-Rank Adaptation (LoRA) and prompt tuning.

## Datasets

The study utilizes several datasets to facilitate the generation and analysis of maritime scenarios:

- GSHHG (Global Self-consistent, Hierarchical, High-Resolution Geography Database): Used for geospatial analysis, specifically to identify land and maritime areas and calculate distances to shorelines. This dataset provides high-resolution shoreline data.

- GeoNames: Provides geographical names and related information, including coordinates, feature codes (e.g., for bays, seas, ports, islands) and country codes. This is crucial for identifying and contextualizing maritime locations and nearby land features.

- Marine Cadastre: This dataset, along with the Danish Maritime Authority AIS Dataset, provides vessel information, including vessel names, MMSI (Maritime Mobile Service Identity), call signs and vessel types. This data is preprocessed to clean and standardize vessel names and call signs.

- Danish Maritime Authority AIS Dataset: Similar to Marine Cadastre, this dataset provides AIS (Automatic Identification System) data for vessels, which is used to enrich the vessel information for scenario generation.

## Data Preprocessing

From both datasets, only the vessel names, types, MMSI numbers and call signs are taken. Marine Cadastre dataset has vessel type codes instead of the vessel type itself. These codes are mapped into the vessel types from the "Frequently Asked Questions" section of their website. Vessels without a vessel name are removed from both datasets, as the vessel name must be stated in maritime distress calls. Null vessel types were converted to "Other" in both datasets. Details of the Marine Cadastre and Danish Maritime Authority AIS data after the preprocessing are shown in the tables 1 and 2 respectively. A combination of Marine Cadastre data between 01.01.2023 and 30.06.2024 and Danish Maritime Authority AIS data between 26.07.2024 and 102.2025 is used. Some vessel types from the Danish Maritime Authority AIS data which are different from the Marine Cadastre vessel types were converted to their matching vessel types from the Marine Cadastre data for consistency. Conversions are shown in table 3. Vessel types from both datasets that are not one of the vessel types in table 1 were changed to "Other". Call signs with the value "UNKNOWN" were converted to null. Repetitive whitespaces
were removed from the vessel names. Vessel names with "NO NAME" were excluded from the extraction. Special characters were removed from call sign values and vessel names. Lastly, two datasets were combined and the duplicate entries were dropped.
Since pleasure crafts and sailing vessels are dominant in the marine cadastre data, 5000 pleasure crafts and 5000 sailing vessels were randomly sampled from it and added to the combination to prevent a biased vessel dataset. Final vessel dataset has 58855 vessels. Extracted vessel data can be found at data/ship_data_dk_us. Special characters and words in parantheses or bracktes are removed from the names of the geographical entities from GeoNames.

## LLM used in the study

The study uses the Llama 3.1 8B model as the base Large Language Model for generating maritime radio chatters.

## Self-Instruct for Maritime Distress Call Training Dataset Generation

The Self-Instruct method is a key component of the methodology, used to augment a manually created dataset of radio chatter instances. This process involves three main steps:

### Context Generation

This step involves generating realistic maritime scenarios that serve as context for the radio chatters. This includes:

- Random Coordinate Generation: Generating random maritime coordinates (latitude and longitude) ensuring they are in open water (not over land) using the GSHHG dataset.

- Vessel Selection: Randomly selecting a vessel from the preprocessed vessel datasets, including its name, MMSI, call sign and type.

- Geospatial Analysis: Identifying the closest land features, water bodies (bays, seas, oceans) and significant places (ports, harbors, islands) to the generated maritime coordinate using the GeoNames dataset. This also involves calculating distances and compass directions to these features.

- Task-Specific Context: Incorporating specific elements into the context based on the type of distress call (e.g., for a collision, details about a collided vessel are added).

### Instance Generation

Once a context is generated, the LLM is prompted to generate radio chatter instances based on this context. This involves:

- Prompt Engineering: Crafting specific prompts that guide the LLM to generate realistic and relevant distress calls, incorporating details from the generated context.

- LLM Inference: Using the Llama 3.1 8B model to generate multiple radio chatter responses for each context.

### Filtering

After instance generation, a rigorous filtering process is applied to ensure the quality and relevance of the generated chatters. This involves checking for:

- Format Correctness: Ensuring the generated chatter adheres to the expected radio communication format.

- Information Accuracy: Verifying that key information from the context (e.g., vessel name, coordinates, distress type) is accurately reflected in the chatter.

- Logical Coherence: Assessing the overall logical flow and realism of the conversation.

- Uniqueness: Filtering out chatters that are too similar to existing seed instances or other generated instances to ensure diversity in the dataset.

- More information about the filters can be found in Filters.md.

## Fine-tuning the LLM with LoRA

After generating and filtering the synthetic dataset, the Llama 3.1 8B model is fine-tuned using Low-Rank Adaptation (LoRA). LoRA is a parameter-efficient fine-tuning technique that injects small, trainable matrices into the transformer architecture, significantly reducing the number of trainable parameters and computational cost while maintaining performance. This allows for efficient adaptation of the LLM to the specific domain of maritime radio chatter.

## Prompt Tuning the LLM

In addition to LoRA, prompt tuning is also employed. Prompt tuning involves optimizing a small set of virtual tokens that are prepended to the input prompt, rather than directly modifying the model's weights. This technique is even more parameter-efficient than LoRA and can be effective for adapting LLMs to new tasks.

## Evaluation of the Distress Calls Generated by the Fine-tuned Adapters

Three evaluation metrics are proposed to assess the quality of the generated maritime distress calls:

- Correctness of the Format: Evaluates if the generated chatter follows the Standard Maritime Communication Phrases (SMCP) and other expected radio communication protocols.

- Information Accuracy: Measures how well the generated chatter incorporates and accurately reflects the information provided in the initial context (e.g., vessel details, coordinates, distress type).

- Uniqueness: Quantifies the diversity of the generated chatters, ensuring that the model is not simply memorizing and reproducing training examples. This is often measured using metrics like ROUGE-L similarity against seed and training data, as well as among the generated samples themselves.

