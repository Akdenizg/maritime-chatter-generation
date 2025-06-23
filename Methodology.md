# Methodology

This section outlines the methodology employed in the master thesis for generating and evaluating realistic maritime radio chatter using Large Language Models (LLMs). The core approach involves leveraging the Self-Instruct method to augment existing datasets, followed by fine-tuning LLMs using Low-Rank Adaptation (LoRA) and prompt tuning.

4.1 Datasets

The study utilizes several datasets to facilitate the generation and analysis of maritime scenarios:

- GSHHG (Global Self-consistent, Hierarchical, High-Resolution Geography Database): Used for geospatial analysis, specifically to identify land and maritime areas and calculate distances to shorelines. This dataset provides high-resolution shoreline data.

- GeoNames: Provides geographical names and related information, including coordinates, feature codes (e.g., for bays, seas, ports, islands) and country codes. This is crucial for identifying and contextualizing maritime locations and nearby land features.

- Marine Cadastre: This dataset, along with the Danish Maritime Authority AIS Dataset, provides vessel information, including vessel names, MMSI (Maritime Mobile Service Identity), call signs and vessel types. This data is preprocessed to clean and standardize vessel names and call signs.

- Danish Maritime Authority AIS Dataset: Similar to Marine Cadastre, this dataset provides AIS (Automatic Identification System) data for vessels, which is used to enrich the vessel information for scenario generation.

4.2 Data Preprocessing

Data preprocessing involves cleaning and standardizing the vessel data from Marine Cadastre and Danish Maritime Authority AIS Dataset. This includes removing special characters from vessel names and call signs and converting vessel types to a consistent format.

4.3 LLM used in the study

The study primarily uses the Llama 3.1 8B model as the base Large Language Model for generating maritime radio chatters. This model is chosen for its capabilities in text generation and its suitability for fine-tuning.

4.4 Self-Instruct for Maritime Distress Call Training Dataset Generation

The Self-Instruct method is a key component of the methodology, used to augment a manually created dataset of radio chatter instances. This process involves three main steps:

4.4.1 Context Generation

This step involves generating realistic maritime scenarios that serve as context for the radio chatters. This includes:

- Random Coordinate Generation: Generating random maritime coordinates (latitude and longitude) ensuring they are in open water (not over land) using the GSHHG dataset.

- Vessel Selection: Randomly selecting a vessel from the preprocessed vessel datasets, including its name, MMSI, call sign and type.

- Geospatial Analysis: Identifying the closest land features, water bodies (bays, seas, oceans) and significant places (ports, harbors, islands) to the generated maritime coordinate using the GeoNames dataset. This also involves calculating distances and compass directions to these features.

- Task-Specific Context: Incorporating specific elements into the context based on the type of distress call (e.g., for a collision, details about a collided vessel are added).

4.4.2 Instance Generation

Once a context is generated, the LLM is prompted to generate radio chatter instances based on this context. This involves:

- Prompt Engineering: Crafting specific prompts that guide the LLM to generate realistic and relevant distress calls, incorporating details from the generated context.

- LLM Inference: Using the Llama 3.1 8B model to generate multiple radio chatter responses for each context.

4.4.3 Filtering

After instance generation, a rigorous filtering process is applied to ensure the quality and relevance of the generated chatters. This involves checking for:

- Format Correctness: Ensuring the generated chatter adheres to the expected radio communication format.

- Information Accuracy: Verifying that key information from the context (e.g., vessel name, coordinates, distress type) is accurately reflected in the chatter.

- Logical Coherence: Assessing the overall logical flow and realism of the conversation.

- Uniqueness: Filtering out chatters that are too similar to existing seed instances or other generated instances to ensure diversity in the dataset.

- More information about the filters can be found in Filters.md.

4.5 Fine-tuning the LLM with LoRA

After generating and filtering the synthetic dataset, the Llama 3.1 8B model is fine-tuned using Low-Rank Adaptation (LoRA). LoRA is a parameter-efficient fine-tuning technique that injects small, trainable matrices into the transformer architecture, significantly reducing the number of trainable parameters and computational cost while maintaining performance. This allows for efficient adaptation of the LLM to the specific domain of maritime radio chatter.

4.6 Prompt Tuning the LLM

In addition to LoRA, prompt tuning is also employed. Prompt tuning involves optimizing a small set of

virtual tokens that are prepended to the input prompt, rather than directly modifying the model's weights. This technique is even more parameter-efficient than LoRA and can be effective for adapting LLMs to new tasks.

4.7 Evaluation of the Distress Calls Generated by the Fine-tuned Adapters

Four evaluation metrics are proposed to assess the quality of the generated maritime distress calls:

- Correctness of the Format: Evaluates if the generated chatter follows the Standard Maritime Communication Phrases (SMCP) and other expected radio communication protocols.

- Information Accuracy: Measures how well the generated chatter incorporates and accurately reflects the information provided in the initial context (e.g., vessel details, coordinates, distress type).

- Logical Coherence: Assesses the naturalness and logical flow of the conversation within the generated chatter, ensuring it makes sense in a real-world maritime distress scenario.

Uniqueness: Quantifies the diversity of the generated chatters, ensuring that the model is not simply memorizing and reproducing training examples. This is often measured using metrics like ROUGE-L similarity against seed and training data, as well as among the generated samples themselves.

