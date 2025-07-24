import os
from matplotlib import pyplot as plt
from unsloth import FastLanguageModel
import json
import random
import datetime
import geo_test
import re
import time

from rouge_score import rouge_scorer
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnPhrase(StoppingCriteria):
    def __init__(self, tokenizer, stop_phrase):
        self.tokenizer = tokenizer
        self.stop_phrase = stop_phrase

    def __call__(self, input_ids, scores, **kwargs):
        decoded_output = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return self.stop_phrase in decoded_output

class DatasetCreator:
    def __init__(self, land_shapefile, geonames_data_path, seed_outputs_path, keywords_path, task_name, prompt_file, temperature, top_k, max_new_tokens, model_name, do_sample, 
                 rouge_l_score, num_data_to_create, vessel_data_path, save_dir):
        
        self.max_seq_length = 20480
        self.dtype = None
        self.load_in_4bit = False
        self.land_shapefile = land_shapefile
        self.geonames_data_path = geonames_data_path
        self.seed_outputs_path = seed_outputs_path
        self.vessel_data_path = vessel_data_path
        self.maritime_analysis = geo_test.MaritimeAnalysis(self.land_shapefile, self.geonames_data_path, self.vessel_data_path, task_name)
        self.single_digits = self.maritime_analysis.single_digits
        self.tens = self.maritime_analysis.tens
        self.teens = self.maritime_analysis.teens
        self.task_name = task_name
        self.prompt_file = prompt_file
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.do_sample = do_sample
        self.num_data_to_create = num_data_to_create
        self.seed_tasks_data = self.read_json(self.seed_outputs_path)
        self.rouge_l_score = rouge_l_score
        self.keywords_path = keywords_path
        self.save_dir = save_dir
        self.non_digits = ['twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety','hundred','thousand','million']

    def read_json(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def words_to_number(self, words):
        words = words.replace("-", " ").split()
        total = 0
        current = 0
        previous = None  # To track the type of the previous word

        for word in words:
            if word in self.single_digits:
                digit = self.single_digits.index(word)
                if previous == 'tens':
                    # If the previous word was a tens word, add the single digit
                    current += digit
                else:
                    # Otherwise, treat it as digit-by-digit
                    current = current * 10 + digit
                previous = 'single_digit'

            elif word in self.tens:
                current += self.tens.index(word) * 10
                previous = 'tens'

            elif word in self.teens:
                current += 10 + self.teens.index(word)
                previous = 'teens'

            elif word == "hundred":
                current *= 100
                previous = 'hundred'

            elif word == "thousand":
                total += current * 1000
                current = 0
                previous = 'thousand'

            elif word.isdigit():
                current += int(word)
                previous = 'digit'

            else:
                pass

        total += current
        return total

    def calculate_rouge_l(self, reference_text, generated_text):
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = scorer.score(reference_text, generated_text)
        return scores['rougeL'].fmeasure    

    def inference(self, model, tokenizer, user_prompt, max_new_tokens=400, **kwargs):
        stopping_criteria = StoppingCriteriaList([StopOnPhrase(tokenizer, "Context 7:")])
        inputs = tokenizer([user_prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=0.9,
                temperature=temperature,
                min_length=None,
                use_cache=True,
                top_k=top_k,
                repetition_penalty=1,
                length_penalty=1,
                stopping_criteria=stopping_criteria,
                **kwargs
            )
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return output_text

    def create_dataset(self):
        seed_outputs = []

        new_tasks_data = {
            "task_name": self.task_name,
            "4bit": self.load_in_4bit,
            "top_k": top_k,
            "top_p": temperature,
            "no_seed_outputs": 0,
            "max_new_tokens": max_new_tokens,
            "total_rejected_results": 0,
            "rejected_results": {},
            f"{self.task_name}": {},
            "used_mmsi_list": [],
        }

        new_outputs = []

        scenario_inspect_list = []

        with open(prompt_file, "r") as f:
            base_prompt = "\n".join(f.readlines())

        # Load model and tokenizer using unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit
        )

        FastLanguageModel.for_inference(model)

        num_compass = 0
        num_closest_port = 0
        num_closest_harbor = 0
        num_country = 0
        num_distance_to_closest_place = 0
        num_distance_to_closest_port = 0
        num_distance_to_closest_harbor = 0
        num_water_body = 0
        num_closest_place = 0

        rejected_results = {
            'paranthesis':[],
            'brackets':[],
            'mayday':[],
            'incomplete': [],
            'blank': [],
            'wrong_category': [],
            'unknown':[],
            'vessel_name_not_after_mayday': [],
            'no_position_indicated': [],
            'no_coast_guard': [],
            'duplicate_sentences': [],
            'vessel_name': [],
            'vessel_MMSI': [],
            'vessel_call_sign': [],
            'vessel_type': [],
            'vessel_coordinate_dms': [],
            'compass': [],
            'hallucinated_mmsi': [],
            'hallucinated_call_sign': [],
            'digit_by_digit': [],
            'false_cargo_logic': [],
            'hallucinated_vessel_type': [],
            'distance_to_nearest_place': [],
            'distance_to_nearest_port': [],
            'distance_to_nearest_harbor': [],
            'non-unique': [],
            'both_port_and_harbor': [],
        }
        if self.task_name == "reporting_collision":
            rejected_results["collided_vessel_name"] = []
            rejected_results["collided_vessel_type"] = []

        keywords = self.read_json(self.keywords_path)
        
        for task in self.seed_tasks_data:
            if task == self.task_name:
                for instance in self.seed_tasks_data[task]["instances"]:
                    instance["output"] = "".join(instance["output"])
                    seed_outputs.append(instance)
        new_tasks_data["no_seed_outputs"] += len(self.seed_tasks_data[task_name]["instances"])

        used_mmsi_list = []
        # Generate the first scenario before entering the while loop
        scenario_inspect, scenario_input = self.maritime_analysis.execute(used_mmsi_list)
        num_attempt = 0
        i = 0
        while i < num_data_to_create:
            num_attempt += 1
            if len(new_outputs) == 0:
                example_outputs = random.sample(seed_outputs,5)
            elif len(new_outputs) == 1:
                example_outputs = random.sample(seed_outputs, 4)
                example_outputs = example_outputs + new_outputs
                random.shuffle(example_outputs)
            else:
                example_outputs = random.sample(seed_outputs, 3)
                new_outputs_chosen = random.sample(new_outputs, 2)
                example_outputs = example_outputs + new_outputs_chosen
                random.shuffle(example_outputs)

            prompt = f"""
            {base_prompt}

            Context 1: {example_outputs[0]['input']}
            Radio Chatter 1: {example_outputs[0]['output']}

            Context 2: {example_outputs[1]['input']}
            Radio Chatter 2: {example_outputs[1]['output']}

            Context 3: {example_outputs[2]['input']}
            Radio Chatter 3: {example_outputs[2]['output']}

            

            Context 6: {scenario_input}
            Radio Chatter 6:"""

            result_initial = self.inference(model, tokenizer, prompt, max_new_tokens=self.max_new_tokens)

            # Define the phrase to search for
            phrase = "Radio Chatter 6:"

            # Find the index of the phrase
            start_index = result_initial.find(phrase)

            # Extract the part of the string that comes after the phrase
            if start_index != -1:
                generated_result = result_initial[start_index + len(phrase):].strip()
                if generated_result == "":    
                    print("Blank output.")
                    continue

            generated_result_minus_chatter_2 = generated_result.split("Context 7")[0].strip()

            parts = generated_result_minus_chatter_2.split('Radio Chatter 6:')
            
            if len(parts) > 1:
                first_chatter = parts[1].split('Radio Chatter 6:')[0].strip()
                result = first_chatter
            else:
                result = generated_result_minus_chatter_2    

            if "(" in result:
                print("Paranthesis detected in the output.")
                rejected_results['paranthesis'].append({'input': scenario_input, 'output': result.split('\n')})
                continue

            if result=="":
                print("Blank output after postprocessing.")
                rejected_results['blank'].append({'input': scenario_input, 'output': result.split('\n')})
                continue

            if "[" in result:
                print("Bracket detected in the chatter.")
                rejected_results['brackets'].append({'input': scenario_input, 'output': result.split('\n')})
                continue

            if not result.endswith("."):
                print("Incomplete chatter.")
                rejected_results['incomplete'].append({'input': scenario_input, 'output': result.split('\n')})
                continue
            
            result_cleaned = result.replace(scenario_input["vessel_name"],"")
            result_cleaned = result_cleaned.replace(scenario_input["closest_place_name"], "")
            result_cleaned = result_cleaned.replace(scenario_input["nearest_port"],"") if scenario_input["nearest_port"] else result_cleaned
            result_cleaned = result_cleaned.replace(scenario_input["nearest_harbor"],"") if scenario_input["nearest_harbor"] else result_cleaned
            if self.task_name != "reporting_undesignated_distress":
                if not any(keyword in result_cleaned.lower() for keyword in keywords[task_name]):
                    print("Wrong category.")
                    rejected_results['wrong_category'].append({'input': scenario_input, 'output': result.split('\n')})
                    continue
            else:
                if any(keyword in result_cleaned.lower() for keyword in keywords[task_name]):
                    print(f"Wrong category")
                    rejected_results['wrong_category'].append({'input': scenario_input, 'output': result.split('\n')})
                    continue    

            keys_to_check = ["vessel_name", "vessel_MMSI", "vessel_call_sign", "vessel_type", "vessel_coordinate_dms"]
            
            if self.task_name == "reporting_collision":
                keys_to_check += ["collided_vessel_name", "collided_vessel_type"]

            # Initialize a list to store keys whose values are not in the string
            missing_keys = []
            is_missing_info = False
            for key in keys_to_check:
                # Get the value for the current key
                value = scenario_input.get(key)
                
                # Check if the value is not None and is not in the target string
                if value is not None and str(value).lower() not in result.lower():
                    missing_keys.append(key)
                    print("Missing vessel info in the chatter.")
                    rejected_results[key].append({'input': scenario_input, 'output': result.split('\n')})
                    is_missing_info = True
                    break
            if is_missing_info:
                continue

            if not scenario_input.get('vessel_MMSI'):
                # Get the part of the result before the first newline character
                first_line = result.split('\n', 1)[0]

                # Remove punctuation
                _result = re.sub(r'[^\w\s]', '', first_line)
                # Normalize spaces
                _result = ' '.join(_result.split())

                # Normalize the first line of the result
                normalized_result = _result.lower()
                words = normalized_result.split()

                # Invert the char_to_string dictionary to map words back to characters/digits
                string_to_char = {v.lower(): k for k, v in self.maritime_analysis.char_to_string.items() if k.isdigit()}

                found_sequence = False

                # Iterate through the words list to check for nine consecutive words
                for j in range(len(words) - 8):
                    # Check if the current slice of nine words matches any sequence of dictionary values
                    current_slice = words[j:j+9]
                    if all(word in string_to_char for word in current_slice) or all(word in string_to_char.values() for word in current_slice):
                        found_sequence = True
                        break
                if found_sequence:
                    print("Hallucinated MMSI")
                    rejected_results['hallucinated_mmsi'].append({'input': scenario_input, 'output': result.split('\n')})
                    continue   

            if not scenario_input.get('vessel_call_sign'):
                # Get the part of the result before the first newline character
                first_line = result.split('\n', 1)[0]

                # Remove punctuation
                _result = re.sub(r'[^\w\s]', '', first_line)
                # Normalize spaces
                _result = ' '.join(_result.split())

                # Normalize the first line of the result
                normalized_result = _result.lower()
                words = normalized_result.replace(scenario_input["vessel_name"].lower(),"").split()

                # Invert the char_to_string dictionary to map words back to characters/digits
                string_to_char = {v.lower(): k for k, v in self.maritime_analysis.char_to_string.items() if not k.isdigit()}

                found_sequence = False

                # Iterate through the words list to check for nine consecutive words
                for j in range(len(words) - 1):
                    # Check if the current slice of nine words matches any sequence of dictionary values
                    current_slice = words[j:j+2]
                    if all(word in string_to_char for word in current_slice):
                        found_sequence = True
                        break
                if found_sequence:
                    print("Hallucinated call sign")
                    rejected_results['hallucinated_call_sign'].append({'input': scenario_input, 'output': result.split('\n')})
                    continue

            if (not scenario_input['vessel_MMSI'] and 'mmsi' in result.lower()) or (not scenario_input['vessel_call_sign'] and 'call sign' in result.lower()):
                print("Unknown mmsi or call sign hallucinated.")
                rejected_results['unknown'].append({'input': scenario_input, 'output': result.split('\n')})
                continue

            # Split the result into sentences based on . or ?
            sentences = re.split(r'\. |\? ', result)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if sentences[0].lower().count("mayday") != 3:
                print("Mayday phrase is not used correctly.")
                rejected_results['mayday'].append({'input': scenario_input, 'output': result.split('\n')})
                continue 

            if scenario_input["vessel_name"].lower() not in sentences[1].lower():
                print("Vessel Name is not after Mayday.")
                rejected_results['vessel_name_not_after_mayday'].append({'input': scenario_input, 'output': result.split('\n')})
                continue 

            # Initialize a dictionary to count sentences
            sentence_counts = {}

            # Iterate over each sentence
            for sentence in sentences:
                # Check if the vessel name is in the sentence
                if scenario_input['vessel_name'] in sentence:
                    # Replace occurrences of the vessel name with a placeholder
                    processed_sentence = sentence.replace(scenario_input['vessel_name'].lower(), "SHIP_NAME")
                else:
                    # If the vessel name is not in the sentence, use the sentence as is
                    processed_sentence = sentence
                processed_sentence = processed_sentence.replace("Coast Guard", "Coast_Guard")
                # Split the processed sentence into words
                words = processed_sentence.split()

                # Check if the sentence length is greater than three words (considering vessel name as single word)
                if len(words) > 3:
                    sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1

            # Find non-unique sentences
            non_unique_sentences = [s for s, count in sentence_counts.items() if count > 1]

            if non_unique_sentences:
                print("Non-unique sentences longer than three words found.")
                rejected_results['duplicate_sentences'].append({'input': scenario_input, 'output': result.split('\n')})
                continue    
            
            result_cargo = result.lower()
            if self.task_name == "reporting_collision" and scenario_input["collided_vessel_type"] == "Cargo Vessel":
                    result_cargo = result.lower().replace("cargo vessel","")
            # Filter the chatter which is not logical in terms of cargo
            if not scenario_input['can_have_cargo'] and "cargo" in result_cargo.lower():
                print("Cargo logic is not valid")
                rejected_results['false_cargo_logic'].append({'input': scenario_input, 'output': result.split('\n')})
                continue

            # Filter the chatter with hallucinated vessel type
            vessel_types = list(self.maritime_analysis.vessel_df['Vessel Type'].unique())
            if "Ship".lower() not in scenario_input["vessel_type"].lower():
                vessel_types.extend(['Ship'])
            vessel_types.extend(['Motor Vessel'])
            vessel_types.remove('Other')
            vessel_types.remove(scenario_input['vessel_type'])
            if self.task_name == "reporting_collision" and scenario_input["collided_vessel_type"] and scenario_input['vessel_type'] != scenario_input["collided_vessel_type"]:
                vessel_types.remove(scenario_input["collided_vessel_type"])
            if any(elem in sentences[1].replace(scenario_input['vessel_name'], "").lower() for elem in [vessel_type.lower() for vessel_type in vessel_types]) \
                or any(phrase in result.lower() for phrase in ["we are a ", "i am a "]):
                print("Vessel type hallunicated or improperly stated")
                rejected_results['hallucinated_vessel_type'].append({'input': scenario_input, 'output': result.split('\n')})
                continue

            if scenario_input['nearest_port'] and scenario_input['nearest_harbor']:
                if scenario_input['nearest_port'] != scenario_input['nearest_harbor'] \
                and scenario_input['nearest_port'].lower() in result.lower() \
                and scenario_input['nearest_harbor'].lower() in result.lower():
                    if not (scenario_input['nearest_port'] in scenario_input['nearest_harbor'] or scenario_input['nearest_harbor'] in scenario_input['nearest_port']):
                        print("Both port and harbor are used.")
                        rejected_results['both_port_and_harbor'].append({'input': scenario_input, 'output': result.split('\n')})
                        continue
                    else:
                        if (scenario_input['nearest_port'] in result.replace(scenario_input['nearest_harbor'], "")\
                            and scenario_input['nearest_harbor'] in result.replace(scenario_input['nearest_port'], "")):
                            print("Both port and harbor are used.")
                            rejected_results['both_port_and_harbor'].append({'input': scenario_input, 'output': result.split('\n')})
                            continue
                        
            chatters = result.replace(',',"").lower().split("\n")
            if len(chatters) == 1 or not any(phrase in chatters[1] for phrase in ["this is coast guard", "coast guard here", "coast guard responding", "this is the coast guard"
            f'coast guard {scenario_input["vessel_name"].lower()}', f'coast guard to {scenario_input["vessel_name"].lower()}']):
                print("Coast Guard doesnt answer properly.")
                rejected_results['no_coast_guard'].append({'input': scenario_input, 'output': result.split('\n')})
                continue
            
            if scenario_input["digit_by_digit"] and any(word in result_cleaned.lower() for word in self.non_digits):
                print("Wrong digit logic")
                rejected_results['digit_by_digit'].append({'input': scenario_input, 'output': result.split('\n')})
                continue

            ### OPTIONAL INFO CHECK
            used_optional_info = {
                "compass_direction": False,
                "closest_place_name": False,
                "distance_to_nearest_place": False,
                "closest_place_country": False,
                "distance_to_nearest_port": False,
                "nearest_port": False,
                "distance_to_nearest_harbor": False,
                "nearest_harbor": False,
                "closest_water_body": False,
                "closest_place_country": False,
            }

            rejected_optional_info_found = False

            for key in used_optional_info.keys():
                if not scenario_input[key]:
                    used_optional_info[key] = 'Not in the scenario'
                elif key == "compass_direction":        
                    phrases_with_place_name = [phrase for phrase in re.split(r'[,.]+', result) if scenario_input['closest_place_name'].lower() in phrase.lower()]
                    directions = ['north', 'east', 'west', 'south']
                    if len(phrases_with_place_name) > 0:
                        used_optional_info["closest_place_name"] = True
                        if any(direction in phrases_with_place_name[0].lower() for direction in directions) and (scenario_input['compass_direction'].lower().replace(" ","") not in phrases_with_place_name[0].lower().replace(" ","")):
                            num_compass += 1
                        else:
                            used_optional_info["compass_direction"] = True    

                        words = phrases_with_place_name[0].lower().split()
                        numbers = []
                        temp_number_words = []

                        for word in words:
                            processed_words = word.replace("-"," ").split()
                            if any(processed_word in self.maritime_analysis.single_digits or processed_word in self.maritime_analysis.tens or processed_word in self.maritime_analysis.teens or processed_word in ["hundred", "thousand"] for processed_word in processed_words):
                                temp_number_words.append(word)
                            else:
                                if temp_number_words:
                                    # Convert the joined number words to a number
                                    combined_number = ' '.join(temp_number_words)
                                    try:
                                        number = self.words_to_number(combined_number)
                                        numbers.append(number)
                                    except ValueError:
                                        pass  # Ignore if conversion fails
                                    temp_number_words = []

                        # Check for any remaining number words at the end
                        if temp_number_words:
                            combined_number = ' '.join(temp_number_words)
                            try:
                                number = self.words_to_number(combined_number)
                                numbers.append(number)
                            except ValueError:
                                pass  # Ignore if conversion fails    

                        if 'mile' in phrases_with_place_name[0].lower() and self.words_to_number(scenario_input['distance_to_nearest_place'].lower()) not in numbers:
                            num_distance_to_closest_place += 1
                            print("Wrong distance to nearest place")
                            rejected_results['distance_to_nearest_place'].append({'input': scenario_input, 'output': result.split('\n')})
                            rejected_optional_info_found = True
                            break
                        else:
                            used_optional_info["distance_to_nearest_place"] = True
                    else:
                        num_closest_place += 1               
                
                elif key == "nearest_port":
                    if not scenario_input['nearest_port']:
                        used_optional_info['nearest_port'] = 'Not in the scenario'
                        used_optional_info['distance_to_nearest_port'] = 'Not in the scenario'
                    elif scenario_input['nearest_port'] and scenario_input['closest_place_name'] != scenario_input['nearest_port']:
                        sentences_with_port_name = [sentence for sentence in sentences if scenario_input['nearest_port'].lower() in sentence.lower()]
                        if len(sentences_with_port_name) > 0:
                            used_optional_info['distance_to_nearest_port'] = True
                            words = sentences_with_port_name[0].lower().split()
                            numbers = []
                            temp_number_words = []

                            for word in words:
                                processed_words = word.replace("-"," ").split()
                                if any(processed_word in self.maritime_analysis.single_digits or processed_word in self.maritime_analysis.tens or processed_word in self.maritime_analysis.teens or processed_word in ["hundred", "thousand"] for processed_word in processed_words):
                                    temp_number_words.append(word)
                                else:
                                    if temp_number_words:
                                        # Convert the joined number words to a number
                                        combined_number = ' '.join(temp_number_words)
                                        try:
                                            number = self.words_to_number(combined_number)
                                            numbers.append(number)
                                        except ValueError:
                                            pass  # Ignore if conversion fails
                                        temp_number_words = []

                            # Check for any remaining number words at the end
                            if temp_number_words:
                                combined_number = ' '.join(temp_number_words)
                                try:
                                    number = self.words_to_number(combined_number)
                                    numbers.append(number)
                                except ValueError:
                                    pass  # Ignore if conversion fails   

                            if 'mile' in sentences_with_port_name[0].lower() and self.words_to_number(scenario_input['distance_to_nearest_port'].lower()) not in numbers:
                                num_distance_to_closest_port += 1
                                print("Wrong distance to nearest port")
                                rejected_results['distance_to_nearest_port'].append({'input': scenario_input, 'output': result.split('\n')})
                                rejected_optional_info_found = True
                                break      
                        else:
                            num_closest_port += 1                 

                elif key == "nearest_harbor":
                    if not scenario_input['nearest_harbor']:
                        used_optional_info['nearest_harbor'] = 'Not in the scenario'
                        used_optional_info['distance_to_nearest_harbor'] = 'Not in the scenario'    
                    elif scenario_input['nearest_harbor'] and scenario_input['closest_place_name'] != scenario_input['nearest_harbor']:
                        sentences_with_harbor_name = [phrase for phrase in sentences if scenario_input['nearest_harbor'].lower() in phrase.lower()]
                        if len(sentences_with_harbor_name) > 0:
                            used_optional_info['distance_to_nearest_harbor'] = True
                            words = sentences_with_harbor_name[0].lower().split()
                            numbers = []
                            temp_number_words = []

                            for word in words:
                                processed_words = word.replace("-"," ").split()
                                if any(processed_word in self.maritime_analysis.single_digits or processed_word in self.maritime_analysis.tens or processed_word in self.maritime_analysis.teens or processed_word in ["hundred", "thousand"] for processed_word in processed_words):
                                    temp_number_words.append(word)
                                else:
                                    if temp_number_words:
                                        # Convert the joined number words to a number
                                        combined_number = ' '.join(temp_number_words)
                                        try:
                                            number = self.words_to_number(combined_number)
                                            numbers.append(number)
                                        except ValueError:
                                            pass  # Ignore if conversion fails
                                        temp_number_words = []

                            # Check for any remaining number words at the end
                            if temp_number_words:
                                combined_number = ' '.join(temp_number_words)
                                try:
                                    number = self.words_to_number(combined_number)
                                    numbers.append(number)
                                except ValueError:
                                    pass  # Ignore if conversion fails    

                            if 'mile' in sentences_with_harbor_name[0].lower() and self.words_to_number(scenario_input['distance_to_nearest_harbor'].lower()) not in numbers:
                                num_distance_to_closest_harbor += 1
                                print("Wrong distance to nearest harbor")
                                rejected_results['distance_to_nearest_harbor'].append({'input': scenario_input, 'output': result.split('\n')})
                                rejected_optional_info_found = True
                                break  
                        else:
                            num_closest_harbor += 1       
                
                elif key == "closest_water_body":
                    if not scenario_input[key]:
                        used_optional_info[key] = 'Not in the scenario'
                    elif scenario_input[key].lower() in result.lower():
                        used_optional_info[key] = True
                    elif scenario_input[key].lower() not in result.lower():    
                        num_water_body += 1  

                elif key == "closest_place_name":
                    if not scenario_input[key]:
                        used_optional_info[key] = 'Not in the scenario'

                elif key == "country":
                    if not scenario_input[key]:
                        used_optional_info[key] = 'Not in the scenario'
                    elif scenario_input[key].lower() in result.lower():
                        used_optional_info[key] = True
                    elif scenario_input[key].lower() not in result.lower():    
                        num_country += 1        

            if rejected_optional_info_found:
                continue
            is_unique = True
            best_score_seed = 0
            best_score_synthetic = 0
            for task in self.seed_tasks_data:
                if task == task_name:
                    for instance in self.seed_tasks_data[task]["instances"]:
                        rouge_l_score = self.calculate_rouge_l(instance["output"].split("degree", 1)[1].split(".", 1)[1].strip(),
                                                                result.split("degree", 1)[1].split(".", 1)[1].strip())
                        if rouge_l_score > self.rouge_l_score:
                            print("Output is not unique.")
                            is_unique = False
                            rejected_results['non-unique'].append({'input': scenario_input, 'output': result.split('\n'), 'similar_chatter': instance["output"].split('\n'), 'rouge_l_score': rouge_l_score})
                            break
                    if not is_unique:
                        break    
            if is_unique:        
                for output in new_outputs:
                    rouge_l_score = self.calculate_rouge_l(output["output"].split("degree", 1)[1].split(".", 1)[1].strip(),
                                                                result.split("degree", 1)[1].split(".", 1)[1].strip())
                    if rouge_l_score > self.rouge_l_score:
                        print("Output is not unique.")
                        is_unique = False
                        rejected_results['non-unique'].append({'input': scenario_input, 'output': result.split('\n'), 'similar_chatter': output["output"].split('\n'), 'rouge_l_score': rouge_l_score})
                        break        
            if is_unique:
                for instance in self.seed_tasks_data[task_name]["instances"]:
                    rouge_l_score = self.calculate_rouge_l(instance["output"].split("degree", 1)[1].split(".", 1)[1].strip(),
                                                            result.split("degree", 1)[1].split(".", 1)[1].strip())
                    if rouge_l_score > best_score_seed:
                        best_score_seed = rouge_l_score
                        best_matching_seed_chatter_id = instance["id"]

                result = {
                    'id': i + 1,
                    'input': scenario_input,
                    'output': result,
                    "best_score_seed": best_score_seed,
                    "best_matching_seed_chatter_id": best_matching_seed_chatter_id,
                    "used_optional_info": used_optional_info
                }              
                new_outputs.append(result)
                scenario_inspect_list.append({"scenario_inspect": scenario_inspect, "scenario_input": scenario_input})
                used_mmsi_list.append(scenario_input["vessel_MMSI"])
                scenario_inspect, scenario_input = self.maritime_analysis.execute(used_mmsi_list)         
                print(f"{len(new_outputs)} ----- {round(100*len(new_outputs)/num_attempt, 2)}%")
                i += 1
        similarities = []        
        for i, instance in enumerate(new_outputs):
            best_score_synthetic = 0
            current_output = instance["output"]

            for j, other_instance in enumerate(new_outputs):
                if i == j:
                    continue
                
                other_output = other_instance["output"]
                # Extract and split as per your logic
                current_segment = current_output.split("degree", 1)[1].split(".", 1)[1].strip()
                other_segment = other_output.split("degree", 1)[1].split(".", 1)[1].strip()
                
                # Calculate Rouge-L score
                rouge_l_score = self.calculate_rouge_l(current_segment, other_segment)

                if rouge_l_score > best_score_synthetic:
                    best_score_synthetic = rouge_l_score
                    best_matching_synthetic_chatter_id = other_instance["id"]

            new_outputs[i]["best_score_synthetic"] = best_score_synthetic
            new_outputs[i]["best_matching_synthetic_chatter_id"] = best_matching_synthetic_chatter_id
            if best_score_synthetic not in similarities:
                similarities.append(best_score_synthetic)                 

        new_tasks_data[task_name] = {}
        new_tasks_data[task_name]["instruction"] = self.seed_tasks_data[task_name]["instruction"]
        new_tasks_data[task_name]["instances"] = []

        for output in new_outputs:          
            new_tasks_data[task_name]["instances"].append(output)
        for instance in new_tasks_data[task_name]['instances']:
            instance['output'] = instance['output'].split('\n') 

        total_rejected_outputs = 0
        new_tasks_data['rejected_results'] = {key: 0 for key in rejected_results.keys()}
        
        for key in rejected_results:
            new_tasks_data['rejected_results'][key] = len(rejected_results[key])
            total_rejected_outputs += new_tasks_data['rejected_results'][key]      

        new_tasks_data['rejected_results']["total_rejected_outputs"] = total_rejected_outputs
        new_tasks_data["optional_info"] = {
            "compass_direction": num_compass, 
            "nearest_port": num_closest_port, 
            "nearest_harbor": num_closest_harbor, 
            "closest_place_country": num_country, 
            "distance_to_nearest_place": num_distance_to_closest_place, 
            "distance_to_nearest_port": num_distance_to_closest_port, 
            "distance_to_nearest_harbor": num_distance_to_closest_harbor,
            "closest_water_body": num_water_body, 
            "closest_place_name": num_closest_place,
            }
        new_tasks_data["used_mmsi_list"] = used_mmsi_list 
        print(f"Number of rejected outputs: {sum(len(rejected_results[key]) for key in rejected_results.keys())}")
        seed_outputs = []

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        if not os.path.exists(f'{self.save_dir}{self.task_name}/{current_time}'):
            os.makedirs(f'{self.save_dir}{self.task_name}/{current_time}', exist_ok=True)

        scores = [result["best_score_seed"] for result in new_tasks_data[self.task_name]["instances"]]
        plt.hist(scores, bins=20, alpha=0.75, color='blue')
        plt.title("ROUGE-L Similarity of Seed Chatters")
        plt.xlabel("ROUGE-L Score")
        plt.ylabel("Number of Chatters")
        plt.savefig(f"{self.save_dir}{self.task_name}/{current_time}/rouge_l_histogram_seed_{current_time}.png")
        plt.show()

        plt.clf()

        scores = [similarity for similarity in similarities]
        plt.hist(scores, bins=20, alpha=0.75, color='blue')
        plt.title("ROUGE-L Similarity of Synthetic Chatters")
        plt.xlabel("ROUGE-L Score")
        plt.ylabel("Number of Chatters")
        plt.savefig(f"{self.save_dir}{self.task_name}/{current_time}/rouge_l_histogram_synthetic_{current_time}.png")
        plt.show()     

        new_tasks_path = f"{self.save_dir}{self.task_name}/{current_time}/{current_time}_new_outputs.json"
        rejected_outputs_paths = f"{self.save_dir}{self.task_name}/{current_time}/{current_time}_rejected_outputs.json"
        scenario_inspection_path = f"{self.save_dir}{self.task_name}/{current_time}/{current_time}_inspection.json"

        with open(new_tasks_path, 'w') as json_file:
            json.dump(new_tasks_data, json_file, indent=4, ensure_ascii=False, sort_keys=False)

        with open(rejected_outputs_paths, 'w') as json_file:
            json.dump(rejected_results, json_file, indent=4, ensure_ascii=False, sort_keys=False)

        with open(scenario_inspection_path, 'w') as json_file:
            json.dump(scenario_inspect_list, json_file, indent=4, ensure_ascii=False, sort_keys=False)
            
if __name__ == "__main__":
    land_shapefile = "./GSHHS_dataset/GSHHS_shp/f/GSHHS_f_L1.shp" # path of the GSHHS data
    geonames_data_path = './all_countries/allCountries.txt' # path of the geonames data
    task_name = "task_name" # Choose from ["reporting_fire", "reporting_flooding", "reporting_collision", "reporting_grounding":
                            #              "reporting_list-danger_of_capsizing", "reporting_sinking", "reporting_attack",
                            #              "reporting_person_overboard", "reporting_drift", "reporting_undesignated_distress"]
    prompt_file = './prompts/prompt.txt' # Choose the prompt file of the task category you chose for task_name variable.

    ### Hyperparamaters for LLM generation
    temperature = 0.9
    top_k = 400
    max_new_tokens = 400
    do_sample = True

    model_name = '.path_to_base_LLM' # path of the LLM you are using
    
    rouge_l_score = 0.7 # Rouge L score threshold of the uniquness filter. 0 means completely unique output, 1 means completely same output.
    num_data_to_create = 500 # number of instances to generate. Choose any number.

    seed_outputs_path = "./seed_outputs.json"
    keywords_path = './data/keywords.json'
    vessel_data_path = '.data/ship_data_dk_us.pkl'
    save_dir = "./experiments/"

    
    print(task_name)

    start_time = time.time()

    dataset_creator = DatasetCreator(land_shapefile, geonames_data_path, seed_outputs_path, keywords_path, task_name, prompt_file, temperature, top_k, max_new_tokens, model_name, do_sample,
                    rouge_l_score, num_data_to_create, vessel_data_path, save_dir)
    
    dataset_creator.create_dataset()

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Convert to hours, minutes, and seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Format as hhmmss
    print(f'Execution took {hours} hours {minutes} minutes {seconds}')