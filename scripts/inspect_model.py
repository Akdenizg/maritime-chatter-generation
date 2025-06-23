import json
import geo_test
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
import re
import pandas as pd
import datetime
import os

class ModelInspection:
    def __init__(self, keywords_path, ship_data_path, land_shapefile, geonames_data_path, seed_path, synthetic_chatter_path, save_dir):
        self.load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
        self.keywords_path = keywords_path
        self.keywords = self.read_json(self.keywords_path)
        self.ship_data_path = ship_data_path
        self.land_shapefile = land_shapefile
        self.geonames_data_path = geonames_data_path
        self.seed_path = seed_path
        self.synthetic_chatter_path = synthetic_chatter_path
        self.synthetic_chatters = self.read_json(self.synthetic_chatter_path)
        self.training_chatter_path = self.synthetic_chatters["hyperparameters"]["chatter_path"] 
        self.num_epoch = self.synthetic_chatters["hyperparameters"]["num_epoch"]
        self.ship_df = pd.read_pickle(self.ship_data_path)
        self.filter_list = [
            'paranthesis',
            'brackets',
            'mayday',
            'incomplete',
            'wrong_category',
            'unknown',
            'vessel_name_not_after_mayday',
            'no_coast_guard',
            'duplicate_sentences',
            'vessel_name',
            'vessel_MMSI',
            'vessel_call_sign',
            'vessel_type',
            'vessel_coordinate_dms',
            'hallucinated_mmsi',
            'hallucinated_call_sign',
            'digit_by_digit',
            'false_cargo_logic',
            'hallucinated_vessel_type',
            'non-unique',
            'both_port_and_harbor',
            'closest_place_name',
            'closest_water_body',
            'closest_place_country',
            "distance_to_nearest_port",
            "distance_to_nearest_harbor",
            "distance_to_nearest_place",
            "compass",
        ]
        self.evaluation_coefficients = {
            "Compliance with SMCP": {
                "paranthesis": 1,
                "brackets": 1,
                "mayday": 1,
                "incomplete": 2,
                "vessel_name_not_after_mayday": 1,
                "duplicate_sentences": 2,
                "no_coast_guard": 1,
                "digit_by_digit": 1,
            },
            "Information Accuracy": {
                "wrong_category": 2,
                "unknown": 1,
                "vessel_name": 2,
                "vessel_MMSI": 2,
                "vessel_call_sign": 2,
                "vessel_type": 2,
                "vessel_coordinate_dms": 2,
                "hallucinated_mmsi": 1,
                "hallucinated_call_sign": 1,
                "false_cargo_logic": 1,
                "hallucinated_vessel_type": 1,
                "compass": 2,
                "both_port_and_harbor": 1,
                "distance_to_nearest_port": 1,
                "distance_to_nearest_harbor": 1,
                "distance_to_nearest_place": 1,
            },
            "Uniqueness": {
                "non-unique": 1
            },
            "Usage of Optional Info": {
                "compass": 1,
                "closest_place_name": 1,
                "distance_to_nearest_place": 1,
                "nearest_port": 1,
                "distance_to_nearest_port": 1,
                "nearest_harbor": 1,
                "distance_to_nearest_harbor": 1,
                "closest_water_body": 1,
                "closest_place_country": 1,
            }
        }
        self.char_to_string = geo_test.MaritimeAnalysis(self.land_shapefile, self.geonames_data_path, self.ship_data_path, task_name=None).char_to_string
        self.save_dir = os.path.join(save_dir, self.synthetic_chatters["task_name"])
        self.single_digits = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight','nine']
        self.tens = ['ten', 'twenty', 'thirty', 'forty','fifty', 'sixty', 'seventy', 'eighty', 'ninety']
        self.teens = ['eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
        self.non_digits = ['twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety','hundred','thousand','million'] 

    def read_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def calculate_rouge_l(self, reference_text, generated_text):
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = scorer.score(reference_text, generated_text)
        return scores['rougeL'].fmeasure

    def words_to_number(self, words):
        words = words.replace("-", " ").split()
        total = 0
        current = 0
        previous = None  # To track the type of the previous word

        for word in words:
            if word in self.single_digits:
                digit = self.single_digits.index(word)
                if previous == 'tens':
                    current += digit
                else:
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

    def assess_context_accuracy(self, filter_list, output, scenario_input, rouge_l_score, task_name):
        passed_filters = {item: True for item in filter_list}
        passed_filters['used_optional_info'] = {
            "compass": True,
            "closest_place_name": True,
            "distance_to_nearest_place": True,
            "nearest_port": True,
            "distance_to_nearest_port": True,
            "nearest_harbor": True,
            "distance_to_nearest_harbor": True,
            "closest_water_body": True,
            "closest_place_country": True,
        }
        issue_counts = {filter_name: 0 for filter_name in passed_filters if filter_name != 'used_optional_info'}
        issue_counts.update({sub_filter_name: 0 for sub_filter_name in passed_filters['used_optional_info']})

        # Paranthesis Filter
        if "(" in output:
            passed_filters['paranthesis'] = False

        # Brackets Filter
        if "[" in output:
            passed_filters['brackets'] = False

        # Mayday Filter
        sentences = re.split(r'\. |\? ', output)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences[0].lower().count("mayday") != 3:
            passed_filters['mayday'] = False

        # Incomplete Filter
        if not output.endswith("."):
            passed_filters['incomplete'] = False

        # Blank Filter
        if output.strip() == "":
            passed_filters['blank'] = False

        # Wrong Category Filter
        result_cleaned = output.replace(scenario_input["vessel_name"], "")
        result_cleaned = result_cleaned.replace(scenario_input["closest_place_name"], "")
        result_cleaned = result_cleaned.replace(scenario_input["nearest_port"], "") if scenario_input["nearest_port"] else result_cleaned
        result_cleaned = result_cleaned.replace(scenario_input["nearest_harbor"], "") if scenario_input["nearest_harbor"] else result_cleaned
        if task_name != "reporting_undesignated_distress":
            if not any(keyword in result_cleaned.lower() for keyword in self.keywords[task_name]):
                passed_filters['wrong_category'] = False
        else:
            if any(keyword in result_cleaned.lower() for keyword in self.keywords[task_name]):
                passed_filters['wrong_category'] = False

        # Unknown Information Filter
        if (not scenario_input['vessel_MMSI'] and 'mmsi' in output.lower()) or (not scenario_input['vessel_call_sign'] and 'call sign' in output.lower()):
            passed_filters['unknown'] = False

        # Vessel Name Not After Mayday Filter
        if len(sentences) > 1 and scenario_input["vessel_name"].lower() not in sentences[1].lower():
            passed_filters['vessel_name_not_after_mayday'] = False

        # Duplicate Sentences Filter
        sentence_counts = {}
        for sentence in sentences:
            if scenario_input['vessel_name'] in sentence:
                processed_sentence = sentence.replace(scenario_input['vessel_name'], "VESSEL_NAME")
            else:
                processed_sentence = sentence
            processed_sentence = processed_sentence.replace("Coast Guard", "Coast_Guard")
            words = processed_sentence.split()
            if len(words) > 3:
                sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1

        non_unique_sentences = [s for s, count in sentence_counts.items() if count > 1]
        if non_unique_sentences:
            passed_filters['duplicate_sentences'] = False

        # Vessel Information Filters
        keys_to_check = ["vessel_name", "vessel_MMSI", "vessel_call_sign", "vessel_type", "vessel_coordinate_dms"]
        for key in keys_to_check:
            value = scenario_input[key]
            if not value:
                passed_filters[key] = 'Not in scenario'
            elif str(value).lower() not in output.lower():
                passed_filters[key] = False

        # Hallucinated MMSI Filter
        if not scenario_input['vessel_MMSI']:
            first_line = output.split('\n', 1)[0]
            _result = re.sub(r'[^\w\s]', '', first_line)
            _result = ' '.join(_result.split())
            normalized_result = _result.lower()
            words = normalized_result.split()
            string_to_char = {v.lower(): k for k, v in self.char_to_string.items() if k.isdigit()}
            found_sequence = False
            for j in range(len(words) - 8):
                current_slice = words[j:j+9]
                if all(word in string_to_char for word in current_slice) or all(word in string_to_char.values() for word in current_slice):
                    found_sequence = True
                    break
            if found_sequence:
                passed_filters['hallucinated_mmsi'] = False

        # Hallucinated Call Sign Filter
        if not scenario_input['vessel_call_sign']:
            first_line = output.split('\n', 1)[0]
            _result = re.sub(r'[^\w\s]', '', first_line)
            _result = ' '.join(_result.split())
            normalized_result = _result.lower()
            words = normalized_result.replace(scenario_input["vessel_name"].lower(),"").split()
            string_to_char = {v.lower(): k for k, v in self.char_to_string.items() if not k.isdigit()}
            found_sequence = False
            for j in range(len(words) - 1):
                current_slice = words[j:j+2]
                if all(word in string_to_char for word in current_slice):
                    found_sequence = True
                    break
            if found_sequence:
                passed_filters['hallucinated_call_sign'] = False

        # False Cargo Logic Filter
        result_cargo = output.lower()
        if task_name == "reporting_collision" and scenario_input["collided_vessel_type"] == "Cargo Vessel":
            result_cargo = output.lower().replace("cargo vessel","")
        if not scenario_input['can_have_cargo'] and "cargo" in result_cargo.lower():
            passed_filters['false_cargo_logic'] = False

        # Hallucinated Vessel Type Filter
        vessel_types = list(self.ship_df['Vessel Type'].unique())
        if "Ship".lower() not in scenario_input["vessel_type"].lower():
            vessel_types.extend(['Ship'])
        vessel_types.extend(['Motor Vessel'])
        vessel_types = [vt for vt in vessel_types if vt != 'Other']
        vessel_types = [vt for vt in vessel_types if vt.lower() != scenario_input['vessel_type'].lower()]
        if any(elem in sentences[1].replace(scenario_input['vessel_name'], "").lower() for elem in [vessel_type.lower() for vessel_type in vessel_types]) \
            or any(phrase in output.lower() for phrase in ["we are a ", "i am a "]):
            passed_filters['hallucinated_vessel_type'] = False

        # Both Port and Harbor Filter
        if scenario_input['nearest_port'] and scenario_input['nearest_harbor']:
            if scenario_input['nearest_port'].lower() in output.lower() and scenario_input['nearest_harbor'].lower() in output.lower():
                if not (scenario_input['nearest_port'] in scenario_input['nearest_harbor'] or scenario_input['nearest_harbor'] in scenario_input['nearest_port']):
                    passed_filters['both_port_and_harbor'] = False
                else:
                    if (scenario_input['nearest_port'] in output.replace(scenario_input['nearest_harbor'], "") \
                        and scenario_input['nearest_harbor'] in output.replace(scenario_input['nearest_port'], "")):
                        passed_filters['both_port_and_harbor'] = False

        # Non-Unique Filter
        if rouge_l_score > 0.7:
            passed_filters['non-unique'] = False
        else:
            passed_filters['non-unique'] = 1 - rouge_l_score

        # No Coast Guard Filter
        chatters = output.lower().replace(',',"").split("\n")
        if len(chatters) == 1 or not any(phrase in chatters[1] for phrase in ["this is coast guard", "coast guard here", "coast guard responding", "this is the coast guard",
                                                                            f"coast guard {scenario_input['vessel_name'].lower()}", f"coast guard to {scenario_input['vessel_name'].lower()}"]):
            passed_filters['no_coast_guard'] = False

        # Digit by Digit Filter
        if scenario_input["digit_by_digit"] and any(word in output.lower() for word in self.non_digits):
            passed_filters['digit_by_digit'] = False

        # Used Optional Info Filters
        # Initialize optional info filters
        used_optional_info = passed_filters['used_optional_info']
        for key in used_optional_info.keys():
            if key == "compass":
                if not scenario_input['compass_direction']:
                    used_optional_info["compass"] = "Not in scenario"
                else:
                    phrases_with_place_name = [phrase for phrase in re.split(r'[,.]+', output) if scenario_input['closest_place_name'].lower() in phrase.lower()]
                    directions = ['north', 'east', 'west', 'south']
                    if len(phrases_with_place_name) > 0:
                        if any(direction in phrases_with_place_name[0].lower() for direction in directions) and (scenario_input['compass_direction'].lower().replace(" ","") not in phrases_with_place_name[0].lower().replace(" ","")):
                            used_optional_info["compass"] = False
                        words = phrases_with_place_name[0].lower().split()
                        numbers = []
                        temp_number_words = []

                        for word in words:
                            processed_words = word.replace("-"," ").split()
                            if any(processed_word in self.single_digits or processed_word in self.tens or processed_word in self.teens or processed_word in ["hundred", "thousand"] for processed_word in processed_words):
                                temp_number_words.append(word)
                            else:
                                if temp_number_words:
                                    combined_number = ' '.join(temp_number_words)
                                    try:
                                        number = self.words_to_number(combined_number)
                                        numbers.append(number)
                                    except ValueError:
                                        pass
                                    temp_number_words = []

                        if temp_number_words:
                            combined_number = ' '.join(temp_number_words)
                            try:
                                number = self.words_to_number(combined_number)
                                numbers.append(number)
                            except ValueError:
                                pass

                        if 'mile' in phrases_with_place_name[0].lower() and self.words_to_number(scenario_input['distance_to_nearest_place'].lower()) not in numbers:
                            used_optional_info['distance_to_nearest_place'] = False
                    else:
                        used_optional_info["compass"] = False

            elif key == "distance_to_nearest_port":
                if not scenario_input["nearest_port"]:
                    used_optional_info["nearest_port"] = "Not in scenario"
                else:
                    used_optional_info["nearest_port"] = scenario_input["nearest_port"].lower() in output.lower()
                if scenario_input['nearest_port'] and scenario_input['closest_place_name'] != scenario_input['nearest_port']:
                    sentences_with_port_name = [sentence for sentence in sentences if scenario_input['nearest_port'].lower() in sentence.lower()]
                    if len(sentences_with_port_name) > 0:
                        words = sentences_with_port_name[0].lower().split()
                        numbers = []
                        temp_number_words = []

                        for word in words:
                            processed_words = word.replace("-"," ").split()
                            if any(processed_word in self.single_digits or processed_word in self.tens or processed_word in self.teens or processed_word in ["hundred", "thousand"] for processed_word in processed_words):
                                temp_number_words.append(word)
                            else:
                                if temp_number_words:
                                    combined_number = ' '.join(temp_number_words)
                                    try:
                                        number = self.words_to_number(combined_number)
                                        numbers.append(number)
                                    except ValueError:
                                        pass
                                    temp_number_words = []

                        if temp_number_words:
                            combined_number = ' '.join(temp_number_words)
                            try:
                                number = self.words_to_number(combined_number)
                                numbers.append(number)
                            except ValueError:
                                pass

                        if 'mile' in sentences_with_port_name[0].lower() and self.words_to_number(scenario_input['distance_to_nearest_port'].lower()) not in numbers:
                            used_optional_info['distance_to_nearest_port'] = False
                    else:
                        used_optional_info['distance_to_nearest_port'] = False
                else:
                    used_optional_info['distance_to_nearest_port'] = 'Not in scenario'
            elif key == "distance_to_nearest_harbor":
                if not scenario_input["nearest_harbor"]:
                    used_optional_info["nearest_harbor"] = "Not in scenario"
                else:
                    used_optional_info["nearest_harbor"] = scenario_input["nearest_harbor"].lower() in output.lower()
                if scenario_input['nearest_harbor'] and scenario_input['closest_place_name'] != scenario_input['nearest_harbor']:
                    sentences_with_harbor_name = [sentence for sentence in sentences if scenario_input['nearest_harbor'].lower() in sentence.lower()]
                    if len(sentences_with_harbor_name) > 0:
                        words = sentences_with_harbor_name[0].lower().split()
                        numbers = []
                        temp_number_words = []

                        for word in words:
                            processed_words = word.replace("-"," ").split()
                            if any(processed_word in self.single_digits or processed_word in self.tens or processed_word in self.teens or processed_word in ["hundred", "thousand"] for processed_word in processed_words):
                                temp_number_words.append(word)
                            else:
                                if temp_number_words:
                                    combined_number = ' '.join(temp_number_words)
                                    try:
                                        number = self.words_to_number(combined_number)
                                        numbers.append(number)
                                    except ValueError:
                                        pass
                                    temp_number_words = []

                        if temp_number_words:
                            combined_number = ' '.join(temp_number_words)
                            try:
                                number = self.words_to_number(combined_number)
                                numbers.append(number)
                            except ValueError:
                                pass

                        if 'mile' in sentences_with_harbor_name[0].lower() and self.words_to_number(scenario_input['distance_to_nearest_harbor'].lower()) not in numbers:
                            used_optional_info['distance_to_nearest_harbor'] = False
                    else:
                        used_optional_info['nearest_harbor'] = False
                else:
                    used_optional_info['distance_to_nearest_harbor'] = 'Not in scenario'
            elif key == "closest_place_name":
                if not scenario_input["closest_place_name"]:
                    used_optional_info["closest_place_name"] = "Not in scenario"
                else:
                    used_optional_info["closest_place_name"] = scenario_input["closest_place_name"].lower() in output.lower()
                    if not used_optional_info["closest_place_name"]:
                        passed_filters['closest_place_name'] = False

            elif key == "closest_water_body":
                if not scenario_input["closest_water_body"]:
                    used_optional_info["closest_water_body"] = "Not in scenario"
                else:
                    used_optional_info["closest_water_body"] = scenario_input["closest_water_body"].lower() in output.lower()
                    if not used_optional_info["closest_water_body"]:
                        passed_filters['closest_water_body'] = False

            elif key == "closest_place_country":
                if not scenario_input["closest_place_country"]:
                    used_optional_info["closest_place_country"] = "Not in scenario"
                else:
                    used_optional_info["closest_place_country"] = scenario_input["closest_place_country"].lower() in output.lower()
                    if not used_optional_info["closest_place_country"]:
                        passed_filters['closest_place_country'] = False

        # Update issue_counts based on passed_filters
        for filter_name, passed in passed_filters.items():
            if isinstance(passed, dict):
                for sub_filter, sub_passed in passed.items():
                    if sub_passed is False:
                        issue_counts[sub_filter] += 1
                    elif sub_passed == 'Not in scenario':
                        issue_counts[sub_filter] = 'Not in scenario'
            elif passed is False:
                issue_counts[filter_name] += 1
            elif passed == 'Not in scenario':
                issue_counts[filter_name] = 'Not in scenario'

        # Calculate context accuracy scores
        scores = {
            category: (
                sum(
                    value for key, value in coefficients.items()
                    if issue_counts[key] == 0
                ) / sum(
                    value for key, value in coefficients.items()
                    if issue_counts[key] in [0, 1]
                )
            ) if sum(
                value for key, value in coefficients.items()
                if issue_counts[key] in [0, 1]
            ) > 0 else None  # Avoid division by zero
            for category, coefficients in self.evaluation_coefficients.items()
        }

        return passed_filters, scores

    def inspect(self):
        data_seed = self.read_json(self.seed_path)
        data_training_chatter =  None if self.training_chatter_path == None else self.read_json(self.training_chatter_path)
        data_synthetic_chatter = self.read_json(self.synthetic_chatter_path)
        task_name = data_synthetic_chatter["task_name"]

        results = {'valid_outputs': [], 'invalid_outputs': []}

        # Collect synthetic outputs for similarity calculations
        synthetic_outputs = []

        results_to_check = data_synthetic_chatter["results"]

        i = 0
        for generated_output in results_to_check:
            output_text = "\n".join(generated_output["output"])
            scenario_input = generated_output["input"]
            scenario_inspection = generated_output["inspection"] 

            # Collect the output_text for later similarity calculation among synthetic chatters
            synthetic_outputs.append(output_text)

            # Calculate best score with seed chatters
            best_score_seed = 0
            best_matching_output_seed = ""

            for instance in data_seed[task_name]["instances"]:
                reference_text = "".join(instance["output"])
                reference_text_compared = "\n".join(instance["output"]).split("degree", 1)[1].split(".", 1)[1].strip() if "degree" in "\n".join(instance["output"]) else "\n".join(instance["output"])
                rouge_l = self.calculate_rouge_l(reference_text_compared, output_text.split("degree", 1)[1].split(".", 1)[1].strip()) if "degree" in output_text else self.calculate_rouge_l(reference_text, output_text)
                if rouge_l > best_score_seed:
                    best_score_seed = rouge_l
                    best_matching_output_seed = reference_text
                    a = 1

            # Calculate best score with training chatters
            best_score_training = 0
            best_matching_output_training = ""

            for instance in data_training_chatter[task_name]["instances"]:
                reference_text = "\n".join(instance["output"])
                reference_text_compared = "\n".join(instance["output"]).split("degree", 1)[1].split(".", 1)[1].strip() if "degree" in "\n".join(instance["output"]) else "\n".join(instance["output"])
                rouge_l = self.calculate_rouge_l(reference_text_compared, output_text.split("degree", 1)[1].split(".", 1)[1].strip()) if "degree" in output_text else self.calculate_rouge_l(reference_text, output_text)
                if rouge_l > best_score_training:
                    best_score_training = rouge_l
                    best_matching_output_training = reference_text

            # Calculate best score with synthetic chatters
            best_score_synthetic = 0
            best_matching_output_synthetic = ""        

            for j, instance in enumerate(results_to_check):
                if i == j:
                    continue
                reference_text = "\n".join(instance["output"])
                reference_text_compared = "\n".join(instance["output"]).split("degree", 1)[1].split(".", 1)[1].strip() if "degree" in "\n".join(instance["output"]) else "\n".join(instance["output"])
                rouge_l = self.calculate_rouge_l(reference_text_compared, output_text.split("degree", 1)[1].split(".", 1)[1].strip()) if "degree" in output_text else self.calculate_rouge_l(reference_text, output_text)
                if rouge_l > best_score_synthetic:
                    best_score_synthetic = rouge_l
                    best_matching_output_synthetic= reference_text        

            # Store scores
            generated_output["rouge_l_score_seed"] = best_score_seed
            generated_output["best_matching_output_seed"] = best_matching_output_seed
            generated_output["rouge_l_score_synthetic"] = best_score_synthetic
            generated_output["best_matching_output_synthetic"] = best_matching_output_synthetic
            generated_output["rouge_l_score_training"] = best_score_training
            generated_output["best_matching_output_training"] = best_matching_output_training

            rouge_l_score = best_score_training

            # Assess context accuracy
            passed_filters, scores = self.assess_context_accuracy(
                self.filter_list,
                output_text,
                scenario_input,
                rouge_l_score,
                task_name
            )

            filters_to_exclude_from_check = ['closest_place_name', 'closest_water_body', 'closest_place_country']
            if all(passed_filters[filter_name] != False for filter_name in self.filter_list if filter_name not in filters_to_exclude_from_check):
                results['valid_outputs'].append({
                    "scenario_inspection": scenario_inspection,
                    "scenario_input": scenario_input,
                    "generated_output": generated_output["output"],
                    "rouge_l_score_seed": best_score_seed,
                    "best_matching_output_seed": best_matching_output_seed.split("\n"),
                    "rouge_l_score_training": best_score_training,
                    "best_matching_output_training": best_matching_output_training.split("\n"),
                    "passed_filters": passed_filters,
                    "context_accuracy": scores,
                })
            else:
                results['invalid_outputs'].append({
                    "scenario_inspection": scenario_inspection,
                    "scenario_input": scenario_input,
                    "generated_output": generated_output["output"],
                    "rouge_l_score_seed": best_score_seed,
                    "best_matching_output_seed": best_matching_output_seed.split("\n"),
                    "rouge_l_score_training": best_score_training,
                    "best_matching_output_training": best_matching_output_training.split("\n"),
                    "passed_filters": passed_filters,
                    "context_accuracy": scores
                })

            print(f'Processed instance {i+1}/{len(results_to_check)}')
            i += 1

        # Now compute similarities among synthetic chatters
        # For each synthetic chatter, compute the highest ROUGE-L score with other synthetic chatters
        highest_scores_synthetic = []

        num_synthetic = len(synthetic_outputs)

        print("Computing similarities among synthetic chatters...")

        for idx in range(num_synthetic):
            output_i = synthetic_outputs[idx]
            max_rouge_l = 0
            for jdx in range(num_synthetic):
                if idx == jdx:
                    continue
                output_j = synthetic_outputs[jdx]
                rouge_l = self.calculate_rouge_l(output_i, output_j)
                if rouge_l > max_rouge_l:
                    max_rouge_l = rouge_l
            highest_scores_synthetic.append(max_rouge_l)

        # Save results to JSON
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")

        rouge_l_score_to_consider = "rouge_l_score_training"

        data_to_save = {
            'hyperparameters': self.synthetic_chatters["hyperparameters"],
            "num_valid_chatter": len(results['valid_outputs']),
            "num_invalid_chatter": len(results['invalid_outputs']),
            "average_compliance": sum(result["context_accuracy"]["Compliance with SMCP"] for result in results['valid_outputs'] + results['invalid_outputs'])/len(results['valid_outputs'] + results['invalid_outputs']),
            "average_information_accuracy": sum(result["context_accuracy"]["Information Accuracy"] for result in results['valid_outputs'] + results['invalid_outputs'])/len(results['valid_outputs'] + results['invalid_outputs']),
            "average_uniqueness": sum(result[rouge_l_score_to_consider] for result in results['valid_outputs'] + results['invalid_outputs'])/len(results['valid_outputs'] + results['invalid_outputs']),
            "average_is_unique": sum(result["context_accuracy"]["Uniqueness"] for result in results['valid_outputs'] + results['invalid_outputs'])/len(results['valid_outputs'] + results['invalid_outputs']),
            'results_valid': results['valid_outputs'],
            'results_invalid': results['invalid_outputs'],
        }

        dir_path = f'{self.save_dir}/{task_name}/{current_time}'
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path,f"{current_time}.json"), 'w') as outfile:
            json.dump(data_to_save, outfile, ensure_ascii=False, indent=4)

        # Create histograms of ROUGE-L scores

        # Histogram of highest matching ROUGE-L scores with seed chatters
        scores_seed = [result["rouge_l_score_seed"] for result in results['valid_outputs'] + results['invalid_outputs']]
        plt.hist(scores_seed, bins=20, alpha=0.75, color='blue', edgecolor='black', label='Histogram')
        plt.axvline(x=0.7, color='red', linestyle='--', linewidth=1.5, label='Threshold (0.7)')
        plt.title("ROUGE-L Similarity of Synthetic Chatters with Seed Chatters")
        plt.xlabel("ROUGE-L Score")
        plt.ylabel("Number of Chatters")
        plt.legend(loc='upper right')  # Add legend
        plt.savefig(os.path.join(dir_path, f"rouge_l_histogram_seed_{current_time}.png"))
        plt.savefig(os.path.join(dir_path, f"rouge_l_histogram_seed_{current_time}.pdf"))
        plt.show()

        plt.clf()

        # Histogram of highest matching ROUGE-L scores with training chatters
        scores_training = [result["rouge_l_score_training"] for result in results['valid_outputs'] + results['invalid_outputs']]
        plt.hist(scores_training, bins=20, alpha=0.75, color='blue', edgecolor='black', label='Histogram')
        plt.axvline(x=0.7, color='red', linestyle='--', linewidth=1.5, label='Threshold (0.7)')
        plt.title("ROUGE-L Similarity of Synthetic Chatters with Training Chatters")
        plt.xlabel("ROUGE-L Score")
        plt.ylabel("Number of Chatters")
        plt.legend(loc='upper right')  # Add legend
        plt.savefig(os.path.join(dir_path, f"rouge_l_histogram_training_{current_time}.png"))
        plt.savefig(os.path.join(dir_path, f"rouge_l_histogram_training_{current_time}.pdf"))
        plt.show()

        plt.clf()

        # Histogram of highest matching ROUGE-L scores among synthetic chatters
        plt.hist(highest_scores_synthetic, bins=20, alpha=0.75, color='blue', edgecolor='black', label='Histogram')
        plt.axvline(x=0.7, color='red', linestyle='--', linewidth=1.5, label='Threshold (0.7)')
        plt.title("ROUGE-L Similarity Among Synthetic Chatters")
        plt.xlabel("ROUGE-L Score")
        plt.ylabel("Number of Chatters")
        plt.legend(loc='upper right')  # Add legend
        plt.savefig(os.path.join(dir_path, f"rouge_l_histogram_synthetic_{current_time}.png"))
        plt.savefig(os.path.join(dir_path, f"rouge_l_histogram_synthetic_{current_time}.pdf"))
        plt.show()

        plt.clf()

if __name__ == "__main__":
    keywords_path = './data/keywords.json' # path to keywords.json
    ship_data_path = './data/ship_data_dk_us.pkl' # path to ship data

    land_shapefile = "./GSHHS_dataset/GSHHS_shp/f/GSHHS_f_L1.shp" # path of the GSHHS data
    geonames_data_path = './all_countries/allCountries.txt' # path of the geonames data
    seed_path = './seed_outputs/seed_outputs.json'
    synthetic_chatter_path = "./synthetic_chatters/...json" # path of the synthetic chatter to be evaluated. This file should be generated by run_models.py
    save_dir = './evaluation'

    model_inspection = ModelInspection(
        keywords_path,
        ship_data_path,
        land_shapefile,
        geonames_data_path,
        seed_path,
        synthetic_chatter_path,
        save_dir
    )
    model_inspection.inspect()