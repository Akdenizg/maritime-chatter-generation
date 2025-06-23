This document provides a detailed explanation of the filters used in the project to ensure the quality and realism of the generated synthetic maritime radio chatter. These filters are applied during both the data generation (generate_instances.py) and evaluation (inspect_model.py) phases.

These filters ensure that the generated chatter adheres to the standard format of maritime communication and other basic structural requirements.

• paranthesis: Rejects chatters containing parentheses ( or ).

• brackets: Rejects chatters containing square brackets [ or ].

• mayday: Ensures the distress call starts with "Mayday, Mayday, Mayday".

• incomplete: Rejects chatters that do not end with a period.

• vessel_name_not_after_mayday: Ensures the vessel's name is mentioned in the first sentence after the initial "Mayday" call.

• wrong_category: Checks if the generated chatter contains keywords relevant to the specified distress category (e.g., "fire", "explosion" for the "Fire, Explosion" category). For the "Undesignated Distress" category, it ensures no specific distress keywords are present. Check keywords.json to see which keywords are used for each disaster category.

• unknown: Rejects chatters that mention MMSI or call sign if these details were not provided in the context.

• vessel_name: Ensures the correct vessel name from the context is present in the chatter.

• vessel_MMSI: Ensures the correct MMSI from the context is present in the chatter.

• vessel_call_sign: Ensures the correct call sign from the context is present in the chatter.

• vessel_type: Ensures the correct vessel type from the context is present in the chatter.

• vessel_coordinate_dms: Ensures the correct vessel coordinates in Degrees, Minutes, Seconds (DMS) format from the context are present in the chatter. = False

• compass: Checks if the compass direction to the nearest place is correctly mentioned as provided in the context.

• distance_to_nearest_place: Ensures the distance to the nearest place is correctly mentioned.

• distance_to_nearest_port: Ensures the distance to the nearest port is correctly mentioned.

• distance_to_nearest_harbor: Ensures the distance to the nearest harbor is correctly mentioned.

• hallucinated_mmsi: Detects if the model generates an MMSI when none was provided in the context.

• hallucinated_call_sign: Detects if the model generates a call sign when none was provided in the context.

• hallucinated_vessel_type: Detects if the model generates a vessel type that is not the one provided in the context.

• no_coast_guard: Ensures that the chatter includes a response from the Coast Guard.

• digit_by_digit: If the context specifies that numbers should be read digit-by-digit, this filter rejects chatters that use words like "twenty", "thirty", etc.

• false_cargo_logic: Rejects chatters that mention "cargo" if the vessel type in the context is not a cargo vessel.

• both_port_and_harbor: Rejects chatters that mention both a port and a harbor if they are not the same or part of each other.

• duplicate_sentences: Rejects chatters that contain duplicate sentences.

• non-unique: Rejects chatters that are too similar to the seed instances or other generated instances, based on a ROUGE-L score threshold of 0.7.



Evaluation Coefficients Summary

Here's a summary of the evaluation scores and the related coefficients used in inspect_model.py. Scores are calculated as weighted average of the filter coefficients. If a filter check fails, the coefficient of that filter is not added to the sum.

Compliance with SMCP:

• paranthesis: 1

• brackets: 1

• mayday: 1

• incomplete: 2

• vessel_name_not_after_mayday: 1

• duplicate_sentences: 2

• no_coast_guard: 1

• digit_by_digit: 1

Information Accuracy:

• wrong_category: 2

• unknown: 1

• vessel_name: 2

• vessel_MMSI: 2

• vessel_call_sign: 2

• vessel_type: 2

• vessel_coordinate_dms: 2

• hallucinated_mmsi: 1

• hallucinated_call_sign: 1

• false_cargo_logic: 1

• hallucinated_vessel_type: 1

• compass: 2

• both_port_and_harbor: 1

• distance_to_nearest_port: 1

• distance_to_nearest_harbor: 1

• distance_to_nearest_place: 1

Uniqueness:

• non-unique: 1

Usage of Optional Info:

• compass: 1

• closest_place_name: 1

• distance_to_nearest_place: 1

• nearest_port: 1

• distance_to_nearest_port: 1

• nearest_harbor: 1

• distance_to_nearest_harbor: 1

• closest_water_body: 1

• closest_place_country: 1

