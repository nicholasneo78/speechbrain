import json

# MANIFEST_PATH_LIST = [
#     "/datasets/mms/transcribed/test_for_D5/EN_test_set/set_1/pred_test_manifest_lid.json",
#     "/datasets/mms/transcribed/test_for_D5/EN_test_set/set_2/pred_test_manifest_lid.json",
#     "/datasets/mms/transcribed/test_for_D5/EN_test_set/set_3/pred_test_manifest_lid.json",
# ]

# OUTPUT_PATH = "/datasets/mms/transcribed/test_for_D5/EN_test_set/pred_test_manifest_lid.json"

MANIFEST_PATH_LIST = [
    "/datasets/mms/transcribed/test_for_D5/EN_test_set/set_1/pred_test_manifest_lid_2024072401.json",
    "/datasets/mms/transcribed/test_for_D5/EN_test_set/set_2/pred_test_manifest_lid_2024072401.json",
    "/datasets/mms/transcribed/test_for_D5/EN_test_set/set_3/pred_test_manifest_lid_2024072401.json",
]

OUTPUT_PATH = "/datasets/mms/transcribed/test_for_D5/EN_test_set/pred_test_manifest_lid_2024072401.json"

combined_manifest_items = []

for MANIFEST_PATH in MANIFEST_PATH_LIST:
    with open(MANIFEST_PATH, mode="r") as fr:
        manifest_items = [json.loads(line.strip("\r\n")) for line in fr.readlines()]

    combined_manifest_items.extend(manifest_items)

# export
with open(OUTPUT_PATH, 'w+', encoding='utf-8') as f:
    for data in combined_manifest_items:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')