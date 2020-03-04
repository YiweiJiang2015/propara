import json
import sys
import enum
from pprint import pprint

#from propara.data.propara_dataset_reader import Action

class Action(enum.Enum):
    NONE = 0
    CREATE = 1
    DESTROY = 2
    MOVE = 3
# Input:  json format generated by ProparaPredictor
# paraid": "1114",
# "sentence_texts": ["Rainwater falls onto the soil.", "The rainwater seeps into the soil.",...."],
# "participants": ["rainwater; water", "bedrock", "funnels", "caves"],
# "states": [["?", "soil", "soil", "bedrock", "bedrock", "bedrock", "bedrock", "bedrock"],....],
# "predicted_actions": ["MOVE", "NONE", "NONE", "NONE", "MOVE", ..., "CREATE", "CREATE"]
#
# Output: paraid \t sentence_id \t participant \t action \t before_val \t after_val
# This class converts the json file format generated by ProparaPredictor into partial grids TSV format


def get_before_after_val(action: Action, predicted_before_location: str, predicted_after_location: str):
    if action == Action.CREATE:
        return '-', '?'
    elif action == Action.DESTROY:
        return '?', '-'
    elif action == Action.MOVE:
        return predicted_before_location, predicted_after_location
    elif action == Action.NONE:
        return '?', '?'


def convert_predicted_json_to_partial_grids(infile_path: str, outfile_path: str):
    out_file = open(outfile_path, "w")

    for line in open(infile_path):
        data = json.loads(line)
        pprint(data)
        para_id = data["para_id"]
        participants = data["participants"]
        actions_sentences_participants = data["top1_original"]
        sentence_texts = data["sentence_texts"]

        num_sentences = len(sentence_texts)
        num_participants = len(participants)
        predicted_after_locations = data["predicted_locations"] if "predicted_locations" in data and len(data["predicted_locations"]) > 0 \
            else [['?' for _ in range(num_participants)] for _ in range(num_sentences)]

        print(num_sentences)
        print(num_participants)

        for sentence_id in range(num_sentences):
            for participant_id in range(num_participants):
                predicted_before_location = predicted_after_locations[sentence_id-1][participant_id] if sentence_id > 0 else '?'
                predicted_after_location = predicted_after_locations[sentence_id][participant_id]
                action = Action(actions_sentences_participants[sentence_id][participant_id])
                (before_val, after_val) = get_before_after_val(action, predicted_before_location, predicted_after_location)
                out_file.write("\t".join([para_id,
                                         str(sentence_id+1),
                                         participants[participant_id],
                                         action.name,
                                         before_val,
                                         after_val]) + "\n")

    out_file.close()

if __name__ == '__main__':

    infile = sys.argv[1]
    outfile = sys.argv[2]
    convert_predicted_json_to_partial_grids(infile_path=infile, outfile_path=outfile)