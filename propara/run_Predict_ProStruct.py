import logging
import os
import sys
import json, shutil

from allennlp.commands import main
from models.prostruct_model import ProStructModel
#from data.prostruct_dataset_reader import ProStructDatasetReader
from service.predictors.prostruct_prediction import ProStructPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

config_file = "../data/emnlp18/prostruct_params_original.json"
output_file = "../data/emnlp18/output/test.pred_1.json"
predictor = "prostruct_prediction"
model = "../data/emnlp18/output/prostruct/model.tar.gz"
test_file = "../data/emnlp18/grids.v1.test.json"
# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": -1}})

serialization_dir = "tmp/prostruct"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
#shutil.rmtree(serialization_dir, ignore_errors=True)
# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "predict",
    "--output-file", output_file,
    "--predictor", predictor,
    model,
    test_file
]

main()

