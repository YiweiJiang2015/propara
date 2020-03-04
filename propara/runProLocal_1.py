import logging
import os
import sys
import json, shutil

from allennlp.commands import main
from models.prolocal_model import ProLocalModel
from data.prolocal_dataset_reader import ProLocalDatasetReader
from service.predictors.prolocal_prediction import ProLocalPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

config_file = "../data/naacl18/prolocal/prolocal_params.json"

# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": 0}})

serialization_dir = "../data/naacl18/prolocal/output1"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
#shutil.rmtree(serialization_dir, ignore_errors=True)
# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "-o", overrides,
]

main()

