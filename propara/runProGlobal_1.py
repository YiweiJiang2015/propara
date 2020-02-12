import logging
import os
import sys
import json, shutil

from allennlp.commands import main
from propara.models.proglobal_model import ProGlobal
from propara.data.proglobal_dataset_reader import ProGlobalDatasetReader
from propara.service.predictors.proglobal_prediction import ProGlobalPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

config_file = "../data/naacl18/proglobal/proglobal_params.json"

# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": -1}})

serialization_dir = "/tmp/debugger_train"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)
# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "-o", overrides,
]

main()
#if __name__ == "__main__":
    #predictor_overrides = {'ProGlobal': 'ProGlobalPrediction'}
    #main(prog="python -m allennlp.run")#, predictor_overrides=predictor_overrides)
