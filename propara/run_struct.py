import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

from allennlp.commands import main

from propara.data.propara_dataset_reader import ProParaDatasetReader
from propara.models.prostruct_model import ProStructModel
from propara.service.predictors.prostruct_prediction import ProStructPredictor


if __name__ == "__main__":
    main(prog="python -m allennlp.run") # meaning of m?
