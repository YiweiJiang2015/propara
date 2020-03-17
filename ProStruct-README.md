#Fork notes
* ProStruct cannot be run on debug mode...

# Original notes
Initialize the environment created as per the file README.md in the root directory.
```
    source activate propara
    export PYTHONPATH=.
```

Command to train a ProStruct model

```
    python propara/run_struct.py train data/emnlp18/prostruct_params_local.json -s data/emnlp18/output/prostruct/
```

Command for applying a pretrained ProStruct model to predict labels on test

```
    python propara/run_struct.py predict --output-file data/emnlp18/output/test.pred.json --predictor "prostruct_prediction" data/emnlp18/output/prostruct/model.tar.gz data/emnlp18/grids.v1.test.json
```

Command that takes the ProStruct model predictions in json format and converts them to TSV format needed for the evaluator

```
    python propara/utils/prostruct_predicted_json_to_tsv_grid.py data/emnlp18/output/test.pred.json data/emnlp18/output/test.pred.tsv
```

To evaluate your model's predictions on the ProPara task (EMNLP'18),
please Download the evaluator code from a separate leaderboard repository: https://github.com/allenai/aristo-leaderboard/tree/master/propara


ProPara leaderboard is now live at: https://leaderboard.allenai.org/propara