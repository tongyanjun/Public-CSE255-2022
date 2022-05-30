## Code Submission

The submission format below assumes that your code  based on KDTree+XGBoost. 
If you used a differet learning algorithm contact your TA and we will define a way for you 
to submit your code.

#### Directory Structure:
* `submission/`:  the directory that contains all of the files in the submission.
   * `code`: contains:  `learn.py` and `predict.py`
   * `public_tables`: contains `country_test_reduct.csv, random_test_reduct.csv, train.csv` Note that these files might be changed in the testing. But their format will remain the same`
   * `data`: contains `Checkpoint.pk` which contains a dictionary that defines the predictor.
* `poverty-dir`: A directory that contains `/anon_dir/` under which all of the anonymized images reside.

#### The code submission must contain the following files:

* `README.md`, explains how the code gets better performance than Freund-xgboost. identifies the differences from the Freund-XGBoost and explains why you think that these differences explain the improvement.
* `learn.py`: A script that performs the learning. Other files it needs are assumed to be in the relative locations defined in the submission. The script generates as output a pkl file which defines the KDTree and the classifier. The pickle file is  `../data/Checkpoint.pkl`
* `predict.py`: A script that use `data/Checkpoint` and generates the files `../data/scores.csv` and `../data/country_scores.csv` according to the input files `public_random_test_reduct.csv` and `../public_data/country_test_reduct.csv`.  

Both `learn.py` and `predict.py` take the command_line parameter `poverty_dir` and expect to find the raw images in `poverty_dir/anon_images/`
