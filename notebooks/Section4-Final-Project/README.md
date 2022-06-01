# CSE 255 HW5

## Main directory structure
The following directories are on github [here](https://github.com/UCSD-Data-Science/Public-CSE255-2022/tree/master/notebooks/Section4-Final-Project)

- `KDTrees+XGBoost/`: contains KD Trees and XGBoost apporach developped by Professor Freund
- `cnn/`: contains CNN and bootstraps apporach developped by TAs. 
- `public_tables/`: contains csv files that define the trianing and test sets: `country_test_reduct.csv, random_test_reduct.csv, train.csv` .
- `XGBoostCreate_submission/` the directory that contains the files to be submitted.
- `README.md` - this file

The following directories should be in your path (already on datahub)

* `poverty-dir`: A pointer to the directory that contains `/anon_dir/` under which all of the anonymized images reside. On datahub this path is: `/datasets/cs255-sp22-a00-public/poverty/`

## How to submit your code

### Solution based on KDTree+XGBoost (small models that can be directly uploaded to Gradescope)

The submission format below assumes that your code based on KDTree+XGBoost. 
If you used a differet learning algorithm, see the instructions in the next section below.

### Submission Directory Structure:

`XGBoostCreate_submission/`:  the directory that contains all of the files in the submission.
Make zip file from this directory and submit it to **Gradesscope** as `code.zip` or `code.tar.gz` or `code.tgz`

If you are submitting as `code.zip`, gradescope will automaticall unzip in the file uploader. This is expected and you can just hit upload.

The following files, and no other files, must be in the directory:
1. `explanation.md`, This is a text file that describes the improvements over Freund-XGBoost.  You need to say what are the changes in the learn.py code and why you think these changes helped improve the performance.
2. `code/`: contains:
   * `learn.py <poverty_dir>`: A script that performs the learning. it takes as input the file 
    `../public_tables/train.csv` and the images in the path `poverty_dir/anon_images/`. The learned predictor is stored in a pickled dictionary file `data/Checkpoint.pkl`. This file is later read by `predict.py`
   * `predict.py <poverty_dir>`: A script that use `data/Checkpoint.pkl` and generates the files `data/results.csv` and `data/country_results.csv` according to the input files `../public_tables/random_test_reduct.csv` and `../public_data/country_test_reduct.csv`. The generated result files should be the same as the ones you submitted to Gradescope. 
   * Other files that your model needs.
3. `data/`: contains `Checkpoint.pkl` which contains the learned XGBoost predictor that can reproduce the same result files you submitted to Gradescope.

### Example calls:
The following commands are assumed to be executed inside the directory `XGBoostCreate_submission/`. The first line is the command as it would be executed on datahub. The following blocks are the output from running the commands on a laptop.

* `python3 code/learn.py /datasets/cs255-sp22-a00-public/poverty/`

```
python3 code/learn.py ~/datasets/poverty_v1.1/
found 19669 files
  0.06 listed files
used 500 images to train KDTree
KDTree training data shape= (25088000, 8)
  9.46 generated encoder tree
193.66 encoded images
217.40 trained trees
217.44 generated pickle file
picklefile= data/Checkpoint.pk
  0.06 listed files
  9.46 generated encoder tree
193.66 encoded images
217.40 trained trees
217.44 generated pickle file
```

* `python3 code/predict.py /datasets/cs255-sp22-a00-public/poverty/`

```
(base) MacBook-Pro:XGBoostCreate_submission yoavfreund$ python3 code/predict.py ~/datasets/poverty_v1.1/
  0.07 read pickle file
4509 image60.npznpz

------------------------------------------------------------
data/../data/results_country.csv
 69.43 generated data/../data/results_country.csv
3779 image13692.npz

------------------------------------------------------------
data/../data/results.csv
127.08 generated data/../data/results.csv
```

### Other solutions (neural networks or your models are too large to upload to Gradescope)

Note that this is only for code submission. You still need to submit your `results.csv` and `country_results.csv` to Gradescope.

### Submission Directory Structure:

1. `explanation.md`: a markdown file with an explanation of you did that improves the results.
2. `code/`: contains:
    * `learn.py <poverty_dir>`: A script that performs the learning of your models. it takes as input the file `../public_tables/train.csv` and the images in the path `poverty_dir/anon_images/`. It should output your models and save them in `data/`. 
    * `predict.py <poverty_dir>`: A script that uses your saved models in `data/` and generates the files `data/results.csv` and `data/country_results.csv` according to the input files `../public_tables/random_test_reduct.csv` and `../public_data/country_test_reduct.csv`. The generated result files should be the same as the ones you submitted to Gradescope. 
    * Other files that your model needs.
3. `data/`: contains your **saved models** that can reproduce the same result files you submitted to Gradescope.

### How to submit

1. Zip your folder with the structure described above.
2. Name your zip file the **same name** you used in the Gradescope leaderboard submission. 
3. Upload your zip file to this link using dropbox: https://www.dropbox.com/request/6L9YhZ8jStt7z1fxiFwc
