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

### Fies to be upoaded to gradescope:
1.  `results.csv`
2. `country_results.csv`
3. `explanation.md`: this file must contain an english explanaton for how your code works. The explanation should be between 300 and 500 words long.

The fourth file depends on whether or not you checkpoint is smaller enough for nbgrader.

* If file is smaller than 50MB then you upload your code (organization described below) as `code.tgz`
* If file is larger than 50MB then you upload it as a `.tgz` file to `https://www.dropbox.com/request/6L9YhZ8jStt7z1fxiFwc`. Create a little text file called `code.id` and state in it the name of the file that you uploaded and the time of the upload.

### Code Directory Structure:

`XGBoostCreate_submission/`, or `NN_submission/` or 'other_submission`:  a directory that contains the following files, 

2. `code/`: contains:
   * `learn.py <poverty_dir>`: A script that performs the learning. it takes as input the file 
    `../public_tables/train.csv` and the images in the path `poverty_dir/anon_images/`. The learned predictor is stored in a pickled dictionary file `data/Checkpoint.pkl`. This file is later read by `predict.py`
   * `predict.py <poverty_dir>`: A script that use `data/Checkpoint.pkl` and generates the files `data/results.csv` and `data/country_results.csv` according to the input files `../public_tables/random_test_reduct.csv` and `../public_data/country_test_reduct.csv`. The generated result files should be the same as the ones you submitted to Gradescope. 
   * Other files that your model needs.
3. `data/`: contains `Checkpoint.pkl` which contains the learned XGBoost predictor that can reproduce the same result files you submitted to Gradescope.

### Check before you submit
Before submitting, check that you can run `predict.py` and that it generates the same results files as the one you submitted. If these do not match, your score on the submission will be zero, you will know if your ccode passed the test a day after your submission.


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
