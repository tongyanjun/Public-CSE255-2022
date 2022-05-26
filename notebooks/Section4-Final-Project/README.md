# HW5

## File structures

- `./KDTrees+XGBoost/`: contains KD Trees and XGBoost apporach developped by Professor Freund
- `./cnn/`: contains CNN and bootstraps apporach developped by TAs. 
- `./public_tables/`: contains csv files that define the trianing and test sets.
- `./gradescope.md`: the instructions for submiting your solution to gradescope. 

## Images on Datahub

All images are on datahub under `/datasets/cs255-sp22-a00-public/poverty/anon_images/`.

You can use the code below to load an image into a numpy array of shape (8, 224, 224).

```python
import numpy as np

path_to_file = '/datasets/cs255-sp22-a00-public/poverty/anon_images/image0.npz'
load = np.load(path_to_file)
x = load.f.x
print(x.shape)
```

Please use the csv files in `./public_tables` to get the information (label, urban or rural, etc.) about each image.
