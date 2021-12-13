# Some Classification Intergration
- Data (total 1770 images)
    - label 0 (healthy): 574
    - label 1 (cancer1): 403 
    - label 2 (cancer2): 793
    suggestion：Divide the data set into training set, validation set, and test set. When extracting validation set and test set, they should be randomly selected in proportion to ensure that the three types of data are    proportionally distributed to training set, validation set and test set
    
## Folders & Files description

- Folders
    - checkpoints
        - here to save training checkpoint, log, etc
    - core
        - training, validating fuctions are in this folder
    - loss
        - some loss fuctions
    - models
        - all models are in this folder, you could import from models
    - my_data
        - train data, test data folder
    - result
        - predict result
    
- Files
    - mean_std_calculation.ipynb
        - calculate train set mean & std
    - predict_classification.py
        - predict are processing in this script
    - torch_dataset.py
        - pytorch dataset setupt according tor data format
    - train_efficientnet_album.py
        - train script, efficientnet training using albumentations for data augmentation
    - continue_train.py
        - continue to train your model