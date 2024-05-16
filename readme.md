<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- ABOUT THE PROJECT -->
## About The Project

This project aims to use Transformer instead of RNN/CNN to imporve model performance on bearing RUL(Remain Useful Life) prediction.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

python 3.7.15
CUDA Version: 10.1

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started in Colab
### Prerequirement & Installation
No!
### Dataset
1. Download zip from `https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset`
2. upload zip to your google drive, don't extract and upload it will take long time.
### Pull github repo
1. create a colab notebook in your drive
2. run below command
```
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/
!git clone https://github.com/qawl987/DAST-self
```
3. close this notebook
### Run the autorun notebook
1. open /content/drive/MyDrive/DAST-self/notebook/FEMTO-st/DAST_run.ipynb
2. unzip the dataset zip in workspace & copy to drive for future reuse
```
!unzip /content/drive/MyDrive/phm-ieee-2012-data-challenge-dataset-master.zip -d /content/
!cp -r "/content/phm-ieee-2012-data-challenge-dataset-master" "/content/drive/MyDrive/TMP-for-DAST"
```
3. move TMP folder dataset to DAST-self project folder
```
!mv /content/drive/MyDrive/TMP-for-DAST/phm-ieee-2012-data-challenge-dataset-master/Full_Test_Set/* /content/drive/MyDrive/DAST-self/data/10FEMTO/raw_data/Validation_set
!mv /content/drive/MyDrive/TMP-for-DAST/phm-ieee-2012-data-challenge-dataset-master/Learning_set/* /content/drive/MyDrive/DAST-self/data/10FEMTO/raw_data/Validation_set
```
4. preprocess
Choose dataset & run the preprocess block
5. DL
Set hyperparameter & run the DL block
6. TL
Set hyperparameter & run the TL block
## Getting Started in Local

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequirement

You need to intall appropriate torch version with GPU according to your CUDA version

### Installation

1. Install package
   ```sh
   pip install -r requirements.txt
   ```

### Dataset

1. Download dataset from `https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset` & extract zip
```
move all folder under Full_Test_Set/ and Learning_set/ to /data/10FEMTO/raw_data/Validation_set/ 
All raw data path /data/10FEMTO/raw_data/Validation_set/
```
<!-- USAGE EXAMPLES -->
## Usage

### Preprocess

1. Create folder /data/10FEMTO/processed_data/
2. run all block in `notebook/FEMTO-st/data_processing/main.ipynb`

### Train
#### Deep learning 
```
# model hyperparameters
selected_indices = [1, 3, 5, 7, 9, 10, 14]
# selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
FEATURE_LEN = len(selected_indices)
FEATURE_SIZE = 20
EMBEDD = 20
HYPER_PARAMETERS = {
    # model parameter
    'batch_size': 256,
    'dim_val': FEATURE_SIZE,
    'dim_attn': EMBEDD,
    'dim_val_t': FEATURE_SIZE,
    'dim_attn_t': EMBEDD,
    'dim_val_s': FEATURE_SIZE,
    'dim_attn_s': EMBEDD,
    'n_heads': 4,
    'n_decoder_layers': 1,
    'n_encoder_layers': 2,
    'lr': 1e-3,
    'epochs': 100,
    'time_step': 40,
    # limit how many last input used, important!
    'dec_seq_len': 6,
    'output_sequence_length': 1,
    'feature_len': FEATURE_LEN,
    'debug': True
}
```
```
# Other Hyperparameters
DATA_PATH = '../../../data/10FEMTO/processed_data/' # set the data path after preprocess
TRAIN_DATASETS = ['Bearing1_1', 'Bearing1_2']
TEST_DATASET = ['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7']
# TRAIN_DATASETS = ['Bearing2_1', 'Bearing2_2']
# TEST_DATASET = ['Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing2_6', 'Bearing2_7']
# TRAIN_DATASETS = ['Bearing3_1', 'Bearing3_2']
# TEST_DATASET = ['Bearing3_3']

MODEL_SAVE_NAME = f'Bearing{TRAIN_DATASETS[0][7]}_pretrain_{len(selected_indices)}'
NORM_TYPE = NormType.NO_NORM # choose norm method before feed into model, no_norm or batch_norm or layer_norm
TRAIN_TYPE = TrainType.DL # set train mode to deep learning
MODEL_SAVE_PATH = '../../../model/norm' # save the model save path
IS_SAVE_MODEL = False # set if save the model
```
- Run DL block in `notebook/FEMTO-st/model_training/main.ipynb`
#### Transfer learning
```
DATA_PATH = '../../../data/10FEMTO/processed_data/'
PRETRAIN_DATASET = 'Bearing2' # set pretrain dataset name
FINETUNE_DATASET = 'Bearing3' # set finetune dataset name

TRAIN_DATASETS = ['Bearing1_1', 'Bearing1_2']
TEST_DATASET = ['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7']
# TRAIN_DATASETS = ['Bearing2_1', 'Bearing2_2']
# TEST_DATASET = ['Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing2_6', 'Bearing2_7']
# TRAIN_DATASETS = ['Bearing3_1', 'Bearing3_2']
# TEST_DATASET = ['Bearing3_3']

MODEL_SAVE_PATH = '../../../model'
MODEL_SAVE_NAME = f'{PRETRAIN_DATASET}_pretrain_{FINETUNE_DATASET}_finetune_7'
PRETRAIN_MODEL_PATH = '../../../model' # set the pretrain model path
PRETRAIN_MODEL_NAME = f'{PRETRAIN_DATASET}_pretrain_7'
IS_SAVE_MODEL = False
NORM_TYPE = NormType.NO_NORM
TRAIN_TYPE = TrainType.TL
```
- Run TL block in `notebook/FEMTO-st/model_training/main.ipynb`

