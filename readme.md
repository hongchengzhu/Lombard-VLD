# Info to validate our results

## Python Environments

python = 3.7.16

torch = 1.13.1

speechbrain

Others: You can install any necessary packages by "pip install xxx".

## Folder structure

- dataset: a part of our self-made DB-MMLC dataset for validate our results
  - data_3_1_clip_16k: the plain and Lombard speech from live speakers.
  - replay: the ***spoofed*** plain and Lombard speech from loudspeakers with the same content as `data_3_1_clip_16k`. In this case, we consider the `replay` attack.
- file_list: the speech pair list for training or inference.
- preprocess: pre-process the speech input.
- results: the stored trainging models.

## How to reproduce our result

First, you should download some necessary files, inculding dataset, file_list, stored model, etc.

The download link is: [download link](https://drive.google.com/drive/folders/1FyE3JABE86trg5SPFKDSXm8ADRyhm_FU?usp=drive_link), you are advised to download all folders and move them to this project folder path.

Then, RUN `python validateECAPAModelL_dif_1.py`, and this code will print `EER 0.24%`. This results corresponds to the overall DB-MMLC results in Section 6.3 of our paper.
