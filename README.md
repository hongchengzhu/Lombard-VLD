# Lombard-VLD
Official implementation of Lombard-VLD (IEEE S\&P 25).

## What we provided

### Our paper

Congratulations! Our paper titled "*Lombard-VLD: Voice Liveness Detection Based on Human Auditory Feedback*" has been accepted by *IEEE Symposium on Security and Privacy* (***S&P 25***)!

Our university reported the acceptance at [link](https://news.whu.edu.cn/info/1015/477347.htm), which may help you understand our paper more deeply.

- [ ] We are modifying the accepted manuscript for a camera-ready version. The final paper will be released before 17th April AOE.

### Speech examples

We provide examples of partial speech (including plain and Lombard) at [link](https://hongchengzhu.github.io/Lombard-VLD-speech-examples/). If you need the entire dataset for research, please feel free to contact us! **Note that we keep all the rights on the released speech examples. Each one should reasonably and legally use these speeches**.

## Code implementation

### Python Environment

python = 3.7.16

torch = 1.13.1

speechbrain

Others: You can install any necessary packages by "pip install xxx".

### Folder structure

- dataset: a part of our self-made DB-MMLC dataset for validating our results
  - data_3_1_clip_16k: the plain and Lombard speech from live speakers.
  - replay: the ***spoofed*** plain and Lombard speech from loudspeakers with the same content as `data_3_1_clip_16k`. In this case, we consider the `replay` attack.
- file_list: the speech pair list for training or inference.
- preprocess: pre-process the speech input.
- results: the stored training models.

### How to reproduce our result

First, you should download some necessary files, including dataset, file_list, stored model, etc.

The download link is: [download link](https://drive.google.com/drive/folders/1FyE3JABE86trg5SPFKDSXm8ADRyhm_FU?usp=drive_link), you are advised to download all folders and move them to this project folder path.

Then, RUN `python validateECAPAModelL_dif_1.py`, and this code will print `EER 0.24%`. These results correspond to the overall DB-MMLC results in Section 6.3 of our paper.

## Acknowledge

Our project was inspired by: [TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN).







