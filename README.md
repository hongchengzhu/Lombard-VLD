# Lombard-VLD
Official implementation of Lombard-VLD (IEEE S\&P 25).

## What we provided

### Our paper

Congratulations! Our paper titled "*Lombard-VLD: Voice Liveness Detection Based on Human Auditory Feedback*" has been accepted by *IEEE Symposium on Security and Privacy* (***S&P 25***)!

Our university reported the acceptance at [link1](https://news.whu.edu.cn/info/1015/477347.htm), which may help you understand our paper more deeply.

Our paper can be accessed at [link2](https://ieeexplore.ieee.org/document/11023446).

### Speech examples

We provide examples of partial speech (including plain and Lombard) at [link3](https://hongchengzhu.github.io/Lombard-VLD-speech-examples/). If you need the entire dataset for research, please feel free to contact us! **Note that we keep all the rights to the released speech examples. Each one should reasonably and legally use these speeches**.

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

The download link is: [link4](https://drive.google.com/drive/folders/1FyE3JABE86trg5SPFKDSXm8ADRyhm_FU?usp=drive_link). You are advised to download and move all folders to this project folder path.

Then, RUN `python validateECAPAModelL_dif_1.py`, and this code will print `EER 0.24%`. These results correspond to the overall DB-MMLC results in Section 6.3 of our paper.

## Citation
If our work is helpful to you, please cite our paper as follows:
```
@INPROCEEDINGS{11023446,
  author={Zhu, Hongcheng and Sun, Zongkun and Ren, Yanzhen and He, Kun and Yan, Yongpeng and Wang, Zixuan and Liu, Wuyang and Yang, Yuhong and Tu, Weiping},
  booktitle={2025 IEEE Symposium on Security and Privacy (SP)}, 
  title={Lombard-VLD: Voice Liveness Detection Based on Human Auditory Feedback}, 
  year={2025},
  volume={},
  number={},
  pages={4303-4320},
  keywords={Loudspeakers;Privacy;Accuracy;Working environment noise;Authentication;Physiology;Environmental factors;Acoustics;Security;Noise measurement;liveness detection;peech spoofing},
  doi={10.1109/SP61157.2025.00226}}
```



## Acknowledge

Our code project was partially inspired by: [TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN).







