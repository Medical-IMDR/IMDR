# Incomplete Modality Disentangled Representation for Ophthalmic Disease Grading and Diagnosis


## Abstract
Ophthalmologists typically require multimodal data sources
to improve diagnostic accuracy in clinical decisions. How-
ever, due to medical device shortages, low-quality data and
data privacy concerns, missing data modalities are common
in real-world scenarios. Existing deep learning methods tend
to address it by learning an implicit latent subspace represen-
tation for different modality combinations. We identify two
significant limitations of these methods: (1) implicit repre-
sentation constraints that hinder the model’s ability to cap-
ture modality-specific information and (2) modality hetero-
geneity, causing distribution gaps and redundancy in fea-
ture representations. To address these, we propose an Incom-
plete Modality Disentangled Representation (IMDR) strat-
egy, which disentangles features into explicit independent
modal-common and modal-specific features by guidance of
mutual information, distilling informative knowledge and en-
abling it to reconstruct valuable missing semantics and pro-
duce robust multimodal representations. Furthermore, we in-
troduce a joint proxy learning module that assists IMDR in
eliminating intra-modality redundancy by exploiting the ex-
tracted proxies from each class. Experiments on four ophthal-
mology multimodal datasets demonstrate that the proposed
IMDR outperforms the state-of-the-art methods significantly.

 ## 💡 Note

- The previous codebase was removed during a server clean-up and is no longer accessible. Please refer to the current version of the repository provided here.

- The full pipeline was originally deployed on an older server, and due to maintenance and update constraints, the code was not synchronized in real time. We apologize for any inconvenience caused.

- Regarding the data, some parts involve third-party benchmarks with copyright or usage restrictions, so we cannot release the fully processed dataset. The repository includes links to all public datasets used, along with ID indices for reproduction. If you need our preprocessed data, please contact me by email, we can share it via Baidu Netdisk or other transfer methods.





## 👨‍💻 Data Preparation
- You may contact us directly or leave your email address in an issue. We regularly check the issues and will send you the download link via cloud storage. 
- In addition, please refer to the official overall dataset:  [FairVision30K Dataset](https://yutianyt.com/projects/fairvision30k/)
- If you would like to process the official dataset yourself, you can refer to our data processing script:


```
python medical_data_process.py
```

```
2D-Fundus & 3D OCT
├── AMD
│   ├── Train
│   │   ├── fundus.png
│   │   └── OCT_Slices
│   │       ├── slice_1.png
│   │       ├── slice_2.png
│   │       └── ...
│   ├── Test
│   │   ├── fundus.png
│   │   └── OCT_Slices
│   │       ├── slice_1.png
│   │       ├── slice_2.png
│   │       └── ...
├── DR ...
├── Glaucoma ... 
```

## Implementation
For the complete pipeline, please refer to [https://github.com/Qinkaiyu/RIMA](https://github.com/Qinkaiyu/RIMA)  Due to the server migration and permission issues, some components of the original pipeline were temporarily unavailable. We will continue to improve this repository and provide fully automated one-click training and evaluation scripts in future updates.

## 🧠 Training Overview
Our method consists of two major components: **teacher model training** and **student model distillation**.  
You may refer to the following scripts for the corresponding implementations:

```
python code/distill_method.py
python code/our_model.py
```

## Baseline 
For reference, the baseline implementations can be found in the following scripts:

```
python code/baseline.py
```



