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


## 👨‍💻 Data Preparation



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
