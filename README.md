# Geometry Distortion-Aware 3D-Mesh Steganalysis Based on Spatial Representation Learning
[![Paper](https://img.shields.io/badge/Paper-TMM%202026-blue)](https://doi.org/10.1109/TMM.2026.3715306)

This repository provides the PyTorch implementation of the mesh distortion encoder proposed in “Geometry Distortion-Aware 3D Mesh Steganalysis Based on Spatial Representation Learning.”

# Introduction
This work is published on IEEE Transactions on Multimedia (TMM), 2026.

The high-dimensional geometric characteristics and substantial redundancy inherent in 3D-meshes make them highly susceptible to exploitation for steganographic purposes, posing significant threats to cyberspace security. To mitigate these risks, 3D-mesh steganalysis has been developed. However, existing methods predominantly rely on handcrafted statistical features, which often fall short in capturing subtle steganographic distortions. To address this limitation, this paper proposes a novel deep learning-based framework, termed the Geometry Distortion-Aware Fusion method (GeoDAF), to enhance the detection of 3D-mesh steganography. GeoDAF leverages neighborhood-level geometric cues and hierarchical feature aggregation to learn more discriminative steganographic representations. As its core, we design an Expand Dimensionality Separable Convolution (EDS-Conv) module, which efficiently models local geometric relationships. Building upon EDS-Conv, a progressive distortion-aware architecture is developed. This architecture retains fine-grained distortion features in the early stage and progressively refines them in subsequent stages. To obtain a comprehensive representation, local distortion features are aggregated using global self-attention. Furthermore, we provide an interpretable analysis of the learned distortion features by tracking geometry distortion representations, thereby shedding light on the underlying mechanisms of 3D-mesh steganalysis.

# Citation
If you find our work useful in your research, please consider citing:
```bibtex
@ARTICLE{11614768,
  author={Ji, Qingzhi and Li, Yue and Tian, Hui and Huang, Hui and Chang, Ching-Chun and Chang, Chin-Chen},
  journal={IEEE Transactions on Multimedia}, 
  title={Geometry Distortion-Aware 3D-Mesh Steganalysis Based on Spatial Representation Learning}, 
  year={2026},
  volume={},
  number={},
  pages={1-12},
  keywords={Distortion;Faces;Modeling;Steganography;Signal detection;Educational institutions;Payloads;Accuracy;Visualization;Filtering;3D-Mesh;steganalysis;separable convolution;distortion representation},
  doi={10.1109/TMM.2026.3715306}}
```
## Contact
If you have any questions or suggestions, please feel free to contact us:
- **Email:** quincyji2020@163.com
