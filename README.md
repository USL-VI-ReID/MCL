<h1 align="center">
Unsupervised Visible-Infrared Person <br>
Re-identification under Unpaired Settings
</h1>

## Official Project Page

> **_This is the official repository for_**  
> **"Unsupervised Visible-Infrared Person Re-identification under Unpaired Settings"**

We are preparing a journal extension of this work, and the full implementation will be made available after the journal submission.

## Abstract

Unsupervised visible-infrared person re-identification (USL-VI-ReID) aims to train a cross-modality retrieval model without labels, reducing the reliance on expensive cross-modality manual annotation. However, existing USL-VI-ReID methods rely on artificially cross-modality paired data as implicit supervision, which is also expensive for human annotation and contrary to the setting of unsupervised tasks. In addition, this full alignment of identity across modalities is inconsistent with real-world scenarios, where unpaired settings are prevalent. To this end, we study the USL-VI-ReID task under unpaired settings, which uses cross-modality unpaired and unlabeled data for training a VI-ReID model. We propose a novel Mapping and Collaborative Learning (MCL) framework. Specifically, we first design a simple yet effective Cross-modality Feature Mapping (CFM) module to map and generate fake cross-modality positive feature pairs, constructing a cross-modal pseudo-identity space for feature alignment. Then, a Static-Dynamic Collaborative (SDC) learning strategy is proposed to align cross-modality correspondences through a collaborative approach, eliminating inter-modality discrepancies across different aspects \ie, cluster-level and instance-level, in scenarios with cross-modal identity mismatches. Extensive experiments on the conducted SYSU-MM01 and RegDB benchmarks under paired and unpaired settings demonstrate that our proposed MCL significantly outperforms existing unsupervised methods, facilitating USL-VI-ReID to real-world deployment.

<p align="center">
  <img src="https://github.com/user-attachments/assets/15a6dc8a-1024-4d1e-b96d-e0d791236735" alt="overview" style="width:75%;">
</p>

## Citation
If you find our work useful, please consider citing:


## Licensing

This repository is released under the [MIT License](https://opensource.org/licenses/MIT). 

This page was built using the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).  
This website is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).
