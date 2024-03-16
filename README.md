# LTS: Learning Temporal Distribution and Spatial Correlation Towards Universal Moving Object Segmentation
[![LTS Dataset](https://img.shields.io/badge/Click%20to%20watch-LTS%20results%20on%20new%20LTS%20dataset!-brightgreen)](https://www.youtube.com/watch?v=BcLnNTne-n0)

[![Standard Dataset](https://img.shields.io/badge/Click%20to%20watch-LTS%20results%20on%20standard%20dataset!-brightgreen)](https://www.youtube.com/watch?v=0UI-cezMFlI)
<div align="center">
    <a href="https://www.youtube.com/watch?v=BcLnNTne-n0">
        <img src="https://i.imgur.com/LW93XnL.png" width="700" alt="video">
    </a>
</div>

## News 🎺🎺🎺
``March 15, 2024``
We have uploaded an .ipynb file for testing purposes, using the video "highway" from CDNet2014. As our algorithm is universial, we are preparing to upload another .ipynb file to test any video you have on hand.

``March 06, 2024``
We are pleased to announce that our paper has been accepted by **IEEE Transactions on Image Processing**. We are currently working with the editors on proofreading for the final public release. The arXiv version of the paper has also been updated (Please check https://arxiv.org/abs/2304.09949). We will also be updating this GitHub repository accordingly.

## Introduction
This repository hosts the dataset and model files for the LTS (Learning Temporal Distribution and Spatial Correlation Towards Universal Moving Object Segmentation) approach, a novel method for moving object segmentation in videos using stationary cameras. Our method focuses on learning from diverse videos by capturing the similarities in temporal pixel distributions and refining segmentation through spatial correlation learning.

## Dataset
We proposed the LTS dataset, covering a wide range of natural scenes. These videos are instrumental in demonstrating the effectiveness and universality of our approach for moving object segmentation.

### Dataset Access
The dataset is split into multiple parts and is accessible via Google Drive. Here are the links to download the dataset:
- [Google Drive Link](https://drive.google.com/drive/folders/1M3YsXmIBhsonYSMySTgI016kSXKwoV1j?usp=drive_link)

### Dataset Structure
The dataset is organized in compressed files for convenient download:
- `part1.tar.gz`
- `part2.tar.gz`
- `part3.zip`
- `part4.tar.gz`
- `part5.zip`
- `part6.zip`
- `raw_videos` directory containing all the raw video files.

After downloading, extract the files to access the video data.

## Model
We currently provide the trained LTS model file. The model is located in the "models" folder. As our algorithm is universal, we do not require any retraining or fine-tuning to adapt to your model.

We have utilized common libraries such as NumPy or Torch to ensure the code runs smoothly. If you encounter any issues with the code, please feel free to submit an issue in this GitHub repository or contact us directly.

### Model File Structure
- `/models`: Contains the LTS model files.
- `/data`: Contains the sample testing data.
  - `/highway`: A sample video from CDNet2014 dataset.
- `/deps`: Contains the custom dependencies.
- `/output`: To save LTS results.
  - `/highway`: A sample video results from CDNet2014 dataset.
    - `/DIDL`: Results from DIDL module.
    - `/mask`: Results of SBR heatmap.
    - `/SBR`: Results of SBR Network, also the final LTS results.

## Progress Tracking

The table below summarizes the current progress and upcoming developments of the project. A check mark (✔️) indicates completed items, while a cross mark (❌) signifies items that are still in progress.

| Feature/Milestone         | Status |
|---------------------------|--------|
| Dataset Release           | ✔️     |
| Model Weights Release     | ✔️     |
| New ADNN Architecture     | ✔️     |
| Testing Procedure Release | ✔️     |
| Training Procedure Release| ❌     |
| UI Interface Development  | ❌     |

## Usage
Please run the .ipynb file provided by us for testing.

## Citation
If you find our dataset and model useful in your research, please consider citing:
```
@article{dong2023learning,
  title={Learning Temporal Distribution and Spatial Correlation for Universal Moving Object Segmentation},
  author={Dong, Guanfang and Zhao, Chenqiu and Pan, Xichen and Basu, Anup},
  journal={arXiv preprint arXiv:2304.09949},
  year={2023}
}
```

## Related Work

If you are interested in our approach to distribution learning, we recommend exploring our previous work in this area. Our earlier research laid the groundwork for the methodologies applied in the current LTS project. Below is a key publication that provides insight into our foundational work on universal background subtraction using arithmetic distribution neural networks:

**Reference:**
- Zhao, C., Hu, K., & Basu, A. (2022). Universal background subtraction based on arithmetic distribution neural network. *IEEE Transactions on Image Processing*, 31, 2934-2949.

```bibtex
@article{zhao2022universal,
  title={Universal background subtraction based on arithmetic distribution neural network},
  author={Zhao, Chenqiu and Hu, Kangkang and Basu, Anup},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={2934--2949},
  year={2022},
  publisher={IEEE}
}
```

This publication provides valuable insights into the principles and techniques that have influenced the development of our LTS model. We encourage interested readers to delve into this work for a deeper understanding of our approach to segmentation and distribution learning.


## License
This work is licensed under the MIT License.

## Contact
For any queries regarding the dataset or the model, feel free to reach out:
- Email: guanfang@ualberta.ca
- Institution: University of Alberta
