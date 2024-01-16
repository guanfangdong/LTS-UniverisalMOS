# LTS: Learning Temporal Distribution and Spatial Correlation Towards Universal Moving Object Segmentation

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
We currently provide the trained LTS model file. At this stage, the model file is available for use, but please note that the accompanying inference and training code are still being refined and cleaned up for release.

### Model File Structure
- `/models`: Contains the LTS model file.

## Progress Tracking

The table below summarizes the current progress and upcoming developments of the project. A check mark (✔️) indicates completed items, while a cross mark (❌) signifies items that are still in progress.

| Feature/Milestone         | Status |
|---------------------------|--------|
| Dataset Release           | ✔️     |
| Model Weights Release     | ✔️     |
| New ADNN Architecture     | ❌     |
| Testing Procedure Release | ❌     |
| Training Procedure Release| ❌     |
| UI Interface Development  | ❌     |

## Usage
Currently, the model can be used for inference on the provided dataset or your own data. Detailed code and usage instructions will be released soon.

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
