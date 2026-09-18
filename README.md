# LTS: Learning Temporal Distribution and Spatial Correlation

This repository contains the released models, sample data, demo notebook and
supporting code for **Learning Temporal Distribution and Spatial Correlation
Toward Universal Moving Object Segmentation**. The method targets moving
object segmentation in videos recorded by stationary cameras.

- Paper: [IEEE Transactions on Image Processing](https://ieeexplore.ieee.org/document/10478283)
- Preprint: [arXiv:2304.09949](https://arxiv.org/abs/2304.09949)
- Results on the [LTS dataset](https://www.youtube.com/watch?v=BcLnNTne-n0)
- Results on the [standard dataset](https://www.youtube.com/watch?v=0UI-cezMFlI)

## Quick start

The checked-in `data/highway` sample and model weights are enough to run the
demo. Run Jupyter from the repository root because the notebook uses paths such
as `./models` and `./data`.

```bash
git clone git@github.com:guanfangdong/LTS-UniverisalMOS.git
cd LTS-UniverisalMOS

# Install a PyTorch/torchvision pair appropriate for your CPU or CUDA version.
# See https://pytorch.org/get-started/locally/ for the matching command.
python -m pip install -r requirements.txt
python -m pip install torch torchvision

jupyter lab test_demo.ipynb
```

Open `test_demo.ipynb` and run all cells. It runs the two released stages in
order:

1. **DIDL** estimates temporal distributions and writes intermediate masks to
   `output/highway/DIDL/`.
2. **SBR** refines those masks using spatial correlation and writes the final
   results to `output/highway/SBR/`. The generated heatmaps are written to
   `output/highway/mask/`.

The demo can use CUDA when it is available. The model-loading and tensor code
are unchanged from the released workflow, so a compatible PyTorch installation
and enough GPU/CPU memory are required for a full run.

## Repository layout

```text
.
├── data/highway/                 # Included CDNet2014 sample input and labels
├── models/                       # Released DIDL and SBR weights
├── output/highway/               # Included sample outputs; demo writes here
├── test_demo.ipynb               # End-to-end DIDL + SBR demonstration
├── deps/
│   ├── ADNNet_data_plus.py       # Data preparation and DIDL helpers
│   ├── demo_didl.py               # DIDL models and temporal inference helpers
│   ├── demo_sbr.py                # SBR U-Net and patch-based refinement helpers
│   ├── common_py/                # I/O, metrics and image-processing helpers
│   ├── bayesian/                 # Bayesian refinement implementation
│   ├── binarymask/               # Training-mask conversion
│   ├── params_input/             # Legacy experiment parameter parser
│   └── function/                 # Distribution operators and old experiments
├── requirements.txt
└── tests/test_regression.py      # Lightweight interface/regression checks
```

The five model files currently included are:

```text
models/f_B_0069.pt
models/f_W_0069.pt
models/netC_0069.pt
models/netP_0069.pt
models/network_dis_0100.pt
```

## Code organisation

The Python code is now importable as the `deps` package. The notebook is kept
as a readable orchestration layer: model definitions, patch sampling, mask
normalization and metric updates live in `deps/demo_didl.py` and
`deps/demo_sbr.py`. Shared utilities have kept their original public function
names so existing experiments remain compatible. The identical files that
previously existed in both `deps/` and `deps/function/` are now small
compatibility entry points; historical variants that have different behaviour
remain separate.

The refactor is intentionally behaviour-preserving for the released pipeline:
the model files, tensor operations, data ordering, output directories and
experiment-specific implementations are retained. The binary-mask metrics
also now work with current NumPy versions, where the old two-dimensional scalar
conversion raised an exception.

Run the lightweight checks with:

```bash
python -m unittest discover -s tests -v
```

The full notebook is the authoritative end-to-end validation because it
requires the optional image, plotting and Jupyter dependencies and processes
the complete sample video. The released sample stores some groundtruth masks
as RGB PNGs; the shared `normalize_mask` helper reduces those equivalent
channels before DIDL/SBR evaluation. SBR also follows the selected PyTorch
device instead of assuming CUDA is present.

The files under `deps/*/exp*.py` are preserved historical experiments rather
than the recommended entry point. Some of their standalone `main` functions
still contain the dataset locations used in the original paper experiments;
use the notebook for the released sample workflow.

## Full LTS dataset

The complete LTS dataset is distributed separately:

- [Google Drive](https://drive.google.com/drive/folders/1M3YsXmIBhsonYSMySTgI016kSXKwoV1j?usp=drive_link)
- [Project server](http://129.128.243.176/LTS-Dataset/)

The release is split into the following archives: `part1.tar.gz` (13G),
`part2.tar.gz` (12G), `part3.zip` (24G), `part4.tar.gz` (12G), `part5.zip`
(7.4G), and `part6.zip` (2.4G). Extract them according to the dataset
instructions before using the legacy data-generation scripts.

## Citation

```bibtex
@ARTICLE{10478283,
  author={Dong, Guanfang and Zhao, Chenqiu and Pan, Xichen and Basu, Anup},
  journal={IEEE Transactions on Image Processing},
  title={Learning Temporal Distribution and Spatial Correlation Toward Universal Moving Object Segmentation},
  year={2024},
  volume={33},
  pages={2447-2461},
  doi={10.1109/TIP.2024.3378473}
}
```

Related work:

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

## License and contact

This work is released under the MIT License.

For questions about the dataset or models, contact `guanfang@ualberta.ca`
(University of Alberta).
