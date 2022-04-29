

The code is based on [SCIGAN](https://github.com/ioanabica/SCIGAN) and [TransTEE](https://github.com/hlzhang109/TransTEE)

## Dependencies

The model was implemented in Python 3.6. We have included a environment.yml file for reproducibility.

The data is provided in the `datasets/` please download via [link](https://drive.google.com/file/d/1eEBJceNPaiA6x4sgufNcvfoGhzpT4Ds-/view?usp=sharing) and uncompress it.

The structure is as follows:
```
datasets/
    XXX.pkl
```
To reproduce the experiment we reported in the paper, simply run
```bash
bash run.sh
```