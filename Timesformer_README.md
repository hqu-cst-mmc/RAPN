# TimeSformer CARC Setup

- SSH into the CARC Discovery node
    ```
    <username>@discovery.usc.edu
    ```
- Execute
    ```
    module load python/3.9.2
    module load anaconda3/2021.05
    ```

- Execute installation procedure of TimesFormer
    - First, create a conda virtual environment and activate it:
        ```
        conda create -n timesformer python=3.7 -y
        conda activate timesformer
        ```
    - torch: `pip install torch`
    - torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
    - [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
    - simplejson: `pip install simplejson`
    - einops: `pip install einops`
    - timm: `pip install timm`
    - PyAV: `conda install av -c conda-forge`
    - psutil: `pip install psutil`
    - scikit-learn: `pip install scikit-learn`
    - OpenCV: `pip install opencv-python`
    - tensorboard: `pip install tensorboard`

- Install other libraries
    ```
    pip install matplotlib
    ```

## JupyterLab setup on CARC

- Install pykernel
```
conda activate <your-env-name>
conda install -c conda-forge ipykernel
python -m ipykernel install --user --name <your-env-name> --display-name 'timesformer-pykernel'
```

- Login to CARC on-demand
    ```
    https://carc-ondemand.usc.edu
    ```

- Open JupyterLab Interactive App
    - To test setup use the default configuration
    - Once JupterLab server is up, connect to jupyter and open DeepSports/training/TimeSformer/train.ipynb
    - Change kernel to the display name i.e timesformer-pykernel
    - Execute first cell