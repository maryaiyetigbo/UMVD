# Unsupervised Microscopy Video Denoising

To appear at **IEEE/CVF Computer Vision and Pattern Recognition Workshop(CVPRW), 2024**.

Authors: **Mary Aiyetigbo,  Alexander Korte, Ethan Anderson, Reda Chalhoub, Peter Kalivas, Feng Luo, Nianyi Li**.

Paper: [arXiv:2404.12163](https://www.arxiv.org/abs/2404.12163)

Website: [https://maryaiyetigbo.github.io/UMVD/](https://maryaiyetigbo.github.io/UMVD/)

<div>
<img src="https://maryaiyetigbo.github.io/UMVD/assets/standard.gif" height="150"/>
<img src="https://maryaiyetigbo.github.io/UMVD/assets/musc.gif" height="150"/>
<img src="https://maryaiyetigbo.github.io/UMVD/assets/GOWT1.gif" height="150"/>
</div>

## Pipeline

![Overview Figure](https://maryaiyetigbo.github.io/UMVD/assets/pipeline_fig.png)


### Environment

```
pip install -r requirements.txt
```

## Datasets
The following datasets were used in our paper. The download links for publicly available datasets are provided. 
1. `Two Photon Calcium Imaging` - Calcium imaging generated using Neural Anatomy and Optical Microscopy (NAOMi) simulation code. [https://bitbucket.org/adamshch/naomi_sim/src/master/](https://bitbucket.org/adamshch/naomi_sim/src/master/)
2. `MUSC` - One photon calcium imaging recordings were collected locally at the Medical University of South Carolina in freely behaving transgenic mice (Drd1-Cre and Drd2-Cre).
3. `Microscopy` - Fluorescence Microscopy dataset. [http://celltrackingchallenge.net/2d-datasets/](http://celltrackingchallenge.net/2d-datasets/)
4. `LIVE-YT-HFR` - The dataset used to train the color natural video data. [https://live.ece.utexas.edu/research/LIVE_YT_HFR/LIVE_YT_HFR/index.html](https://live.ece.utexas.edu/research/LIVE_YT_HFR/LIVE_YT_HFR/index.html)

## Training
To train the calcium imaging.
```shell
python train.py \
        --data-path dataset/calcium
        --dataset SingleVideo
        --n-frames 7
        --batch-size None
        --in-channels 1
        --out-channels 1
        --lr 1e-3
        --num-epochs 25
```

## Citation

```
@article{aiyetigbo2024unsupervised,
  title={Unsupervised Microscopy Video Denoising},
  author={Aiyetigbo, Mary and Korte, Alexander and Anderson, Ethan and Chalhoub, Reda and Kalivas, Peter and Luo, Feng and Li, Nianyi},
  journal={arXiv preprint arXiv:2404.12163},
  year={2024}
}
```

