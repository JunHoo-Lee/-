# FastMRI challenge 이뽈리트 조 ReadMe
## 2017-17322 이준후, 2017-12297 손진우

해당 코드는 TiTan X에서 돌아가는것을 확인하였습니다.

## SSIM score

Mean(0.9906, 0.9908, )

## Note
1. train.py 파일을 실행하면 path 정보를 가지고 있는 yaml 파일이 생겨나며, 기본값은 ppt대로 설정하였습니다.
2. log 파일은 runs 에 존재합니다.
3. reconstrucion h5 file은 runs/reconstructions 폴더에 있습니다.


## usage

1. pip install -r requirements.txt
2. python3 train.py
3. python3 evaulate.py
4. python3 leaderboard_eval.py


folder overview

├── Code
│   ├── evaluate.py
│   ├── leaderboard_eval.py
│   ├── requirements.txt
│   ├── ReadMe.txt
│   ├── train.py
│   └── utils
│       ├── __init__.py
│       ├── coil_combine.py
│       ├── common
│       │   └── loss_function.py
│       ├── data
│       │   ├── __init__.py
│       │   ├── mri_data.py
│       │   ├── subsample.py
│       │   ├── transforms.py
│       │   └── volume_sampler.py
│       ├── distance.py
│       ├── fftc.py
│       ├── losses.py
│       ├── math.py
│       ├── models
│       │   ├── __init__.py
│       │   ├── umy.py
│       │   ├── unet.py
│       │   ├── uorigin.py
│       │   ├── usqueeze.py
│       │   └── varnet.py
│       ├── pl_modules
│       │   ├── __init__.py
│       │   ├── data_module.py
│       │   ├── mri_module.py
│       │   ├── unet_module.py
│       │   └── varnet_module.py
│       └── tools.py
├── Data
│   ├── image_Leaderboard
│   ├── kspace_Leaderboard
│   ├── kspace_train
│   ├── kspace_val
│   ├── train
│   └── val
└── runs
    ├── checkpoints
    ├── lightning_logs
    └── reconstructions

