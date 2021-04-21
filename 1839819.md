# 新人培训PaddleX作业


## 下载PaddleX



```python
!pip install paddlex -i https://mirror.baidu.com/pypi/simple
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple
    Collecting paddlex
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/02/a4/90f3f1f997b65952d9517c74a3f672ba83721c124a481bc0cdc295e8adbc/paddlex-1.3.7-py3-none-any.whl (514kB)
    [K     |████████████████████████████████| 522kB 15.4MB/s eta 0:00:01
    [?25hRequirement already satisfied: flask-cors in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (3.0.8)
    Requirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.1.2)
    Requirement already satisfied: psutil in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.7.2)
    Collecting pycocotools; platform_system != "Windows" (from paddlex)
      Downloading https://mirror.baidu.com/pypi/packages/de/df/056875d697c45182ed6d2ae21f62015896fdb841906fe48e7268e791c467/pycocotools-2.0.2.tar.gz
    Requirement already satisfied: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.4.4)
    Requirement already satisfied: sklearn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.0)
    Collecting paddleslim==1.1.1 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/d1/77/e257227bed9a70ff0d35a4a3c4e70ac2d2362c803834c4c52018f7c4b762/paddleslim-1.1.1-py2.py3-none-any.whl (145kB)
    [K     |████████████████████████████████| 153kB 56.6MB/s eta 0:00:01
    [?25hRequirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.1.1.26)
    Requirement already satisfied: visualdl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (2.1.1)
    Requirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.36.1)
    Collecting xlwt (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/44/48/def306413b25c3d01753603b1a222a011b8621aed27cd7f89cbc27e6b0f4/xlwt-1.3.0-py2.py3-none-any.whl (99kB)
    [K     |████████████████████████████████| 102kB 74.3MB/s ta 0:00:01
    [?25hCollecting shapely>=1.7.0 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/98/f8/db4d3426a1aba9d5dfcc83ed5a3e2935d2b1deb73d350642931791a61c37/Shapely-1.7.1-cp37-cp37m-manylinux1_x86_64.whl (1.0MB)
    [K     |████████████████████████████████| 1.0MB 13.8MB/s eta 0:00:01
    [?25hRequirement already satisfied: Six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask-cors->paddlex) (1.15.0)
    Requirement already satisfied: Flask>=0.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask-cors->paddlex) (1.1.1)
    Requirement already satisfied: setuptools>=18.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex) (41.4.0)
    Requirement already satisfied: cython>=0.27.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex) (0.29)
    Requirement already satisfied: matplotlib>=2.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex) (2.2.3)
    Requirement already satisfied: scikit-learn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from sklearn->paddlex) (0.22.1)
    Requirement already satisfied: pyzmq in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddleslim==1.1.1->paddlex) (18.1.1)
    Requirement already satisfied: numpy>=1.14.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from opencv-python->paddlex) (1.16.4)
    Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.8.53)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (3.14.0)
    Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.7.1.1)
    Requirement already satisfied: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (7.1.2)
    Requirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (3.8.2)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.0.0)
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.21.0)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (2.22.0)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask>=0.9->flask-cors->paddlex) (1.1.0)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask>=0.9->flask-cors->paddlex) (0.16.0)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask>=0.9->flask-cors->paddlex) (7.0)
    Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask>=0.9->flask-cors->paddlex) (2.10.1)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools; platform_system != "Windows"->paddlex) (0.10.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools; platform_system != "Windows"->paddlex) (2.4.2)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools; platform_system != "Windows"->paddlex) (2.8.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools; platform_system != "Windows"->paddlex) (1.1.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib>=2.1.0->pycocotools; platform_system != "Windows"->paddlex) (2019.3)
    Requirement already satisfied: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (0.14.1)
    Requirement already satisfied: scipy>=0.17.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (1.3.0)
    Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (3.9.9)
    Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (0.18.0)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.23)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.6.1)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.6.0)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.2.0)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->paddlex) (2.8.0)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.4.10)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (0.10.0)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (2.0.1)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.4)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (16.7.9)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.0)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (3.0.4)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2.8)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (1.25.6)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2019.9.11)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->Flask>=0.9->flask-cors->paddlex) (1.1.1)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.6.0)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl>=2.0.0->paddlex) (7.2.0)
    Building wheels for collected packages: pycocotools
      Building wheel for pycocotools (setup.py) ... [?25ldone
    [?25h  Created wheel for pycocotools: filename=pycocotools-2.0.2-cp37-cp37m-linux_x86_64.whl size=278368 sha256=2cc548127882add2b8196fa0dc3758ce9b1da0d3133d85f53944ff6d007c5ba5
      Stored in directory: /home/aistudio/.cache/pip/wheels/fb/44/67/8baa69040569b1edbd7776ec6f82c387663e724908aaa60963
    Successfully built pycocotools
    Installing collected packages: pycocotools, paddleslim, xlwt, shapely, paddlex
    Successfully installed paddleslim-1.1.1 paddlex-1.3.7 pycocotools-2.0.2 shapely-1.7.1 xlwt-1.3.0


## 准备泪河数据集



```python
!unzip JPGimages.zip
```


```python
# 数据格式转换
!paddlex --data_conversion --source labelme --to SEG \
        --pics JPGimages \
        --annotations annotations \
        --save_dir dataset_seg
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    [0m


```python
# 数据集划分
!paddlex --split_dataset --format SEG --dataset_dir dataset_seg --val_value 0.2 --test_value 0.1
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    Dataset Split Done.[0m
    [0mTrain samples: 293[0m
    [0mEval samples: 83[0m
    [0mTest samples: 41[0m
    [0mSplit files saved in dataset_seg[0m
    [0m[0m


```python
# 引入seg的transforms进行图像增强预处理
from paddlex.seg import transforms
train_transforms = transforms.Compose([
    transforms.RandomVerticalFlip(prob=0.1),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], min_val=[0, 0, 0], max_val=[255.0, 255.0, 255.0]),
])

eval_transforms = transforms.Compose([
    transforms.RandomVerticalFlip(prob=0.1),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], min_val=[0, 0, 0], max_val=[255.0, 255.0, 255.0]),
])
```


```python
# 数据集读取 
import paddlex
train=paddlex.datasets.SegDataset("dataset_seg", "dataset_seg/train_list.txt", "dataset_seg/labels.txt", transforms=train_transforms, num_workers='auto', buffer_size=100, parallel_method='process', shuffle=False)
val=paddlex.datasets.SegDataset("dataset_seg", "dataset_seg/val_list.txt", "dataset_seg/labels.txt", transforms=eval_transforms, num_workers='auto', buffer_size=100, parallel_method='process', shuffle=False)
test=paddlex.datasets.SegDataset("dataset_seg", "dataset_seg/test_list.txt", "dataset_seg/labels.txt", transforms=eval_transforms, num_workers='auto', buffer_size=100, parallel_method='process', shuffle=False)
```

    2021-04-21 15:44:30 [INFO]	293 samples in file dataset_seg/train_list.txt
    2021-04-21 15:44:30 [INFO]	83 samples in file dataset_seg/val_list.txt
    2021-04-21 15:44:30 [INFO]	41 samples in file dataset_seg/test_list.txt



```python
num_classes = len(train.labels)

# 训练模型 由于时间过长，只训练了7个epoch
model = paddlex.seg.UNet(num_classes=num_classes)
model.train(
    num_epochs=20,
    train_dataset=train,
    train_batch_size=4,
    eval_dataset=val,
    learning_rate=0.01,
    save_dir='output/unet',
    use_vdl=True)
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/math_op_patch.py:298: UserWarning: /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddlex/cv/nets/segmentation/model_utils/loss.py:74
    The behavior of expression A * B has been unified with elementwise_mul(X, Y, axis=-1) from Paddle 2.0. If your code works well in the older versions but crashes in this version, try to use elementwise_mul(X, Y, axis=0) instead of A * B. This transitional warning will be dropped in the future.
      op_type, op_type, EXPRESSION_MAP[method_name]))
    2021-04-21 15:49:56,895 - INFO - If regularizer of a Parameter has been set by 'fluid.ParamAttr' or 'fluid.WeightNormParamAttr' already. The Regularization[L2Decay, regularization_coeff=0.000040] in Optimizer will not take effect, and it will only be applied to other Parameters!


    2021-04-21 15:49:57 [INFO]	Downloading unet_coco_v3.tgz from https://paddleseg.bj.bcebos.com/models/unet_coco_v3.tgz


    100%|██████████| 97117/97117 [00:01<00:00, 58555.36KB/s]


    2021-04-21 15:49:59 [INFO]	Decompressing output/unet/pretrain/unet_coco_v3.tgz...
    2021-04-21 15:50:04 [INFO]	Load pretrain weights from output/unet/pretrain/unet_coco_v3.
    2021-04-21 15:50:04 [WARNING]	[SKIP] Shape of pretrained weight output/unet/pretrain/unet_coco_v3/logit/weights doesn't match.(Pretrained: (21, 64, 3, 3), Actual: (2, 64, 3, 3))
    2021-04-21 15:50:05 [INFO]	There are 90 varaibles in output/unet/pretrain/unet_coco_v3 are loaded.



```python
# 模型预测

import paddlex as pdx
model = pdx.load_model('output/unet/best_model')
image_name = 'dataset_seg/JPEGImages/lhgd389.jpg'
result = model.predict(image_name)
pdx.seg.visualize(image_name, result, weight=0.4, save_dir='./output/unet')
```

    2021-04-21 19:12:04 [INFO]	Model[UNet] loaded.
    2021-04-21 19:12:04 [INFO]	The visualized result is saved as ./output/unet/visualize_lhgd389.jpg


### 预测结果如下

![](https://ai-studio-static-online.cdn.bcebos.com/0bc27e8ec38e49dd9bb8b5996eccfafd3208e6b8c13e412ea7f1aedb992babc4)

