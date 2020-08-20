# **End-to-end-trainable-network-for-superpixel-and-image-segmentation**



### Installation

#### 

#### Caffe Installation

1. Go to 'lib' folder if you are not already there:

```
cd lib/
```

1. We make use of layers in 'Video Propagation Networks' caffe repository and add additional layers for SSN superpixels:

```
git clone https://github.com/varunjampani/video_prop_networks.git
```

1. Manually copy all the source files  (files in `lib/include` and `lib/src` folders) to the corresponding locations in the `caffe` repository. In the `ssn_superpixels/lib` directory:

```
cp src/caffe/layers/* video_prop_networks/lib/caffe/src/caffe/layers/.
cp src/caffe/test/* video_prop_networks/lib/caffe/src/caffe/test/.
cp src/caffe/proto/caffe.proto video_prop_networks/lib/caffe/src/caffe/proto/caffe.proto
cp include/caffe/layers/* video_prop_networks/lib/caffe/include/caffe/layers/.
```

1. Install Caffe following the installation [instructions](http://caffe.berkeleyvision.org/installation.html). In the `ssn_superpixels/lib` directory:

```
cd video_prop_networks/lib/caffe/
mkdir build
cd build
cmake ..
make -j
cd ../../../..
```

Note: If you install Caffe in some other folder, update `CAFFEDIR` in `config.py` accordingly.

#### 

#### Install a cython file

We use a cython script taken from 'scikit-image' for enforcing connectivity in superpixels. To compile this:

```
cd lib/cython/
python setup.py install --user
cd ../..
```

### 

### Usage: BSDS segmentation

#### 

#### Data download

Download the BSDS dataset into `data` folder:

```
cd data
sh get_bsds.sh
cd ..
```

#### 

#### Superpixel computation

1. First download the trained segmentation models using the `get_models.sh` script in the `models` folder:

```
cd models
sh get_models.sh
cd ..
```

1. Use `compute_spixel.py` to compute superpixels on BSDS dataset:

```
python compute_spixels.py  --datatype TEST --n_spixels 100 --num_steps 10 --caffemodel ./models/intermediate_bsds_model --result_dir ./bsds_100/
```

You can change the number of superpixels by changing the `n_spixels` argument above, and you can update the `datatype` to `TRAIN` or `VAL` to compute superpixels on the corresponding data splits.

2. Use `compute_sgement.py` to compute superpixels on BSDS dataset:

```
python compute_sgement.py  --datatype TEST --n_spixels 100 --num_steps 10 --caffemodel ./models/intermediate_bsds_model --result_dir ./bsds_seg/
```