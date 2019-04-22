Face-Specific Data Augmentation for Unconstrained Face Recognition
===========

This page contains the face augmentation layer and pre-trained model along with training and validation files from the paper _I. Masi\*, A. Tran\*, T. Hassner\*, G. Sahin, G. Medioni, "[Face-Specific Data Augmentation for Unconstrained Face Recognition](https://link.springer.com/article/10.1007/s11263-019-01178-0) ", in the International Journal of Computer Vision 2019 [1]_.

![Teaser](https://i.imgur.com/9gjNCKN.png)

## Features
* ResNet-101 model trained with face-specific augmentation
* Training script and and face-specific augmentation layer

## Dependencies

* [Caffe](http://caffe.berkeleyvision.org/) for training
    - Possibly using [A-Softmax](https://github.com/wy1iu/sphereface)
* [OpenCV Python Wrapper](http://opencv.org/)
* [Matplotlib](http://matplotlib.org/)
* [Numpy](http://www.numpy.org/)
* [Python2.7](https://www.python.org/download/releases/2.7/)
* [LMDB](https://lmdb.readthedocs.io/en/release/)

The code has been tested on Linux only. On Linux you can rely on the default version of python, installing all the packages needed from the package manager or on Anaconda Python and install required packages through `conda`. 


## Face Recognition Model Usage

To download the CNN Face recognition model along wit the training and validation files, please fill [this form](https://docs.google.com/forms/d/1RvmVpTEBQFMIvigVKT6FHQQVs3aUpBxVGTfL4XRy6hc/). The model size and other files are about 1GB.

The CNN model can be used off-the-shelf to extract face descriptors.

To get a face descriptor of an image, images should be aligned as done in training for best performance. A tutorial for feature extractor for Caffé is available [here](https://medium.com/@accssharma/image-feature-extraction-using-pretrained-models-in-caffe-491d0c0b818b).

The activation to be extracted as a face descriptor is `pool5`.

```python
layer {                                                                                                                                                
        bottom: "res5c"                                                                                                                                
        top: "pool5"                                                                                                                                   
        name: "pool5"                                                                                                                                  
        type: "Pooling"                                                                                                                                
        pooling_param {                                                                                                                                
                kernel_size: 7                                                                                                                         
                stride: 1                                                                                                                              
                pool: AVE                                                                                                                              
        }                                                                                                                                              
}                 
```

## Layer Usage _(Coming soon...)_

The augmentation layers empoyes an _hybrid_ approach to on-the-fly augmentation. For each training image, we precompute the actual 3D transformations and store estimated poses and landmarks offline along with other necessary information. During training, rendering training faces to multiple novel views using multiple generic 3D faces is performed on-the-fly so that we can spare the space of storing all augmented RGB images.

In order to use the face rendering layer, the data needs to preprocessed with:
- A face detector; we used an improved version of this [Face Detection ](https://sites.google.com/site/irisprojectjanus/products-services); a baseline version is avaiable [here.](https://sites.google.com/site/irisprojectjanus/products-services)
- [FacePoseNet (FPN)](https://github.com/fengju514/Face-Pose-Net) to recompute 6D pose estimates and other metadata.

[FacePoseNet (FPN)](https://github.com/fengju514/Face-Pose-Net) is useful to robustly compute the 3D rotation and 3D translation of given an image. FPN is available [here](https://github.com/fengju514/Face-Pose-Net) but all these preprocessing steps are not bundeled togethe with this release.

The final output of this preprocessing should yield a set of Lightning Memory-Mapped Database (LMDB) that store all the metadata needed as input to the layer as follows:
 
 ```bash
[1199]$ ls -lt preproc/
drwxr-xr-x     2 iacopo glaive      4096 Sep 17 09:57 output_ldmks_combined.lmdb
drwxr-xr-x     2 iacopo glaive      4096 Sep 17 09:56 output_offset_combined.lmdb
drwxr-xr-x     2 iacopo glaive      4096 Sep 17 09:56 output_pose_combined.lmdb
drwxr-xr-x     2 iacopo glaive      4096 Sep 16 19:49 output_best_bbox_combined.lmdb
drwxr-xr-x     2 iacopo glaive      4096 Sep 16 19:49 output_yaw_combined.lmdb
```
where: 
 - `output_ldmks_combined.lmdb` stores the landmarks (used only for 2D alignment)
 - `output_pose_combined.lmdb` stores the 3D pose
 - `output_offset_combined.lmdb` stores the information for adjusting the recomputed pose to the bounding box.
 - `output_best_bbox_combined.lmdb` stores the face detection bounding box
 - `output_yaw_combined.lmdb` stores the yaw angle
 
 **Note that no image is stored at this point. All the rendered images will be generated online.**
 
_An example of pre-generated LMDBs files on bunch of samples a is available  in order to chek the format is availalble_

The layer is available in the folder [data_aug_layer](data_layer).

### Other Usages

#### Visualization of the preprocessing.
We also provide a jupyter notebook to visualize the preprocessing metadata such as bounding box, 3D pose, thus landmarks, and finally yaw values. The notebook is available here  [[jupyter notebook viz]](notebooks/check_data_cow.ipynb)

#### Use of the layer out of the training
If one wants can also use the layer outside from training. This jupyter notebook shows how to use the layer to continously fetch images from the layer queue and visualize them: [[jupyter notebook viz]](notebooks/#)

### Parameters

#### Input Data Format

The training and validation files follows this syntax: 
```
<image_key> <relative_path_to_image> <identity_label>
```
for instance, something similar to the following. Note that `!!root_dir!!` will be replaced online and indicates the absolute path where the data actually lives:

```
XXXm.014ww_XXX_MS000001 !!root_dir!!/XXXm.014ww_XXX/XXXm.014ww_XXX_MS000001.jpg 0
XXXm.014ww_XXX_MS000007 !!root_dir!!/XXXm.014ww_XXX/XXXm.014ww_XXX_MS000007.jpg 0
XXXm.014ww_XXX_MS000011 !!root_dir!!/XXXm.014ww_XXX/XXXm.014ww_XXX_MS000011.jpg 0
XXXm.0bxgx4XXX_MS000034 !!root_dir!!/XXXm.0bxgx4XXX/XXXm.0bxgx4XXX_MS000034.jpg 1
....
```
For better performance it is better if `!!root_dir!!` lives on a SSD disk.

The key stores the image filename that needs to be in the format `<subject_id>_<image_id>`.
The system can work also on other datasets but it has to follow this format.

### Training Parameters
The following parameters can be tuned in the face-specific augmentation layer:

```python
## Training Params
train_params = \                                                                                           
{
'batch_size' : '16',
'effective_batch' : '96', 
'num_core' : '8',
'perc_real' : '0.5',
'qsize' : '64',
'yaw_frontal' : '30.0',
'data_root' : data_set['data_root'],
'nGPU' : 1,
} 
```

where important params are the following:
- `batch_size` indicates the size of the mini-batch used
-  `effective_batch`, the effective batch size used
- `num_core` the number of core used to multiprocess the augmentation
- `perc_real` [0..1] indicates the percentage of real images to store in the batch (`perc_real=1.0` means only real images)
- `yaw_frontal` indicates the absolute yaw value to consider an image as profile.

This params will be then used by the face-specifc augmentaiton layer in the network definition as:

```
$ head -n 13 network/ResNet-101-aug-layer.prototxt
name: "ResNet-101-aug-layer"
layer {
  name: "data"
  type: "Python"
  top: "data"
  top: "label"
  python_param {
    module: "face_aug_datalayers_prefetch_mp_queue_caffe"
    layer: "FaceAugDataLayer"
    param_str: "{'im_shape': [224, 224], 'split': 'train', 'batch_size': !!batch_size!!, 'source': '!!train_list!!', 'mean_file': '!!mean_file!!', 'num_core' : !!num_core!!, 'perc_real' : !!perc_real!!, 'qsize' : !!qsize!!, 'yaw_frontal' : !!yaw_frontal!!, 'data_root' : !!data_root!! }"
  }
}
...
```


## Citation

Please cite our paper with the following bibtex if you use our face recognition model or our layer:

``` latex
@article{masi2019facespecific,
    author="Masi, Iacopo and Trần, Anh Tuấn and Hassner, Tal and Sahin, Gozde and Medioni, G{\'e}rard",
    title="Face-Specific Data Augmentation for Unconstrained Face Recognition",
    journal="International Journal of Computer Vision",
    year="2019",
    month="Apr",
    day="01",
    issn="1573-1405",
    doi="10.1007/s11263-019-01178-0",
    url="https://doi.org/10.1007/s11263-019-01178-0"
}


```

## References

[1] I. Masi\*, A. Tran\*, T. Hassner\*, G. Sahin, G. Medioni, "Face-Specific Data Augmentation for Unconstrained Face Recognition", IJCV 2019, 

[2] I. Masi\*, A. Tran\*, T. Hassner\*, J. Leksut, G. Medioni, "Do We Really Need to Collect Million of Faces for Effective Face Recognition? ", ECCV 2016, 

[3] I. Masi, S. Rawls, G. Medioni, P. Natarajan "Pose-Aware Face Recognition in the Wild", CVPR 2016

[4] T. Hassner, S. Harel, E. Paz and R. Enbar "Effective Face Frontalization in Unconstrained Images", CVPR 2015

[5] F. Chan, A. Tran, T. Hassner, I. Masi, R. Nevatia, G. Medioni, "FacePoseNet: Making a Case for Landmark-Free Face Alignment," ICCV Workshops, 2017

\* denotes equal authorship

## Changelog
- April 2019, First  Release 

## Disclaimer

_The SOFTWARE PACKAGE provided in this page is provided "as is", without any guarantee made as to its suitability or fitness for any particular use. It may contain bugs, so use of this tool is at your own risk. We take no responsibility for any damage of any sort that may unintentionally be caused through its use._

## Contacts

If you have any questions, drop an email to _iacopo@isi.edu_ and _talhassner@gmail.com_ or leave a message below with GitHub (log-in is needed).
