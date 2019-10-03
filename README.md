# MitoSegNet Documentation

---

## Before running the MitoSegNet

If you have a CUDA-capable GPU (all Nvidia GPUs from the G8x series onwards, including GeForce, Quadro and Tesla), then you will need to download the GPU-enabled version of the MitoSegNet. You will furthermore have to download 
and install CUDA 10.0 and cuDNN 7.6.1. 

If you do not have a CUDA-capable GPU then you can alternatively download the CPU-enabled MitoSegNet, which will run without CUDA or cuDNN. Be aware that running the MitoSegNet on the CPU will be, depending on your CPU performance, 
slower than the GPU version and it is not recommended to use this version for training on large data as this will likely take very long. 

To get the best possible experience and results, we recommend to download the GPU-enabled MitoSegNet. 

### Installing CUDA 10.0 

https://developer.nvidia.com/cuda-10.0-download-archive

### Installing cuDNN 7.6.1 

To access the different versions of cuDNN click on the link below and create an account.  

https://developer.nvidia.com/rdp/form/cudnn-download-survey

### GPU usage

Be aware that the memory of your GPU limits the size of your images used for training as well as the batch size. In case training or prediction stop due to an out-of-memory error, consider reducing the size of your input images
or decrease the batch size. 

## Running the MitoSegNet

## Easy Mode

If you are unfamiliar with deep learning concepts such as batch size, learning rate or augmentation operations, then it is recommended to use the Easy Mode, in which most of the deep learning parameters are pre-set. 

### Predict on pretrained model

* Select directory in which 8-bit raw images are stored: Images have to be 8-bit, tif format and single plane. Use the macro MitoSegNet_PreProcessing.ijm for automated conversion of a large number of images (Prerequisite for macro usage is installation of Bio-Formats plugin on Fiji)
* Select pretrained_model_656.hdf5 (which can be found in installation folder)
* Enter minimum object size (in pixels) to exclude noise from the final segmentation 
* Depending if you have all images in one folder, or multiple set of images in sub-folders you can select to apply the model to one folder or multiple folders (Folder > Subfolder > Images)
* Select to predict on GPU or CPU (only on GPU version)
* Post-segmentation filtering: shows each generated segmentation automatically to allow User to choose which masks to save and which to discard

Once all entries are filled, click on "Start prediction" and wait until a Done window opens to notify of the successful completion.

If segmentation with a pretrained model did not generate good results you can try to finetune the pretrained model to your own data

### Finetune pretrained model

Select "New" if you are starting a new finetuning project or "Existing" if you want to continue to work on a previously generated finetuning project. 

* Specify name of the finetuning project folder
* Select directory in which 8-bit raw images are stored:
* Select directory in which hand-labelled (ground truth) images are stored 
* Select pretrained_model_656.hdf5
* Specify the number of augmentations (defines the number of times the original image will be duplicated and augmented. Start with a minimum of 10 augmentations per image. Increase until results no longer improve)
* Specify the number of epochs (defines the repetitions the training process will run through to learn how to segment the images based on the new input)
* Select to train on GPU or CPU (only on GPU version)

Once all entries are filled, click on "Start training" and wait until a Done window opens to notify of the successful completion. 
In the parent directory of the raw image and label image folder a Finetune_folder will be generated in which all the newly augmented data, image arrays and finetuned models will be stored.


## Advanced Mode

If you understand concepts such as data augmentation, weighted loss functions and learning rate then you might be interested in creating a more customized deep learning model
using the advanced mode. It is highly recommended to first familiarize yourself with the basic concepts of how convolutional neural networks work before attempting to use the
advanced mode. 

### Start new project

To start a new project, click on the "Start new project" button. 

* Choose a project name
* Select the directory in which you want the project folder to be generated in 
* Select the directory in which your 8-bit images are stored. Please make sure that no other folders or files than the images intended to be used for training your model are in the chosen directory. 
* Select the folder in which your ground truth (hand-segmented) images are stored. Make sure that the name of the  corresponding images in the 8-bit folder and the ground truth folder are the same

Once the project folder has been created, you can click on "Continue working on existing project".

Continue working on existing project

Start with generating the training data. 

### Create augmented data

* Select the recently created project directory 
* Based on the size of the images you are using the software will present you a list of possible tile sizes and tile numbers. When using the GPU, be aware that the maximum tile size possible will be limited by the GPU memory. If you run out of memory, try to select a smaller tile size
* Choose the number of augmentations per image you want to generate 
* Specify augmentation operations: visit https://keras.io/preprocessing/image/ to see what the different augmentation operations do
	* Horizontal flip
	* Vertical flip 
	* Width shift range
	* Height shift range
	* Shear range
	* Rotation range
	* Zoom range
	* Brightness range
* Create weight map: a weight map shows objects that are in close proximity to each other and is used to force the convolutional neural network to learn border separations. Be aware that when selecting "Create weight map" that the augmentation process will take longer and the training data will use more disk space 

### Train model

* Select the recently created project directory 
* Specify name of a new model or train on an existing model
* Specify number of epochs: how many times should the model train on the entire training data
* Specify learning rate: the learning rate controls how quickly or slowly a neural network model learns a problem
* Specify batch size: select the number of tiles that are fed into the network each iteration. The maximum batch size is limited by your GPU memory
* Use weight map: be aware that using a weight map will increase GPU memory usage during training
* Specify class balance weight factor: the class balance weight factor can correct for imbalanced classes, which is often the case for segmentented microscopy images (more background than object pixels). The MitoSegNet calculates the foreground to background pixel ratio and can be used to determine an appriopriate class balance weight factor

### Class balance weight factor calculation example 

Foreground to background pixel ratio: 1 to 19. This means for one object pixel there are 19 black pixels with no information. To get a foreground to background pixel ratio of 1 to 1,
we can set the weight factor to 1/19 which is roughly 0.05. That means that only 5% of the background pixels will be presented to the network during training. 

### Model prediction

* Select the project directory in which a trained model file has been generated 
* Select the folder in which previously unseen 8-bit images are located in to test model prediction 
	* If the folder contains only images files you may select "One folder" but in case it contains subfolders in which the images are located, then select "Multiple folders" to generate segmentations for all subfolders 
* Enter minimum object size (in pixels) to exclude noise from the final segmentation 
* Select to train on GPU or CPU (only on GPU version)
* Post-segmentation filtering: shows each generated segmentation automatically to allow User to choose which masks to save and which to discard

