# Semantic Segmentation for Medical Images
### Implemenation of Unets for Lung Segmentation in Xrays in Keras
![inp-out](https://raw.githubusercontent.com/theidentity/Unets/master/documentation/sample.png)
---
### Key Details
* Implementation of a Unet with Keras
* Based on the work [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

|Item| Details|
|---|---|
|**Input**|256 x 256 grayscale Xray Image|
|**Output**| 256 x 256 segmentation map|
|Train Images|110|
|Manual train masks|110|
|Validation Images|28|
|Manual validation masks|28|


* Thanks to [zhixuhao](https://github.com/zhixuhao/unet) for the keras implementation of unets
* Have improved upon that to run with image generators in keras dynamically and augment while training
---
### Dependencies
* Keras 2.1.5
* Numpy 1.14.2
* OpenCV 2.4.9.1 
	* Just using it to write and resize images
	* You may replace with PIL if you prefer
---
### Things to note
* While running ensure that the xrays and images are in separate folders and have the same labels
* Follow similar folder hierarchy in **data/** to your work easier ;)
---
### Running Unets
```python
# Initialize the Unet
u1 = Unet()

# Round one of training
u1.train(lr=1e-4,num_epochs=20)

# Improve upon existing model
u1.continue_training(lr=1e-4,num_epochs=20)

# Visualize image and output side by side
u1.generate_output(save=True,mode='side_by_side',output_folder='data/outputs/side_by_side/')

# Crop images based on output mask and return the mask
u1.generate_output(save=True,mode='cropped',output_folder='data/outputs/cropped/')

# Get just the masks
u1.generate_output(save=True,mode='mask_only',output_folder='data/outputs/masks_only/')

```
