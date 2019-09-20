This method focuses on detecting image blur from an out-ofsample set of images consisting of in-focus non-blurred images and motion blurred images (basically seperating intentional blur from unintentional, actual blurred images). The novel process applies Laplacian operator
over a set of salient objects in an image, which extract a set of image features, and subsequently classifying the blur and nonblur images using Long Short-Term Memory model (LSTM).
The LSTM treats each salient contour varying in x and y axis of an image as a timestep to classify the images. Our method recognizes the imperfections in the saliency detection
algorithms, and incorporates the position, size, and relative blur value of each of the contours to define the image. 
The model is trained on a training size of 180 images, and evaluated on a test set containing 60 images,
and returns an accuracy of 80%.

To run the code, simply change the path set in the beginning of the code to train and test folder path.
The training set images are seperated in folders for blurred and non-blurred.
The test images are tagged in an Excel file.

--------------------------------------------------------------------------------------------------------------
Added pure Laplacian blur detection at threshold = 400 for benchmarking, returns 69% accuracy