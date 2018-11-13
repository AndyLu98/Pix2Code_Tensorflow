# Pix2Code_Tensorflow

A tensorflow and modified pix2code project that tries to automatically generate HTML code from design sketches. 

In the original project by Tony Beltramelli, the Keras framework was used. I converted all the Keras code into TensorFlow code since some frameowork/cloud computing services do not support Keras yet. 

Compared to the original project, this version pre-extracts an image's features using pre-trained VGG network. I found that by doing so a better accuracy can be achieved. 
