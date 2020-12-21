**Installation guide**

Download whole project from here

[https://drive.google.com/drive/folders/1Zl\_gffeAMddcx2de8PZ9jDA5TGHK2mqM?usp=sharing](https://drive.google.com/drive/folders/1Zl_gffeAMddcx2de8PZ9jDA5TGHK2mqM?usp=sharing)

**Language** and environment: python-64 bit v 3.8.6

**Libraries** :

- nltk==3.5
- NumPy==1.18.5
- Keras==2.3.1
- TensorFlow==2.2
- flask==1.1.2

Also, if you are running code on a windows system then you might have to install **Graphviz-install-2.44.1-win64.exe** from the link

- [https://drive.google.com/file/d/1\_H8xuS9OxQdmZmLmZMdmG8M8gf\_Cv2Q6/view?usp=sharing](https://drive.google.com/file/d/1_H8xuS9OxQdmZmLmZMdmG8M8gf_Cv2Q6/view?usp=sharing)

NOTE: Keras and TensorFlow will require these specific versions. Latest versions will cause error.

Also, for if you do not want to do **preprocessing** of the image, **train model every time** , do not want to clean captions every time then you must have these files with the correct folder location

All these files can be downloaded from here

[https://drive.google.com/drive/folders/1klZz75mC5rS9ECjXsFG2ve87jB-dIum7?usp=sharing](https://drive.google.com/drive/folders/1klZz75mC5rS9ECjXsFG2ve87jB-dIum7?usp=sharing)

- ImageFeatures.pkl
- Descriptions.txt
- model\_1.h5 (save under models\_new folder)
- model\_10.h5 (save under models\_new folder)

**Dataset:**

You can download the dataset from the link below

[https://drive.google.com/drive/folders/1GkkL2L9WC6JrG-3KagPPy0W57iFl4QXK?usp=sharing](https://drive.google.com/drive/folders/1GkkL2L9WC6JrG-3KagPPy0W57iFl4QXK?usp=sharing)

**Predicting Image:**

There should be one image with the name example1.jpg as shown in folder structure.



**Run the code** :

Just run AutomaticImageCaptioning.py as a regular python code. No arguments are needed.

NOTE: if you are training the model it will crash the program after training task because the models will be saved in root directory so you must manually move the file and rerun the code that time it will pick the file form the folder. Other way is you can change the path of the model file in line 48 and line 49 in the AutomaticImageCaptioning.py file and same in ImageCaptioningWebsite.py file (line 38 ,39)

**Running the website code**

Run ImageCaptioningWebsite.py file which will run the server on the localhost at 5000 port. (Make sure you have a directory named uploads.) Once the website is running just upload any jpg/jpeg file and click upload.
