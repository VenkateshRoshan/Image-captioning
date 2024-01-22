
## Data Preparation
* Splitting dataset frame into Train and Test data frame.

### Preprocessing the Text 
* After performing basic cleaning steps , Splitting each description into words and and adding <stseq> at the start and <endseq> at the end of the description list.
* Gave index to each word.
* From each descriptions , creating new list as adding each word from the start of the description in to partial_desc and the next word to Y_train as OUTPUT of the model and continued till the end of the description.
* Converting this into the padded docs by using max length of description.
* ![image](https://github.com/VenkateshRoshan/Materials/assets/42509229/3d445e6a-f163-4854-a107-de4f7430ddc5)


## Without Using Pretrained Model :
* ### Image Loading :
  * ![image](https://github.com/VenkateshRoshan/Materials/assets/42509229/8038088e-13d7-41aa-b068-2c6337c40572)
  * Using Computer Vision module reading the image and resizing it into shape of (64,64) and adding it to the list.
* ### Model Training :
  * Using Convolutional Neural Networks to train Images and the final CNN layer, where we have reduced the image representation to a 256D sized vector.
  * Feeding it into NLP model which will generates the output textual represenation . Used BiDirectional LSTM for this NLP Module.
  * ![image](https://github.com/VenkateshRoshan/Materials/assets/42509229/71b67b2c-7f97-4ed2-aabf-d80ab09c7efb)
 
* Got the InAppropriate outputs , so for the updation I used pretrained models such as VGG16 , InceptionV3 , MobileNetV3.

## With Using MobileNet Pretrained Model :
* ### Image Loading :
  * Convert all images to size (224,224) and preprocessed the images from Keras MobileNet module.
  * Fitting the images into MobileNetV3 model.
* ### Model Training :
  * Using the output of the MobileNetV3 model and adding it into NLP model to train on the padded sequences.
  * ![image](https://github.com/VenkateshRoshan/Materials/assets/42509229/49cd3460-85ae-4458-beef-648d7054f890)

* Got some good expected outputs and bad outputs.
 
Results : 

* ![image](https://github.com/VenkateshRoshan/Materials/assets/42509229/bc31f287-7d2f-41b6-a43c-9fdea3b1ce5c)

* ![image](https://github.com/VenkateshRoshan/Materials/assets/42509229/cb1652a8-f157-4771-b16a-46425dca89c2)

* ![image](https://github.com/VenkateshRoshan/Materials/assets/42509229/bc5416da-f035-4a4a-9627-7ed772675a40)

# Graph :

![image](https://github.com/VenkateshRoshan/Materials/assets/42509229/f3d1d863-380e-4ca7-97e6-7a2de925bd6f)

## Potential shortcomings :
* Lack of quality data (The resolutions are not proper and varies for every image).
* Very hard to categorize the images based on the geometries and characters .
* Unable to capture textual information , descriprion is very diverse.
* Hyperparameter tuning.

## Improvements to my solution :
* Can Improve the Model accuracy using Data Augumentation.
* Need to work more on Image data preparation.
* Experiment with different hyperparameters can Improve the solution's accuracy.
* Can expect better results by including optimal character recognition (OCR) to capture textual information.

### * I can still work on it and provide the best solution , if I have time.


