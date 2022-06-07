# computer_vision

## Description

This convolutional neural network generates a model that predicts a person's age given a photo with a Mean Absolute Error (MAE)=5.973 (below the upper limit of 8.0). The model uses ResNet50 as the backbone, ReLU for activation, epochs=24, Adam lr=0.0001, and a batch size of 32. 

__Project Overview__
- The Good Seed supermarket chain seeks to evaluate if machine learning can help them identify underage customers. 
- They want to make sure they do not sell alcohol to people below the legal drinking age.
- The store has cameras in the checkout area which are triggered when a person is buying alcohol.
- They want a program that *determines a person's age from a photo*.

__Data Description__
- Data is available from [ChaLearn Looking at People](https://gesture.chalearn.org).
- The set of photographs of people have a corresponding label with their ages indicated.
- A csv file (`labels.csv`) with two columns: file_name and real_age.

__Project Goal__
- Create a regression model that predicts a person's age based on an image.
- Build a generator with the ImageDataGenerator generator to process images in batches (to reduce computational resources).
- The Mean Absolute Error(MAE) must be less than 8.0.

## Dependencies
This project requires Python and the following Python libraries installed:

    Pandas
    matplotlib
    seaborn
    tensorflow
    keras
    
Additionally, access to a GPU platform is ideal.

Authors

Renee Raven

License

This project is licensed under the MIT License - see the LICENSE file for details

