from keras.models import load_model
from keras.src.utils import load_img, img_to_array
from numpy import argmax

# loading pre trained model
model = load_model('digit_classifier.h5')


def load_image(filename):
    # load the image
    img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
    img.save("out2.png")
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


def predict_digit(filename):
    img = load_image(filename)
    predict_value = argmax(model.predict(img))
    return predict_value


def get_image_prediction(path):
    prediction = predict_digit(path)
    return prediction
