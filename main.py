import cv2
import numpy as np
from numpy import concatenate

def colorize_image(input_image_path):
    # Load the pre-trained model
    prototxt_url = 'https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt'
    caffemodel_url = 'https://github.com/richzhang/colorization/releases/download/v2/colorization_release_v2.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt_url, caffemodel_url)

    # Load the black and white image
    bw_img = cv2.imread(input_image_path)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(bw_img, cv2.COLOR_BGR2GRAY)

    # Resize the grayscale image to 224x224 (the input size for the model)
    gray_img_resized = cv2.resize(gray_img, (224, 224))

    # Convert the resized grayscale image to LAB color space
    lab_img = cv2.cvtColor(gray_img_resized, cv2.COLOR_GRAY2LAB)

    # Split the LAB image into L and AB channels
    L_channel, AB_channels = cv2.split(lab_img)

    # Normalize the L channel
    L_channel = L_channel / 255.0

    # Reshape the L channel to match the input shape of the model
    L_channel = L_channel.reshape((1, 1, 224, 224))

    # Normalize the AB channels
    AB_channels = AB_channels - 128.0
    AB_channels = AB_channels / 128.0

    # Reshape the AB channels to match the input shape of the model
    AB_channels = AB_channels.transpose((2, 0, 1))
    AB_channels = AB_channels.reshape((1, 2, 224, 224))

    # Concatenate the L and AB channels to create the input blob for the model
    input_blob = concatenate((L_channel, AB_channels), axis=1)

    # Pass the input blob through the model to get the predicted colorized image
    net.setInput(input_blob)
    output = net.forward()
    colorized_img = output.transpose((0, 2, 3, 1))
    colorized_img = cv2.resize(colorized_img[0], (bw_img.shape[1], bw_img.shape[0]))

    return colorized_img

if __name__ == "__main__":
    # Show the introductory menu
    print("Hello, welcome to COLORIZER!")

    # Ask the user for input images
    num_images = int(input("How many images do you want to colorize? "))
    input_images = []
    for i in range(num_images):
        input_path = input(f"Enter the path to image {i + 1}: ")
        input_images.append(input_path)

    for i, input_image_path in enumerate(input_images):
        try:
            colorized_img = colorize_image(input_image_path)
            cv2.imshow(f"Colorized Image {i + 1}", colorized_img)
            cv2.waitKey(0)
        except Exception as e:
            print(f"Error processing image {i + 1}: {e}")

    cv2.destroyAllWindows()
