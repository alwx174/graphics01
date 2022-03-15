import numpy as np
import math
def get_greyscale_image(image, colour_wts):
    """
    Gets an image and weights of each colour and returns the image in greyscale
    :param image: The original image
    :param colour_wts: the weights of each colour in rgb (ints > 0)
    :returns: the image in greyscale
    """
    
    greyscale_image = np.matmul(image, colour_wts)
    return greyscale_image
    
def reshape_bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    return new_image
    
def gradient_magnitude(image, colour_wts):
    """
    Calculates the gradient image of a given image
    :param image: The original image
    :param colour_wts: the weights of each colour in rgb (> 0) 
    :returns: The gradient image
    """
    greyscale = get_greyscale_image(image, colour_wts)
    # greyscale = np.array([[1, 2, 6], [3, 4, 5], [5,6,7]])
    # test = np.square(np.gradient(np.array([[1, 2, 6], [3, 4, 5], [5,6,7]], dtype=float), axis=1)) + np.square(np.gradient(np.array([[1, 2, 6], [3, 4, 5], [5,6,7]], dtype=float), axis=0))
    # test = np.sqrt(test)

    # moving the rows to prepare I(x+1,y) and I(x, y+1)
    x_forward = np.roll(greyscale, -1 ,axis=0)
    y_forward = np.roll(greyscale, -1 ,axis=1)
    # perform the subtraction I(x+1,y)- I(x,y) and I(x, y+1) - I(x,y)
    top_x = (y_forward - greyscale)
    top_y = (x_forward - greyscale)

    # top_x = ((y_forward - greyscale) + 255) * 1/2
    # top_y = ((x_forward - greyscale) + 255) * 1/2
    
    # perform addition,square and root of gradient to calculate gradient magnitude
    top_x, top_y = np.square(top_x), np.square(top_y)
    final_res = top_x + top_y
    final_res = np.sqrt(final_res)
    # final_res = (final_res + 150) * (1/2)
    gradient = final_res

    # hieght, width = greyscale.shape
    # tt = np.zeros(greyscale.shape)
    # for i in range(hieght-1):
    #     for j in range(width-1):
    #         tt[i,j] = np.square(greyscale[i+1, j] - greyscale[i, j])
    #         tt[i,j] += np.square(greyscale[i, j+1] - greyscale[i, j])
    #         tt[i,j] = np.sqrt(tt[i, j])

    # gradient = np.square(np.gradient(greyscale, axis=1)) + np.square(np.gradient(greyscale, axis=0))
    # gradient = np.sqrt(gradient)
    # print(2)
    # index_mat_x = np.ones(greyscale.shape[0])
    # for i in range(greyscale.shape[0]):
    #     if i == 0 or i == 1:
    #         continue
    #     index_mat_x = np.column_stack((index_mat_x, np.ones(greyscale.shape[0])*i))
    # np.roll(np.power(test.A, -1*np.ones(test.A.shape)), 1,axis=0)
    ###Your code here###
    ###**************###
    
    return gradient
    
def visualise_seams(image, new_shape, carving_scheme, colour):
    """
    Visualises the seams that would be removed when reshaping an image to new image (see example in notebook)
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :param carving_scheme: the carving scheme to be used.
    :param colour: the colour of the seams (an array of size 3)
    :returns: an image where the removed seams have been coloured.
    """
    ###Your code here###
    ###**************###
    return seam_image
    
def reshape_seam_crarving(image, new_shape, carving_scheme):
    """
    Resizes an image to new shape using seam carving
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :param carving_scheme: the carving scheme to be used.
    :returns: the image resized to new_shape
    """
    ###Your code here###
    ###**************###
    return new_image