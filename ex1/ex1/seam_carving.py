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
    

# def reshape_bilinear(image, new_shape):
#     """
#     Resizes an image to new shape using bilinear interpolation method
#     :param image: The original image
#     :param new_shape: a (height, width) tuple which is the new shape
#     :returns: the image resized to new_shape
#     """
#     new_height, new_width = new_shape
#     old_height, old_width, pixels = image.shape
#     height_ratio = old_height / new_height
#     width_ratio = old_width / new_width
#     new_image = np.zeros([new_height, new_width, pixels])
#     rows, columns, _ = image.shape
#     for row in range(new_height):
#         for column in range(new_width):
#             x = row * height_ratio
#             y = column * width_ratio
#             x_floor = math.floor(x)
#             y_floor = math.floor(y)
#             # x_ceil = min(math.ceil(x), new_shape[1]-1)
#             # y_ceil = min(math.ceil(y), new_shape[0]-1)
#             x_ceil = min( rows - 1, math.ceil(x))
#             y_ceil = min(columns- 1, math.ceil(y))
#             if (x_ceil == x_floor) and (y_ceil == y_floor):
#                 q = image[int(x), int(y), :]
#             elif (x_ceil == x_floor):
#                 q1 = image[int(x), int(y_floor), :]
#                 q2 = image[int(x), int(y_ceil), :]
#                 q = q1 * (y_ceil - y) + q2 * (y - y_floor)
#             elif (y_ceil == y_floor):
#                 q1 = image[int(x_floor), int(y), :]
#                 q2 = image[int(x_ceil), int(y), :]
#                 q = (q1 * (x_ceil - x)) + (q2	 * (x - x_floor))
#             else:
#                 v1 = image[x_floor, y_floor, :]
#                 v2 = image[x_ceil, y_floor, :]
#                 v3 = image[x_floor, y_ceil, :]
#                 v4 = image[x_ceil, y_ceil, :]

#                 q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
#                 q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
#                 q = q1 * (y_ceil - y) + q2 * (y - y_floor)

#             new_image[row,column] = q

#     return np.array(new_image, np.int32)

def reshape_bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    new_height, new_width = new_shape
    old_height, old_width, pixels = image.shape
    height_ratio = old_height / new_height
    width_ratio = old_width / new_width
    new_image = np.zeros([new_height, new_width, pixels])
    rows, columns, _ = image.shape
    for row in range(new_height):
        for column in range(new_width):
            x = row * height_ratio
            y = column * width_ratio
            new_image[row,column] = calculate_bilinear_pixel(image, x, y)

    return np.array(new_image, np.int32)


def calculate_bilinear_pixel(image, row, column):
    """
    calculate the bilinear pixel in image[row, column]
    """
    x_floor = math.floor(row)
    y_floor = math.floor(column)
    x_ceil = min( image.shape[0] - 1, math.ceil(row))
    y_ceil = min(image.shape[1]- 1, math.ceil(column))
    if (x_ceil == x_floor) and (y_ceil == y_floor):
        q = image[int(row), int(column)]
    elif (x_ceil == x_floor):
        q1 = image[int(row), int(y_floor)]
        q2 = image[int(row), int(y_ceil)]
        q = q1 * (y_ceil - column) + q2 * (column - y_floor)
    elif (y_ceil == y_floor):
        q1 = image[int(x_floor), int(column)]
        q2 = image[int(x_ceil), int(column)]
        q = (q1 * (x_ceil - row)) + (q2	 * (row - x_floor))
    else:
        v1 = image[x_floor, y_floor]
        v2 = image[x_ceil, y_floor]
        v3 = image[x_floor, y_ceil]
        v4 = image[x_ceil, y_ceil]

        q1 = v1 * (x_ceil - row) + v2 * (row - x_floor)
        q2 = v3 * (x_ceil - row) + v4 * (row - x_floor)
        q = q1 * (y_ceil - column) + q2 * (column - y_floor)

    return q
# def bl_resize(original_img, new_h, new_w):
# 	#get dimensions of original image
# 	old_h, old_w, c = original_img.shape
# 	#create an array of the desired shape. 
# 	#We will fill-in the values later.
# 	resized = np.zeros((new_h, new_w, c))
# 	#Calculate horizontal and vertical scaling factor
# 	w_scale_factor = (old_w ) / (new_w ) if new_h != 0 else 0
# 	h_scale_factor = (old_h ) / (new_h ) if new_w != 0 else 0
# 	for i in range(new_h):
# 		for j in range(new_w):
# 			#map the coordinates back to the original image
# 			x = i * h_scale_factor
# 			y = j * w_scale_factor
# 			#calculate the coordinate values for 4 surrounding pixels.
# 			x_floor = math.floor(x)
# 			x_ceil = min( old_h - 1, math.ceil(x))
# 			y_floor = math.floor(y)
# 			y_ceil = min(old_w - 1, math.ceil(y))

# 			if (x_ceil == x_floor) and (y_ceil == y_floor):
# 				q = original_img[int(x), int(y), :]
# 			elif (x_ceil == x_floor):
# 				q1 = original_img[int(x), int(y_floor), :]
# 				q2 = original_img[int(x), int(y_ceil), :]
# 				q = q1 * (y_ceil - y) + q2 * (y - y_floor)
# 			elif (y_ceil == y_floor):
# 				q1 = original_img[int(x_floor), int(y), :]
# 				q2 = original_img[int(x_ceil), int(y), :]
# 				q = (q1 * (x_ceil - x)) + (q2	 * (x - x_floor))
# 			else:
# 				v1 = original_img[x_floor, y_floor, :]
# 				v2 = original_img[x_ceil, y_floor, :]
# 				v3 = original_img[x_floor, y_ceil, :]
# 				v4 = original_img[x_ceil, y_ceil, :]

# 				q1 = v1 * (x_ceil - x) + v2 * (x - x_floor)
# 				q2 = v3 * (x_ceil - x) + v4 * (x - x_floor)
# 				q = q1 * (y_ceil - y) + q2 * (y - y_floor)

# 			resized[i,j,:] = q
# 	return resized.astype(np.uint8)
    
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