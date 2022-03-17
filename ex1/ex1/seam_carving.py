from decimal import Clamped
import numpy as np
import math

from sympy import Integer
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

    # gradient = np.gradient(greyscale, axis=1) + np.gradient(greyscale, axis=0)
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
    
def visualise_seams(image, new_shape, show_horizontal, colour):
    """
    Visualises the seams that would be removed when reshaping an image to new image (see example in notebook)
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :param show_horizontal: the carving scheme to be used.
    :param colour: the colour of the seams (an array of size 3)
    :returns: an image where the removed seams have been coloured.
    """
    ###Your code here###
    ###**************###
    # gray scale scalars
    greyscale_wt = [0.299, 0.587, 0.114]

    # recieve gradient matrix
    gradient_matrix = gradient_magnitude(image, greyscale_wt)
    gradient_matrix2 = get_greyscale_image(image, greyscale_wt)
    if show_horizontal:
        gradient_matrix = gradient_matrix.T
    
    num_of_rows = gradient_matrix.shape[0]
    num_of_cols = gradient_matrix.shape[1]

    # calculate cheapest paths matrix using dynamic programming
    for row in range(1, num_of_rows):
    
        for col in range(num_of_cols ):
            C_u = 0
            # when pixel is in the leftmost col
            if col == 0:
                # C_r = abs(gradient_matrix[row,col-1] - gradient_matrix[row,col+1]) + abs(
                #     gradient_matrix[row-1,col] - gradient_matrix[row,col+1]
                # )
                # C_r = abs(gradient_matrix2[row-1,col] - gradient_matrix2[row,col+1])
                C_r = abs(gradient_matrix2[row-1,col] - gradient_matrix2[row,col+1])
                # C_r = 0
                val = (gradient_matrix[row-1,col+1] + C_r,)
            # when pixel is in the rightmost col
            elif col == (num_of_cols - 1):
                # C_l = abs(gradient_matrix[row,col-1] - gradient_matrix[row,col+1]) + abs(
                #     gradient_matrix[row-1,col] - gradient_matrix[row,col-1]
                # )
                # C_l = abs(gradient_matrix2[row-1,col] - gradient_matrix2[row,col-1])
                C_l = abs(gradient_matrix2[row-1,col] - gradient_matrix2[row,col-1])
                # C_l = 0
                val = (gradient_matrix[row-1,col-1] + C_l,)
            # when pixel is in the middle
            else:
                # calculate forward cost
                # C_u = abs(gradient_matrix2[row,col-1] - gradient_matrix2[row,col+1])
                # C_l = abs(gradient_matrix2[row,col-1] - gradient_matrix2[row,col+1]) + abs(
                #     gradient_matrix2[row-1,col] - gradient_matrix2[row,col-1]
                # )
                # C_r = abs(gradient_matrix2[row,col-1] - gradient_matrix2[row,col+1]) + abs(
                #     gradient_matrix2[row-1,col] - gradient_matrix2[row,col+1]
                # )
                C_u = abs(gradient_matrix2[row,col-1] - gradient_matrix2[row,col+1])
                C_l = abs(gradient_matrix2[row,col-1] - gradient_matrix2[row,col+1]) + abs(
                    gradient_matrix2[row-1,col] - gradient_matrix2[row,col-1]
                )
                C_r = abs(gradient_matrix2[row,col-1] - gradient_matrix2[row,col+1]) + abs(
                    gradient_matrix2[row-1,col] - gradient_matrix2[row,col+1]
                )
                # C_u, C_r, C_l = 0,0,0
                val = (gradient_matrix[row-1,col+1] + C_r, gradient_matrix[row-1,col-1] + C_l)
            

            min_val = min(gradient_matrix[row-1,col] + C_u, *val)
            gradient_matrix[row,col] += min_val
    
    # np.where(gradient_matrix == np.amin(gradient_matrix, axis=1)[num_of_rows - 1])

    def pick_cheaper_seam(image, gradient_matrix, num_of_seams):
        new_image = np.copy(image)
        last_row = gradient_matrix[gradient_matrix.shape[0]-1]
        # sorted_row = np.sort(last_row)
        # cheaper_values = sorted_row[:num_of_seams]
        # indexes = np.where(last_row == cheaper_values)
        indexes = np.argpartition(last_row, num_of_seams)[:num_of_seams]
        current_row = gradient_matrix.shape[0] - 1
        # color last row
        for i in range(num_of_seams):
            new_image[current_row][int(indexes[i])] = np.array(colour)

        for col in indexes:
            current_row = new_image.shape[0] - 1
            
            while (current_row > 0):
                min_choice_col = 0
                if col == 0:
                    min_choice_col = col if gradient_matrix[current_row-1, col] <= gradient_matrix[current_row-1, col+1] else col + 1
                elif col == num_of_cols - 1:
                    min_choice_col = col if gradient_matrix[current_row-1, col] <= gradient_matrix[current_row-1, col-1] else col - 1
                else:
                    left, upper, right = gradient_matrix[current_row-1, col-1], gradient_matrix[current_row-1, col], gradient_matrix[current_row-1, col+1]
                    if (upper <= left) and (upper <= right):
                        column = col
                    elif left <= right:
                        column = col-1
                    else:
                        column = col+1
                    min_choice_col = column
                gradient_matrix[current_row, col] = np.Infinity
                current_row -= 1
                new_image[current_row, min_choice_col] = np.array(colour)
                col = min_choice_col
                
        # while(current_row > 0):
        #     for i in range(num_of_seams):
                
        #         new_image[current_row][indexes[i]] = colour

        return new_image
    seam_image = pick_cheaper_seam(image, gradient_matrix, 150)
    print(2)    
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

    # def visualise_seams(image, new_shape, show_horizontal, colour):
    # """
    # Visualises the seams that would be removed when reshaping an image to new image (see example in notebook)
    # :param image: The original image
    # :param new_shape: a (height, width) tuple which is the new shape
    # :param show_horizontal: the carving scheme to be used.
    # :param colour: the colour of the seams (an array of size 3)
    # :returns: an image where the removed seams have been coloured.
    # """
    # ###Your code here###
    # ###**************###
    # # gray scale scalars
    # greyscale_wt = [0.299, 0.587, 0.114]

    # # recieve gradient matrix
    # gradient_matrix = gradient_magnitude(image, greyscale_wt)

    # if show_horizontal:
    #     gradient_matrix = gradient_matrix.T
    
    # num_of_rows = gradient_matrix.shape[0]
    # num_of_cols = gradient_matrix.shape[1]

    # # calculate cheapest paths matrix using dynamic programming
    # for col in range(num_of_cols - 1):
    #     for row in range(1, num_of_rows - 1):
    #         C_u = 0
    #         # when pixel is in the leftmost col
    #         if col == 0:
    #             C_l = abs(gradient_matrix[row,col-1] - gradient_matrix[row,col+1]) + abs(
    #                 gradient_matrix[row-1,col] - gradient_matrix[row,col-1]
    #             )
    #             val = (gradient_matrix[row+1,col+1]+ C_l,)
    #         # when pixel is in the rightmost col
    #         elif col == (num_of_cols - 1):
    #             C_r = abs(gradient_matrix[row,col-1] - gradient_matrix[row,col+1]) + abs(
    #                 gradient_matrix[row-1,col] - gradient_matrix[row,col+1]
    #             )
    #             val = (gradient_matrix[row+1,col-1] + C_r,)
    #         # when pixel is in the middle
    #         else:
    #             # calculate forward cost
    #             C_u = abs(gradient_matrix[row,col-1] - gradient_matrix[row,col+1])
    #             C_l = abs(gradient_matrix[row,col-1] - gradient_matrix[row,col+1]) + abs(
    #                 gradient_matrix[row-1,col] - gradient_matrix[row,col-1]
    #             )
    #             C_r = abs(gradient_matrix[row,col-1] - gradient_matrix[row,col+1]) + abs(
    #                 gradient_matrix[row-1,col] - gradient_matrix[row,col+1]
    #             )
    #             val = (gradient_matrix[row-1,col-1] + C_l, gradient_matrix[row+1,col-1] + C_r)

    #         min_val = min(gradient_matrix[row+1,col] + C_u, *val)
    #         gradient_matrix[row,col] += min_val
    
    # np.where(gradient_matrix == np.amin(gradient_matrix, axis=1)[num_of_rows - 1])

    # def pick_cheaper_seam(image, gradient_matrix, num_of_seams):
    #     new_image = np.copy(image)
    #     last_row = gradient_matrix[gradient_matrix.shape[0]-1]
    #     # sorted_row = np.sort(last_row)
    #     # cheaper_values = sorted_row[:num_of_seams]
    #     # indexes = np.where(last_row == cheaper_values)
    #     indexes = np.argpartition(last_row, num_of_seams)[:num_of_seams]
    #     current_row = gradient_matrix.shape[0] - 1
    #     # color last row
    #     for i in range(num_of_seams):
    #         new_image[current_row][int(indexes[i])] = np.array(colour)

    #     for col in indexes:
    #         current_row = new_image.shape[0] - 1
            
    #         while (current_row > 0):
    #             min_choice_col = 0
    #             if col == 0:
    #                 min_choice_col = col if gradient_matrix[current_row-1, col] <= gradient_matrix[current_row-1, col+1] else col + 1
    #             elif col == new_image.shape[1]-1:
    #                 min_choice_col = col if gradient_matrix[current_row-1, col] <= gradient_matrix[current_row-1, col-1] else col - 1
    #             else:
    #                 left, upper, right = gradient_matrix[current_row-1, col-1], gradient_matrix[current_row-1, col], gradient_matrix[current_row-1, col+1]
    #                 if (upper <= left) and (upper <= right):
    #                     column = col
    #                 elif left <= right:
    #                     column = col-1
    #                 else:
    #                     column = col+1
    #                 min_choice_col = column
    #             gradient_matrix[current_row, col] = 255
    #             current_row -= 1
    #             new_image[current_row, min_choice_col] = np.array(colour)
    #             col = min_choice_col
                
    #     # while(current_row > 0):
    #     #     for i in range(num_of_seams):
                
    #     #         new_image[current_row][indexes[i]] = colour

    #     return new_image
    # seam_image = pick_cheaper_seam(image, gradient_matrix, 146)
    # print(1)    
    # return seam_image