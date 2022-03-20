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
    
def gradient_magnitude(image, colour_wts):
    """
    Calculates the gradient image of a given image
    :param image: The original image
    :param colour_wts: the weights of each colour in rgb (> 0) 
    :returns: The gradient image
    """
    greyscale = get_greyscale_image(image, colour_wts)

    # moving the rows to prepare I(x+1,y) and I(x, y+1)
    x_forward = np.roll(greyscale, -1 ,axis=0)
    y_forward = np.roll(greyscale, -1 ,axis=1)
    # perform the subtraction I(x+1,y)- I(x,y) and I(x, y+1) - I(x,y)
    top_x = (y_forward - greyscale)
    top_y = (x_forward - greyscale)

    # perform addition,square and root of gradient to calculate gradient magnitude
    top_x, top_y = np.square(top_x), np.square(top_y)
    final_res = top_x + top_y
    final_res = np.sqrt(final_res)

    gradient = final_res
    
    return gradient
    

def cost_matrix(image, grey_scale_wt):

    # gray scale scalars
    greyscale_wt = [0.299, 0.587, 0.114]

    # recieve gradient matrix
    gradient_matrix = gradient_magnitude(image, greyscale_wt)
    grey_image = get_greyscale_image(image, greyscale_wt)
    
    num_of_rows = gradient_matrix.shape[0]
    num_of_cols = gradient_matrix.shape[1]

    # calculate cheapest paths matrix using dynamic programming
    for row in range(1, num_of_rows):
    
        for col in range(num_of_cols ):
            C_u = 0
            # when pixel is in the leftmost col
            if col == 0:
                C_r = abs(grey_image[row-1,col] - grey_image[row,col+1]) + abs(grey_image[row,num_of_cols-1] - grey_image[row,col+1])
                val = (gradient_matrix[row-1,col+1] + C_r,)
            # when pixel is in the rightmost col
            elif col == (num_of_cols - 1):
                C_l = abs(grey_image[row-1,col] - grey_image[row,col-1]) + abs(grey_image[row,col-1] - grey_image[row,0])
                val = (gradient_matrix[row-1,col-1] + C_l,)
            # when pixel is in the middle
            else:
                # calculate forward cost
                C_u = abs(grey_image[row,col-1] - grey_image[row,col+1])
                C_l = C_u + abs(grey_image[row-1,col] - grey_image[row,col-1])
                C_r = C_u + abs(grey_image[row-1,col] - grey_image[row,col+1])
                val = (gradient_matrix[row-1,col+1] + C_r, gradient_matrix[row-1,col-1] + C_l)
            
            min_val = min(gradient_matrix[row-1,col] + C_u, *val)
            gradient_matrix[row,col] += min_val
    return gradient_matrix

def remove_seam(image, seam):
    mask = np.ones(image.shape, dtype=bool)
    for pixel in seam:
        mask[pixel[0], pixel[1]] = False
    image = image[mask].reshape(image.shape[0],image.shape[1] - 1, image.shape[2])
    return image



def get_best_seam(calculated_cost_matrix):
    seam = []
    height = calculated_cost_matrix.shape[0]
    best_column = np.argmin(calculated_cost_matrix[height - 1])
    seam.append([height-1, best_column])
    for row in range(height -2, -1, -1):
        if best_column == 0:
            best_column += np.argmin(calculated_cost_matrix[row, best_column : best_column + 2])
        elif best_column == (height -1):
            best_column += np.argmin(calculated_cost_matrix[row, best_column -1 : best_column +1]) - 1
        else:
            best_column += np.argmin(calculated_cost_matrix[row, (best_column -1): (best_column + 2)]) - 1
        seam.append([row,best_column])
    
    return seam

def visualise_seams(image, new_shape, show_horizontal, colour):
    """
    Visualises the seams that would be removed when reshaping an image to new image (see example in notebook)
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :param show_horizontal: the carving scheme to be used.
    :param colour: the colour of the seams (an array of size 3)
    :returns: an image where the removed seams have been coloured.
    """
#     ###Your code here###
#     ###**************###

    # gray scale scalars
    greyscale_wt = [0.299, 0.587, 0.114]

    currect_shape = new_shape[1]

    
    if show_horizontal:
        image = image.transpose([1,0,2])
        currect_shape = new_shape[0]
    
    # compute the number of seams for either horizontal or vertical case
    num_of_vertical_seams = image.shape[1] - currect_shape

    image = np.copy(image)
    edited_image = np.copy(image)
    seams_to_remove = []

    if num_of_vertical_seams < 0:
        return "support only seam removal"
    for i in range(num_of_vertical_seams):
         calculated_cost_matrix = cost_matrix(edited_image, greyscale_wt)
         seam = get_best_seam(calculated_cost_matrix)
         seams_to_remove.append(seam)
         edited_image = remove_seam(edited_image, seam)
    
    # used for reshaping :)
    # when 'colour' flag equals -1, we want to delete seams without restoring them
    if colour != -1:
        resized_image = paint_seams(edited_image, seams_to_remove, colour)
    else:
        resized_image = edited_image

    if show_horizontal:
        resized_image = resized_image.transpose([1,0,2])

    return resized_image



def paint_seams(image, seams_to_remove, colour_to_paint):
    """
        this function adds the seams provided in the 'seam_list' to the given image
        In this project its used for both painting the seams we want to remove and to enlarge the image
        if colour_to_paint is an RGB color, the seams will be inserted with the color.
        
        - if 'colour_to_paint' is -1, the seams will be duplicated instead of being colored
    """
    image_array = [np.copy(image[i]) for i in range(image.shape[0])]
    # when we want to paint the seams in the given 'colour_to_paint' color
    if colour_to_paint != -1:
        colour_to_paint = np.array(colour_to_paint)
        for seam in reversed(seams_to_remove):
            for place in seam:
                row = place[0]
                col = place[1]
                if col >= len(image_array[row]):
                    image_array[row] = np.append(image_array[row], [colour_to_paint], axis=0)
                
                else:
                    image_array[row] = np.insert(image_array[row], col, [colour_to_paint], axis=0)

    # if we want to add more seams - for reshape
    else: 
        for seam in reversed(seams_to_remove):
            for place in seam:
                row = place[0]
                col = place[1]

                # the only change is this line (it is code duplication but saves repeatedly checking the same if statement - better time complexity!)
                colour_to_paint = np.array(image[row,col])

                if col >= len(image_array[row]):
                    image_array[row] = np.append(image_array[row], [colour_to_paint], axis=0)
                else:
                    image_array[row] = np.insert(image_array[row], col, [colour_to_paint], axis=0)

    image = np.array(image_array, dtype=np.uint8)
    return image



def pick_cheaper_seam(image, gradient_matrix, num_of_seams):
        new_image = np.copy(image)
        seam_list = []
        last_row = gradient_matrix[gradient_matrix.shape[0]-1]
        num_of_cols = gradient_matrix.shape[1]

        # used to select the indices the cheapers k values for the last row
        indices = np.argpartition(last_row, num_of_seams)[:num_of_seams]

        current_row = gradient_matrix.shape[0] - 1

        # pick 'num_of_seams' seams from image at once
        for col in indices:
            current_row = new_image.shape[0] - 1
            seam = []
            seam.append([current_row, col])
            
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
                # gradient_matrix[current_row, col] = np.Infinity
                current_row -= 1
                # build up the seam
                seam.append([current_row, min_choice_col])
                col = min_choice_col
            # add seam to seam's list
            seam_list.append(seam)

        return seam_list
    
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
    greyscale_wt = [0.299, 0.587, 0.114]
    #check if expending the image is needed
    expanding_image = (carving_scheme and image.shape[0] < new_shape[0]) or (not carving_scheme and image.shape[1] < new_shape[1])
    
    if expanding_image:
        if carving_scheme:
            image = image.transpose([1,0,2])
            num_of_seams = new_shape[0] - image.shape[1]
        else:
            num_of_seams = new_shape[1] - image.shape[1]
        
        image = np.copy(image)

        gradient_matrix = gradient_magnitude(image, greyscale_wt)
        seam_list = pick_cheaper_seam(image, gradient_matrix, num_of_seams)
        # The '-1' flag in the 'colour' argument indicates no painting of seams is needed
        new_image = paint_seams(image, seam_list, -1)

        if carving_scheme:
            new_image = new_image.transpose([1,0,2])
    else: 
        # for removal of seams 
        # the '-1' flag in the 'colour' argument indicates no painting of seams is needed
        new_image = visualise_seams(image, new_shape, carving_scheme, -1)
    return new_image