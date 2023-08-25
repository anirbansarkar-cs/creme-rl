import numpy as np
from cremerl import shuffle

def get_swap_greedy(x, x_mut, tile_ranges):
    ori = x.copy()
    mut = x_mut.copy()
    for tile_range in tile_ranges:
        ori[:, tile_range[0]:tile_range[1]] = x_mut[:, tile_range[0]:tile_range[1]]
        mut[:, tile_range[0]:tile_range[1]] = x[:, tile_range[0]:tile_range[1]]

    return ori, mut

def generate_tile_ranges(sequence_length, window_size, stride):
    ranges = []
    start = np.arange(0, sequence_length - window_size + stride, stride)

    for s in start:
        e = min(s + window_size, sequence_length)
        ranges.append([s, e])

    if start[-1] + window_size - stride < sequence_length:  # Adjust the last range
        ranges[-1][1] = sequence_length

    return ranges


def get_batch(x, tile_range, tile_ranges_ori, trials):
    test_batch = []
    for i in range(trials):
        test_batch.append(x)
        x_mut = shuffle.dinuc_shuffle(x.copy())
        test_batch.append(x_mut)

        ori = x.copy()
        mut = x_mut.copy()
        
        ori, mut = get_swap_greedy(ori, mut, tile_ranges_ori)
        
        ori[:, tile_range[0]:tile_range[1]] = x_mut[:, tile_range[0]:tile_range[1]]
        mut[:, tile_range[0]:tile_range[1]] = x[:, tile_range[0]:tile_range[1]]
        
        test_batch.append(ori)
        test_batch.append(mut)

    #print(np.array(test_batch).shape)
    return np.array(test_batch)


def get_batch_score(pred, trials):

    score = []
    score_sep = []
    for i in range(0, pred.shape[0], 2):
        # print(f"Viewing number {i}")
        score1 = pred[0] - pred[i]
        score2 = pred[i+1] - pred[1]
        score.append((np.sum((score1, score2)[0])).tolist()) #np.sum(score1+score2, keepdims=True)
        score_sep.append((score1+score2).tolist())
        
    # print(score)
        
    final = np.sum(np.array(score), axis=0)/trials

    #max_ind = np.argmax(final)
    #block_ind = np.argmax(np.array(score)[:, max_ind])
    #print(np.array(total_score)[:, max_ind])
    total_score_sep = np.sum(np.array(score_sep), axis=0)/trials

    #print(np.max(score))
    return final



def extend_sequence(one_hot_sequence):
    A, L = one_hot_sequence.shape

    # Create an all-ones row
    ones_row = np.zeros(L)

    # Add the all-ones row to the original sequence
    new_sequence = np.vstack((one_hot_sequence, ones_row))

    return np.array(new_sequence, dtype='float32')

def taking_action(sequence_with_ones, tile_range):
    start_idx, end_idx = tile_range

    # Ensure the start_idx and end_idx are within valid bounds
    #if start_idx < 0 or start_idx >= sequence_with_ones.shape[1] or end_idx < 0 or end_idx >= sequence_with_ones.shape[1]:
    #    raise ValueError("Invalid tile range indices.")

    # Copy the input sequence to avoid modifying the original sequence
    modified_sequence = sequence_with_ones.copy()

    # Modify the last row within the specified tile range
    modified_sequence[-1, start_idx:end_idx] = 1

    return np.array(modified_sequence, dtype='float32')


def convert_elements(input_list):
    input_list = input_list.tolist()
    num_columns = 5  # Number of elements to process in each group

    # Calculate the number of elements needed to pad the list
    padding_length = num_columns - (len(input_list) % num_columns)
    last_value = input_list[-1]
    padded_list = input_list + [last_value] * padding_length

    # Convert the padded list to a NumPy array for efficient operations
    input_array = np.array(padded_list)
    reshaped_array = input_array.reshape(-1, num_columns)

    # Check if each row has the same value (all 0s or all 1s)
    row_all_zeros = np.all(reshaped_array == 0, axis=1)
    row_all_ones = np.all(reshaped_array == 1, axis=1)

    # Replace all 0s with 0 and all 1s with 1 in the result array
    output_array = np.where(row_all_zeros, 0, np.where(row_all_ones, 1, reshaped_array[:, 0]))

    # Flatten the result array to get the final output list
    output_list = output_array.flatten()

    return output_list
