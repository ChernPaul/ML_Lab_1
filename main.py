import sys
import numpy as np
import pandas as pd

FILEPATH_1_DATA = "data_files\\data.txt"
FILEPATH_2_DATA_TRUE = "data_files\\data_true.txt"
FILEPATH_3_DATA_FAKE = "data_files\\data_fake.txt"

HEADER_0_USERNAME = 'USERNAME'
HEADER_1_DATASET = 'DATASET'

MIN_INT = -sys.maxsize - 1
VALUE_OF_ZERO = 0
VALUE_OF_ONE = 1
VALUE_OF_TWO = 2
ANOMALY_BORDER_VALUE = 0.05


def get_dictionary_key_by_value(dictionary, value):
    for k, v in dictionary.items():
        if v == value:
            return k


def add_pair_in_state_to_index_dictionary(dictionary, state, index):
    dictionary.update({state: index})


def create_state_to_index_dictionary(set_of_conditions):
    result_dictionary = {MIN_INT: 0}
    number_of_elements = len(set_of_conditions)
    set_as_array = np.array(list(set_of_conditions))
    indexes = np.arange(number_of_elements) + 1
    func = np.vectorize(add_pair_in_state_to_index_dictionary)
    func(result_dictionary, set_as_array, indexes)
    return result_dictionary


def increase_count_in_matrix_from_pair_of_states(current_state, next_state, matrix):
    string_index = STATE_TO_INDEX_DICTIONARY.get(current_state)
    column_index = STATE_TO_INDEX_DICTIONARY.get(next_state)
    matrix[string_index, column_index] += 1


def divide_row_by_sum_of_row(row_index, matrix):
    sum_el_of_row = np.sum(matrix[row_index, :])
    if sum_el_of_row != 0:
        matrix[row_index] = np.divide(matrix[row_index], sum_el_of_row).astype(float)


def calculate_transfer_matrix(array_of_states, number_of_states):
    size_matrix_of_states = number_of_states + VALUE_OF_ONE
    size_of_array_dataset = array_of_states.size
    last_array_index = size_of_array_dataset - VALUE_OF_ONE
    pre_result_matrix = np.zeros((size_matrix_of_states, size_matrix_of_states))

    increase_count_in_matrix_from_pair_of_states(MIN_INT, array_of_states[VALUE_OF_ZERO], matrix=pre_result_matrix, )

    func = np.vectorize(increase_count_in_matrix_from_pair_of_states, excluded=['matrix'])
    func(array_of_states[0:last_array_index - 1], array_of_states[1:last_array_index], matrix=pre_result_matrix)

    strings_indexes = np.arange(size_matrix_of_states)
    func = np.vectorize(divide_row_by_sum_of_row, excluded=['matrix'])
    func(strings_indexes, matrix=pre_result_matrix)
    return pre_result_matrix


# def update_user_to_transfer_matrix_dictionary(dictionary, user, transfer_matrix):
#     dictionary.update({user: transfer_matrix})


def update_user_to_transfer_matrix_dictionary_special(dictionary, user, list_of_strings, number_of_states):
    array_of_sets = get_np_array_of_int(list_of_strings)
    transfer_matrix = calculate_transfer_matrix(array_of_sets, number_of_states)
    dictionary.update({user: transfer_matrix})


def create_user_to_transfer_matrix_dictionary(array_of_users, array_of_datasets_as_list_of_strings, number_of_states):
    result_dictionary = {}
    func = np.vectorize(update_user_to_transfer_matrix_dictionary_special, excluded=['number_of_states'])
    func(result_dictionary, array_of_users, array_of_datasets_as_list_of_strings, number_of_states=number_of_states)
    return result_dictionary


def get_string_dataset_as_list_of_datasets(dataset_as_string, list_to_fill):
    list_to_fill.append(dataset_as_string.split(';'))


def convert_array_of_string_datasets(array_of_string_datasets):
    result_list = []
    func = np.vectorize(get_string_dataset_as_list_of_datasets, excluded=['list_to_fill'], cache=True)
    func(array_of_string_datasets, list_to_fill=result_list)
    # del result_list[0]
    result_array = np.array(result_list)
    return result_array


def get_np_array_of_int(list_of_strings):
    return np.array(list_of_strings).astype(int)


def lazy_print_matrix(transfer_matrix):
    result = ""
    number_of_states = transfer_matrix.shape[0]
    sub_result = ""

    for i in range(0, number_of_states, 1):
        sub_result += print_matrix_str(transfer_matrix[i, :], i) + '\n'

    result = result + "\n" + sub_result
    return result


def print_matrix_str(str_of_matrix, row_index):
    result = ""
    for i in range(0, str_of_matrix.shape[0], 1):
        result += "   " + str(get_dictionary_key_by_value(STATE_TO_INDEX_DICTIONARY, row_index))+"|" + \
                  str(get_dictionary_key_by_value(STATE_TO_INDEX_DICTIONARY, i))+"|" + str(np.round(str_of_matrix[i], 4))
    return result


# def anomaly_predictor(transfer_matrix_value, border_value):
#     if border_value > transfer_matrix_value:
#         return True
#     return False

def set_list_of_predictions(transfer_matrix_value, border_value, list_):
    if border_value >= transfer_matrix_value:
        list_.append(True)
    else:
        list_.append(False)

# def get_transfer_matrix_for_user(username, number_of_states):
#     try:
#         result = USER_TO_TRANSFER_MATRIX_DICTIONARY.get(username)
#         return result
#     except KeyError:
#         return np.zeros((number_of_states, number_of_states))


def is_user_in_dictionary(username):
    try:
        USER_TO_TRANSFER_MATRIX_DICTIONARY.get(username)
        return True
    except KeyError:
        return False

def is_state_in_dictionary(state):
    try:
        tmp = STATE_TO_INDEX_DICTIONARY[state]
        return True
    except KeyError:
        return False


def add_transfer_value_for_pair_of_states_to_list(initial_state, next_state, transfer_matrix, list_):
    if is_state_in_dictionary(int(initial_state)) and is_state_in_dictionary(int(next_state)):
        row_index = STATE_TO_INDEX_DICTIONARY[int(initial_state)]
        column_index = STATE_TO_INDEX_DICTIONARY[int(next_state)]
        list_.append(transfer_matrix[row_index, column_index])
    else:
        list_.append(0)


def detect_anomalies_for_user_in_dataset(username, dataset):
    output_size = len(dataset)

    transfer_matrix = USER_TO_TRANSFER_MATRIX_DICTIONARY[username]
    matrix_values = []

    matrix_values.append(transfer_matrix[0, STATE_TO_INDEX_DICTIONARY[int(dataset[0])]])

    func = np.vectorize(add_transfer_value_for_pair_of_states_to_list, excluded=['transfer_matrix', 'list_'], cache=True)
    func(dataset[0:output_size - VALUE_OF_ONE], dataset[1:output_size], transfer_matrix=transfer_matrix, list_=matrix_values)

    res_list_of_predictions = []

    func1 = np.vectorize(set_list_of_predictions, excluded=['border_value', 'list_'], cache=True)
    func1(matrix_values, border_value=ANOMALY_BORDER_VALUE, list_=res_list_of_predictions)

    result = np.array(res_list_of_predictions)
    return result



def detect_anomalies_for_users_in_datasets(array_of_usernames, list_of_datasets_as_list):
    result_list = []
    for i in range(0, array_of_usernames.size, 1):
        if not is_user_in_dictionary(array_of_usernames[i]):
            continue
        result_list.append(detect_anomalies_for_user_in_dataset(array_of_usernames[i], list_of_datasets_as_list[i]))

    result = np.array(result_list)
    return result


def calculate_number_of_anomalies_in_dataset(dataset):
    return np.sum(dataset)


def is_user_have_anomaly_behaviour(array_of_predictions):
    if np.sum(array_of_predictions) > 0:
        return True
    else:
        return False


def calculate_number_of_anomaly_users(array_of_arrays_of_predictions):
    result = []
    for i in range(0, array_of_arrays_of_predictions.shape[0], 1):
        result.append(is_user_have_anomaly_behaviour(array_of_arrays_of_predictions[i]))
    return np.sum(result)

def print_detection_results(array_of_users, array_of_predictions):
    for i in range(0, array_of_users.size, 1):
        print(array_of_users[i] + ": ", array_of_predictions[i])


if __name__ == '__main__':

    data = pd.read_csv(FILEPATH_1_DATA, header=0, sep=':')

    dataset_column = data[HEADER_1_DATASET]
    dataset_united_fields = ';'.join(dataset_column)
    entire_dataset_array = np.array(dataset_united_fields.split(';')).astype(int)
    unique_dataset = set(entire_dataset_array)

    global STATE_TO_INDEX_DICTIONARY
    STATE_TO_INDEX_DICTIONARY = create_state_to_index_dictionary(unique_dataset)

    array_of_users = data[HEADER_0_USERNAME].to_numpy()
    array_of_datasets_as_array_of_lists = convert_array_of_string_datasets(dataset_column.to_numpy())

    global USER_TO_TRANSFER_MATRIX_DICTIONARY
    USER_TO_TRANSFER_MATRIX_DICTIONARY = create_user_to_transfer_matrix_dictionary(array_of_users,
                                                                                   array_of_datasets_as_array_of_lists,
                                                                                   len(unique_dataset))

    data_false = pd.read_csv(FILEPATH_3_DATA_FAKE, header=0, sep=':')
    array_of_users_fake = data_false[HEADER_0_USERNAME].to_numpy()
    array_of_datasets_as_array_of_lists_fake = convert_array_of_string_datasets(data_false[HEADER_1_DATASET].to_numpy())

    data_true = pd.read_csv(FILEPATH_2_DATA_TRUE, header=0, sep=':')
    array_of_users_true = data_true[HEADER_0_USERNAME].to_numpy()
    array_of_datasets_as_array_of_lists_true = convert_array_of_string_datasets(data_true[HEADER_1_DATASET].to_numpy())


    test_fake = detect_anomalies_for_users_in_datasets(array_of_users_fake, array_of_datasets_as_array_of_lists_fake)
    print_detection_results(array_of_users_fake, test_fake)
    print()
    print("Число аномальных пользователей в дата фейк", calculate_number_of_anomaly_users(test_fake))
    print()
    test_true = detect_anomalies_for_users_in_datasets(array_of_users_true, array_of_datasets_as_array_of_lists_true)
    print_detection_results(array_of_users_true, test_true)
    print()
    print("Число аномальных пользователей в дата тру", calculate_number_of_anomaly_users(test_true))

