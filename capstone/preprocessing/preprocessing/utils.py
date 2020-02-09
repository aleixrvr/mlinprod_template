from tensorflow.keras.models import load_model


def labels_to_indexes(labels, labels_list):

    labels_index = {labels_list[ind]: ind for ind in range(len(labels_list))}
    indexes = [labels_index[label] for label in labels]
    return indexes, labels_index
