from sklearn.preprocessing import LabelEncoder


def fit_label_encoder(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)
    return encoder


def transform_label_encoder(encoder, labels):
    encoded_labels = encoder.transform(labels)
    return encoded_labels

