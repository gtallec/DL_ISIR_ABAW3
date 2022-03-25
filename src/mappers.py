import tensorflow as tf

def projective_map(projection):
    tensor_projection = tf.constant(projection, dtype=tf.int64)

    def project(features, labels):
        projected_labels = tf.gather(labels,
                                     tensor_projection,
                                     axis=1)
        return features, projected_labels

    return project

def label_projection(projection):
    tensor_projection = tf.constant(projection, dtype=tf.int64)

    def project(labels):
        projected_labels = tf.gather(labels,
                                     tensor_projection,
                                     axis=-1)
        return projected_labels
    return project

def get_labels(images, labels):
    return labels

