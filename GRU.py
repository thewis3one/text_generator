import tensorflow as tf
import numpy as np
import os

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        predicted_logits = predicted_logits + self.prediction_mask

        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.chars_from_ids(predicted_ids)

        return predicted_chars, states

def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

def Load_Preprocess(path, seq_len, encoding=None):
    text = ''
    with open(path, 'r', encoding=encoding) as r:
        text = r.read()

    vocab = sorted(set(text))
    print(f'{len(vocab)} unique characters')

    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    sequences = ids_dataset.batch(seq_len+1, drop_remainder=True)

    return sequences, ids_from_chars, chars_from_ids

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def batch_data(sequences, BATCH_SIZE):
    dataset = sequences.map(split_input_target)

    BUFFER_SIZE = 10000

    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    return dataset

def create_model(dataset='None', vocab_size=0, embedding_dim=128, rnn_units=1024):
    model = MyModel(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)

    model.summary()

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("Mean loss:        ", example_batch_mean_loss)

    tf.exp(example_batch_mean_loss).numpy()

    model.compile(optimizer='adam', loss=loss)

    return model


def train(model, EPOCHS, dataset, name_of_field):
    try:
        model.load_weights('fields/' + name_of_field + '/weights.h5')
    except(FileNotFoundError):
        EPOCHS

    history = model.fit(dataset, epochs=EPOCHS)
    model.save_weights('fields/' + name_of_field + '/weights.h5')

    return history

def gen_text(text_len, one_step_model, input_text=' ', end_char='.'):
    states = None
    next_char = tf.constant([input_text])
    result = [next_char]

    for n in range(text_len):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    while(next_char != end_char):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    return result[0].numpy().decode('utf-8')
