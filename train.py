import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from model import MyModel
from onestep import OneStep
from preprocessing import delete_line_with_word
import numpy as np
import os
import time
import re



#Funzione che converte gli IDs in testo

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)



#Funzione che prende in input una sequenza, e genera input e target text

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text



#------------INIZIO DEL PROGRAMMA-------------#



#Download and read data
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
""" text = open(path_to_file, 'rb').read().decode(encoding='utf-8')"""



#Preprocessing data

delete_line_with_word('chat.txt', 'sticker')
delete_line_with_word('chat.txt', 'audio omesso')
text = open(r'chat.txt').read().lower()
text = re.sub("[\(\[].*?[\)\]]", "", text)

vocab = sorted(set(text))
print('{} caratteri unici.'.format(len(vocab)))



#Prima di iniziare ad allenare il modello, dobbiamo convertire le stringhe in numeri

ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True)



#Dividiamo il testo in più parti

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 100
examples_per_epoch = len(text) # (seq_length+1)
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
dataset = sequences.map(split_input_target)



#Prima di dare in pasto i dati al programma, randomizziamo i dati e li mettiamo in piccoli "batch"

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = (dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

model = MyModel(vocab_size=len(ids_from_chars.get_vocabulary()), embedding_dim=embedding_dim, rnn_units=rnn_units)



#Controlliamo la lunghezza dell'output

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()



sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()



#Dato che il modello ritorna logits, impostiamo il flag dei logits
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
mean_loss = example_batch_loss.numpy().mean()
tf.exp(mean_loss).numpy()

model.compile(optimizer='adam', loss=loss)



#Definisco i checkpoint

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "checkpoint_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)



EPOCHS = 30

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
start = time.time()
states = None
next_char = tf.constant(['ROMEO:'])
result = [next_char]

for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()

print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)



#Salvataggio del modello

tf.saved_model.save(one_step_model, 'one_step')