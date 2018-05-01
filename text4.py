import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.utils import np_utils
from keras.optimizers import RMSprop
import tensorflow as tf

sequence_len = 30
step_len = 3
hidden_dim = 512
dropout = .4
batch_size = 32
gen_len = 500

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


# From keras lstm thing
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


raw_text = open("1984_input.txt").read().lower()
split_text = set(raw_text.split())
words_list = raw_text.split()

num_words = len(split_text)

word_to_idx = {c: i for i, c in enumerate(split_text)}
idx_to_word = {i: c for i, c in enumerate(split_text)}

sequences = []
next_words = []
for i in range(0, len(words_list) - sequence_len, step_len):
    sequences.append([word_to_idx[word] for word in words_list[i:i + sequence_len]])
    next_words.append(word_to_idx[words_list[i + sequence_len]])

X = np.zeros((len(sequences), sequence_len, num_words), dtype=np.bool_)
Y = np.zeros((len(sequences), num_words), dtype=np.bool_)
for i in range(len(sequences)):
    X[i] = np_utils.to_categorical(sequences[i], num_words)
    Y[i] = np_utils.to_categorical(next_words[i], num_words)

model = Sequential()
model.add(LSTM(hidden_dim, return_sequences=True, input_shape=(sequence_len, num_words), dropout=dropout))
model.add(LSTM(hidden_dim, dropout=dropout))
model.add(Dense(num_words, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])
model.load_weights("text4_weights.hdf5")

for i in range(1, 300):
    model.fit(X, Y, batch_size=batch_size, epochs=1)
    model.save_weights("text4_weights.hdf5")

    start_ind = np.random.randint(len(words_list) - sequence_len)
    start_seed = words_list[start_ind:start_ind + sequence_len]
    with open("text4_out.txt", mode="a") as file:
        file.write("Using seed: {}\n".format(" ".join(start_seed)))
        for diversity in [.25, .5, .75, 1]:
            seed = start_seed[:]
            file.write("Using diversity: {}\n".format(diversity))
            generated = " ".join(seed) + " "
            for curr_word in range(gen_len):
                x = np.zeros((1, sequence_len, num_words))
                curr_seq = [word_to_idx[word] for word in seed]
                x[0,:,:] = np_utils.to_categorical(curr_seq, num_words)
                preds = model.predict(x)[0]
                index = sample(preds, diversity)
                gen_word = idx_to_word[index]
                generated += gen_word + " "
                seed.append(gen_word)
                seed = seed[1:]
            file.write(generated + "\n")
        file.write("\n\n" + "_"*50 + "\n\n")
