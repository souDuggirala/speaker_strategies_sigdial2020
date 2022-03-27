import io
import json
import numpy as np

#read in color names from vocab.json
file = open('vocab.json')
old_vocab_dict = json.load(file)
old_color_names = old_vocab_dict.keys()

#split by whitespace and make a unique list of words
#color_names = np.unique(np.array([old_color_name.split() for old_color_name in old_color_names]).flatten())
color_names = np.unique(np.array([color_word for color_name in old_color_names for color_word in color_name.split()]))

#turn into new dictionary
indices = np.arange(0, color_names.size).tolist()
vocab_dict = dict(zip(color_names, indices))

#write to file
out_file = open('comp_vocab.json', 'w')
json.dump(vocab_dict, out_file)
out_file.close()