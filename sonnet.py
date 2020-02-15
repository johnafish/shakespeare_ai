""" Sonnet
Generate rhyming shakespearean sonnets from a file of 10-syllable phrases.
"""
import re
import pronouncing
import random

# Generate rhyming couplet
def generate_couplet(rhymes):
    while 1:
        i = random.randrange(len(rhymes.keys()))
        if len(rhymes[i]) >= 1:
            pool = rhymes[i] + [i]
            return random.sample(set(pool), 2)

# Generate quatrain from two interspliced couplets
def generate_quatrain(rhymes):
    a = generate_couplet(rhymes)
    b = generate_couplet(rhymes)
    return [a[0], b[0], a[1], b[1]]

# Generate sonnet from 3 quatrains and a couplet
def generate_sonnet(rhymes):
    a = generate_quatrain(rhymes)
    b = generate_quatrain(rhymes)
    c = generate_quatrain(rhymes)
    d = generate_couplet(rhymes)
    return a + b + c + d

# Convert sonnet from index array to string
def conv_sonnet(indices, line_dict):
    sonnet = ""
    for i in indices:
        sonnet += line_dict[i] + "\n"
    return sonnet

# Get lines from iambic file
iambic_lines = open("iambic.txt", "r")
lines = [line.strip() for line in iambic_lines.readlines()]

# Array of final word of each line
final_words = [re.sub(r'[^\w\s]', '', line.split(" ")[-1]) for line in lines]

# Enumerated dictionaries
line_dict = {i:w for i, w in enumerate(lines)}
word_dict = {i:w for i, w in enumerate(final_words)}

# Rhyming indices
# { 0 : [1, 2, 3] } means that line 0 rhymes with lines 1, 2, 3
rhyme_dict = {}

for i, word1 in word_dict.items():
    # Get potential rhymes from pronouncing package
    potential_rhymes = pronouncing.rhymes(word1)
    rhymes = []
    for x, word2 in word_dict.items():
        # Disallow rhyming with self or same word
        if word1 != word2:
            if word2 in potential_rhymes:
                rhymes.append(x)
    rhyme_dict[i] = rhymes

# Write generated sonnets to text file
sonnet_file = open("sonnets.txt", "w+")
NUM_SONNETS = 50
for i in range(NUM_SONNETS):
    sonnet = conv_sonnet(generate_sonnet(rhyme_dict), line_dict)
    sonnet_file.write(sonnet)
    sonnet_file.write("\n")
sonnet_file.close()
