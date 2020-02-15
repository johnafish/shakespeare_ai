from big_phoney import BigPhoney
import os

def write_iambic(input_file, output_file):
    for line in input_file.readlines():
        syllables = phoney.count_syllables(line)
        if syllables == 10:
            output_file.write(line)

# Initialization
phoney = BigPhoney()
output_file = open("iambic.txt", "a")

for filename in os.listdir(os.getcwd()+"/outputs"):
    print(filename)
    input_file = open("outputs/"+filename, "r+")
    write_iambic(input_file, output_file)
    input_file.close()

output_file.close()
