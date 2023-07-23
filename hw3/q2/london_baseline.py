# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils

def calculate_acc_based_on_london(input_file):
    predictions = []
    total = 0
    correct = 0

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            predictions.append('London')

    total, correct = utils.evaluate_places(input_file, predictions)

    print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))

if __name__ == "__main__":
    calculate_acc_based_on_london("birth_dev.tsv")