import os
import argparse
from tqdm import tqdm

def extract_data(path):
   with open(path, 'r') as file:
    content = file.readlines()

    model = content[0].split(' ')[-1].rstrip()
    dataset = content[1].split(' ')[-1].rstrip()
    data  = content[-15:]

    timespent = data[-1].split(' ')[-2].rstrip()
    table = data[1:14]

    print(model)
    print(dataset)
    print(timespent)

    d = []
    for line in table:
      for item in line.rstrip().split(' '):
        if item != "":
          d.append(item)

    labels = ['loss','accuracy','true_positives','true_negatives','false_positives','false_negatives','val_loss','val_accuracy','val_true_positives','val_true_negatives','val_false_positives','val_false_negatives']
    
    d = d[5:]
    t = {}
    index = 2
    size = 4
    for label in labels:
      t[label] = d[index:(index + size)]
      index += 6
    
    save_json(model, t)

def save(model, table):
  with open('results/models/models.md', 'a') as file:
    file.write(f'# {model}\n\n')
    file.write(f'|item|max|mean|min|std|\n')
    file.write(f'|--|--|--|--|--|\n')
    for key, line in table.items():
      file.write(f'|{key}|{line[0]}|{line[1]}|{line[2]}|{line[3]}|\n')

def save_json(model, table):
  with open('results/models/models.json', 'a') as file:
    file.write(f'{{\n    "name" : "{model}",\n')
    for key, line in table.items():
      file.write(f'    "{key}": [\n    {line[0]},\n    {line[1]},\n    {line[2]},\n    {line[3]}],\n')
    file.write(f'\n}}')

def cli_setup():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', '-i', help="Input log", required=True)
  return parser

def main():
  parser = cli_setup()
  args = parser.parse_args()

  input_file = args.input

  os.listdir(input_file)

  onlyfiles = [f for f in os.listdir(input_file) if os.path.isfile(os.path.join(input_file, f))]
  for mfile in onlyfiles:
    extract_data(f'logs/{mfile}')

if __name__ == '__main__':
    main()