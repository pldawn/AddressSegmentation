import sys


class Merge:
  def __init__(self):
    self.schema = ['B', 'I', 'E', 'S', 'O']
    self.entities = ['COUNTRY', 'PROVINCE', 'CITY', 'DISTRICT', 'COUNTY', 'TOWN', 'TOWNSHIP', 'BLOCK', 'VILLAGE', 'REGIONNUM', 'STREET', 'STREETNUM', 'LANDMARK', 'BUILDING', 'FLOOR', 'TABLET']

  def merge_to_entities(self, inputs, labels):
    nn = len(inputs)
    if nn != len(labels):
      print('length of inputs dismatch length of labels')
      sys.exit(1)
    output = ''
    bag = []  # current merging inputs and labels
    mark = '' # current merging entity

    for i in range(nn):
      if labels[i][0] not in self.schema or labels[i][2:] not in self.entities:
        if bag:
          output += ' '.join(bag) + ' '
          bag = []
        output += inputs[i] + '/' + 'X' + ' '
        continue

      if labels[i] == 'O':
        output += inputs[i] + '/' + labels[i] + ' '
        continue

      if labels[i][0] == 'S':
        output += '[' + labels[i][2:] + ' ' + inputs[i] + '/' + labels[i] + ']' + ' '
        continue

      if labels[i][0] == 'B':
        if not bag:
          bag.append(inputs[i] + '/' + labels[i])
          mark = labels[i][2:]
        else:
          output += ' '.join(bag) + ' '
          bag = []
          bag.append(inputs[i] + '/' + labels[i])
          mark = labels[i][2:]
        continue

      if labels[i][0] == 'I':
        if not bag:
          output += inputs[i] + '/' + labels[i] + ' '
        elif mark != labels[i][2:]:
          output += ' '.join(bag) + ' '
          bag = []
          mark = ''
          output += inputs[i] + '/' + labels[i] + ' '
        else:
          bag.append(inputs[i] + '/' + labels[i])
        continue

      if labels[i][0] == 'E':
        if not bag:
          output += inputs[i] + '/' + labels[i] + ' '
        elif mark != labels[i][2:]:
          output += ' '.join(bag) + ' '
          bag = []
          mark = ''
          output += inputs[i] + '/' + labels[i] + ' '
        else:
          bag.append(inputs[i] + '/' + labels[i])
          output += '[' + labels[i][2:] + ' ' + ' '.join(bag) + ']' + ' '
          bag = []
          mark = ''
        continue
    if bag:
      output += ' '.join(bag)
    output = output.strip()
    return output


  def mergeline(self, inputs, labels):
    inputs = inputs.strip().split(' ')
    labels = labels.strip().split(' ')
    output = self.merge_to_entities(inputs, labels)
    return output


def main(argv):
  argc = len(argv)
  if argc < 4:
    print("Usage:%s <input> <label> <output>" % (argv[0]))
    sys.exit(1)
  inputsPath = argv[1]
  labelsPath = argv[2]
  outputsPath = argv[3]
  merge = Merge()
  merge.mergefile(inputsPath, labelsPath, outputsPath)


if __name__ == '__main__':
  main(sys.argv)
