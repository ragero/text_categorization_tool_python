import itertools

def generate_parameters_list(parameters): 
    
  all_parameters = []
  for values in parameters.values(): 
    all_parameters.append(values)
  all_permutations = []
  for combination in itertools.product(*all_parameters):
    all_permutations.append(combination)
  parameters_list = []
  for combination in all_permutations: 
    param = {}
    for i, key in enumerate(parameters.keys()): 
      param[key] = combination[i]
    parameters_list.append(param)
  return parameters_list