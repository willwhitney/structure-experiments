import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable


from util import *
# from generations import *
from params import *

# results_path = "/home/wwhitney/speedy/code/structure-experiments/results/"
results_path = "/misc/vlgscratch3/FergusGroup/wwhitney/structure-experiments/results/"

checkpoint = torch.load(results_path + opt.load + '/model.t7')
model = checkpoint['model'].type(dtype)
i = checkpoint['i']
cp_opt = checkpoint['opt']

setattrs(opt, cp_opt, exceptions=['load', 'resume', 'use_loaded_opt'])
opt.name = cp_opt['name'] + '_'

opt.save = results_path + opt.name
make_result_folder(opt.save)
write_options(opt.save)

data_path = results_path + opt.load + '/dataset.t7'
print("Loading stored dataset from {}".format(data_path))
data_checkpoint = torch.load(data_path)
train_data = data_checkpoint['train_data']
test_data = data_checkpoint['test_data']

train_loader = DataLoader(train_data,
                          num_workers=0,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_data,
                         num_workers=0,
                         batch_size=batch_size,
                         shuffle=True)

def sequence_input(seq):
    return [Variable(x.type(dtype)) for x in seq]

def save_interpolation(label, model, priming, scale):
    for p in priming:
        p.volatile = True

    samples = model.generate_interpolations(priming, 50, scale)
    samples = [[x.data for x in sample_row]
               for sample_row in samples]
    for j in range(10):
        image = [[x[j] #.view(3,image_width,image_width)
                  for x in sample_row]
                  for sample_row in samples]
        save_tensors_image(
            opt.save + '/' + label +'_interp_'+str(j)+'.png',
            image)
        image = [x[2:] for x in image]
        stacked = [image_tensor([image[i][t] for i in range(len(image))])
                   for t in range(len(image[0]))]
        save_gif(opt.save + '/' + label +'_interp_' + str(j) + '.gif',
                 stacked,
                 bounce=True,
                 duration=0.1)

for sequence in train_loader:
    sequence.transpose_(0, 1)
    sequence = sequence_input(sequence)

    for scale in [1, 5, 10, 15, 20]:
        save_interpolation(str(scale), model, sequence[:2], scale)
    break
