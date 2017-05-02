generations, _,_ = model(sequence)
mus_data = [gen[0].data for gen in generations]
seq_data = [x.data for x in sequence]
for j in range(5):
    mus = [x[j].view(3,image_width,image_width)
           for x in mus_data]
    truth = [x[j].view(3,image_width,image_width)
             for x in seq_data]
    save_tensors_image('full_'+str(j)+'.png',
                       [truth, mus])



generating_latent = z_0[0].clone()
generations = model.generator(generating_latent)
mus_data = generations[0].data

for j in range(5):
    mus = mus_data[j].view(3,image_width,image_width)
    save_tensors_image('original_'+str(j)+'.png', [mus])


for l in range(8):
    generating_latent = z_0[0].clone()
    generating_latent[:, l*25 : (l+1)*25].data.copy_(z_1_latents[l].data)
    generations = model.generator(generating_latent)
    mus_data = generations[0].data
    for j in range(5):
        mus = mus_data[j].view(3,image_width,image_width)
        save_tensors_image('replace_'+str(j)+'_'+str(l)+'.png', [mus])
