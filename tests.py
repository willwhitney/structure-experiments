import scipy.stats
import torch

def mv_logpdf(p, target):
    return scipy.stats.multivariate_normal.logpdf(target.data.numpy(),
                                                  mean=p[0].data[0].numpy(),
                                                  cov=p[1].data[0].numpy())

def test_KL(q, p):
    n_samples = 10000
    results = torch.zeros(n_samples)
    for i in range(n_samples):
        thing = sample(q)
        log_q_sample = mv_logpdf(q, thing)
        log_p_sample = mv_logpdf(p, thing)
        results[i] = log_q_sample - log_p_sample

    return results.mean()

# test KL module with sampling
z1_prior = (Variable(torch.zeros(batch_size, hidden_dim)),
            Variable(torch.ones(batch_size, hidden_dim)))
other = (Variable(torch.zeros(batch_size, hidden_dim) + 1.3),
         Variable(torch.ones(batch_size, hidden_dim) * 2))

print(KL(z1_prior, other))
print(test_KL(z1_prior, other))


# test LL model
p = (Variable(torch.zeros(batch_size, data_dim)),
     Variable(torch.ones(batch_size, data_dim)))
target = Variable(torch.ones(batch_size, data_dim))
for i in range(100):
    p = (Variable(torch.rand(batch_size, data_dim)),
         Variable(torch.rand(batch_size, data_dim)))
    target = Variable(torch.rand(batch_size, data_dim))
    print((LL(p, target).data[0] - mv_logpdf(p, target)) / mv_logpdf(p, target))
