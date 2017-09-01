# import torch
# import progressbar
# import traceback
# import logging
# import pdb

# from params import *
# from generations import save_all_generations

# class Bookkeeper:
#     def __init__(self, i):
#         self.step = i
#         self.last_covariance = i
#         self.reset_stats()

#     def update(self, i, model, sequence, generations, 
#                nll, divergences, batch_z_vars):
#         self.step = i
#         seq_divergence, seq_prior_div, seq_trans_div = divergences
#         self.mean_divergence += seq_divergence.data[0]
#         self.mean_prior_div += seq_prior_div.data[0]
#         self.mean_trans_div += seq_trans_div.data[0]
#         self.mean_nll += nll.data[0]

#         self.mean_loss += nll.data[0] + seq_divergence.data[0]

#         p_vars, q_vars = batch_z_vars

#         self.q_var_means.append(sum([v.data.mean() for v in q_vars]) / \
#                                len(q_vars))
#         self.q_var_min = min(*[v.data.min() for v in q_vars], self.q_var_min)
#         self.q_var_max = max(*[v.data.max() for v in q_vars], self.q_var_max)

#         if opt.seq_len > 1:
#             self.p_var_means.append(sum([v.data.mean() for v in p_vars]) / \
#                                len(p_vars))
#             self.p_var_min = min(*[v.data.min() for v in p_vars], 
#                                  self.p_var_min)
#             self.p_var_max = max(*[v.data.max() for v in p_vars], 
#                                  self.p_var_max)

#         self.progress.update(i % opt.print_every)
#         is_update_time = (i >= opt.max_steps or
#                           (i % opt.print_every == 0 and i > 0))
#         if is_update_time:
#             self.progress.finish()
#             clear_progressbar()
#             elapsed_time = self.progress.end_time - self.progress.start_time
#             elapsed_seconds = elapsed_time.total_seconds()
#             batches = opt.print_every / opt.batch_size

#             q_var_mean = sum(self.q_var_means) / len(self.q_var_means)
#             if opt.seq_len > 1:
#                 p_var_mean = sum(self.p_var_means) / len(self.p_var_means)
#             else:
#                 p_var_mean = 0
           
#             log_values = (self.step,
#                           self.mean_loss / batches,
#                           self.mean_nll / batches,
#                           self.mean_divergence / batches,
#                           self.mean_prior_div / batches,
#                           self.mean_trans_div / batches,
#                           self.q_var_min, 
#                           q_var_mean,
#                           self.q_var_max,
#                           self.p_var_min, 
#                           p_var_mean,
#                           self.p_var_max,
#                           # mean_grad_norm / batches,
#                           elapsed_seconds / opt.print_every * 1000,
#                           opt.lr)

#             print(("Step: {:8d}, Loss: {:8.3f}, NLL: {:8.3f}, "
#                    "Divergence: {:8.3f}, "
#                    "Prior divergence: {:8.3f}, "
#                    "Trans divergence: {:8.3f}, "
#                    "q(z) vars: [{:7.3f}, {:7.3f}, {:7.3f}], "
#                    "p(z) vars: [{:7.3f}, {:7.3f}, {:7.3f}], "
#                    # "Grad norm: {:10.3f}, "
#                    "ms/seq: {:6.2f}").format(*log_values[:-1]))

#             # make list of n copies of format string, then format
#             format_string = ",".join(["{:.8e}"]*len(log_values))
#             logging.debug(format_string.format(*log_values))
#             self.reset_stats()

#             try:
#                 save_all_generations(i, model, sequence, generations)
#             except:
#                 traceback.print_exc()
#                 pdb.set_trace()

#         if i >= opt.max_steps or (i % opt.save_every == 0 and i > 0):
#             save_dict = {
#                     'model': model,
#                     'opt': vars(opt),
#                     'i': i,
#                     'optimizer': optimizer,
#                     'scheduler': scheduler,
#                 }
#             torch.save(save_dict, opt.save + '/model.t7')

#         if opt.seq_len > 1:
#             if i == opt.max_steps or i - self.last_covariance > opt.cov_every:
#                 construct_covariance(opt.save + '/covariance/', 
#                                      model, train_loader, 2000,
#                                      label="train_" + str(i))
#                 construct_covariance(opt.save + '/covariance/',
#                                      model, test_loader, 2000,
#                                      label="test_" + str(i))
#                 self.last_covariance = i

#     def reset_stats(self):
#         self.mean_loss = 0
#         self.mean_divergence = 0
#         self.mean_nll = 0
#         self.mean_prior_div = 0
#         self.mean_trans_div = 0
#         self.mean_grad_norm = 0

#         self.q_var_means = []
#         self.q_var_min = 100
#         self.q_var_max = -100

#         self.p_var_means = []
#         self.p_var_min = 100
#         self.p_var_max = -100
#         self.progress = progressbar.ProgressBar(max_value=opt.print_every)

class PeriodicTimer:
    def __init__(self, step, frequency, fn):
        self.last_triggered = step
        self.frequency = frequency
        self.fn = fn

    def update(self, step, variables):
        if step >= self.last_triggered + self.frequency:
            self.fn(step, variables)
            self.last_triggered = step

class Bookkeeper:
    def __init__(self, step, variables, update_reducer):
        self.step = step
        self.variables = variables
        self.update_reducer = update_reducer
        self.timers = []

    def update(self, step, update_vars):
        self.step = step
        self.update_reducer(step, self.variables, update_vars)
        for timer in self.timers:
            timer.update(step, self.variables)

    def every(self, n, fn):
        self.timers.append(PeriodicTimer(self.step, n, fn))
