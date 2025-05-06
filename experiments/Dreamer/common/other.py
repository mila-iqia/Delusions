import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

from . import dists

ENABLE_FP16 = False


class RandomAgent:

    def __init__(self, action_space, logprob=False):
        self._logprob = logprob
        if hasattr(action_space, 'n'):
            self._dist = dists.OneHotDist(torch.zeros(action_space.n))
        else:
            dist = tdist.Uniform(action_space.low, action_space.high)
            self._dist = tdist.Independent(dist, 1)

    def __call__(self, obs, state=None, mode=None):
        action = self._dist.sample((len(obs['reset']),))
        output = {'action': action}
        if self._logprob:
            output['logprob'] = self._dist.log_prob(action)
        return output, None


def sequence_scan(fn, state, *inputs, reverse=False):
    # FIXME reverse check, ignore for now
    # state -> (batch, state related)
    # inputs[N] -> (sequence, batch, units)
    # FIXME try to optimize this awful
    # FIXME IDEA FOR JIT BUT REQUIRES REMOVING *inputs
    # static scan also works with tensors....

    # tf.nest.flatten is really good here
    # what a mess
    ## fun fact now jax pytree features would really be usefull here

    indices = range(inputs[0].shape[0])
    select_index = lambda inputs, i: [input[i] for input in inputs]
    last = (state,)
    outs = []
    if reverse:
        indices = reversed(indices)
    for index in indices:
        last = fn(last[0], *select_index(inputs, index))  # zero since we want the state to be a dict
        last = last if isinstance(last, tuple) else (last,)

        outs.append(last)  # outs are envolved in tuples
    if reverse:
        outs = outs[::-1]

    # FIXME this is awfulllllllllll
    # create right structure
    if isinstance(outs[0][0], dict):
        # create dictionary structure
        output = list({} for _ in range(len(outs[0])))  # create lists
        for o in outs:
            for i_d, dictionary in enumerate(o):
                if isinstance(dictionary, dict):  # FIXME
                    for key in dictionary.keys():
                        if key not in output[i_d]:
                            output[i_d][key] = [dictionary[key]]
                        else:
                            output[i_d][key].append(dictionary[key])
                elif isinstance(dictionary, torch.Tensor):
                    # here we append elements to list
                    if not isinstance(output[i_d], list):
                        output[i_d] = []
                    output[i_d].append(dictionary)
                else:
                    raise NotImplementedError(f"sequence scan - creating structure - type {type(dictionary)}")

        # torch stack all entries
        for i_o, dictionary in enumerate(output):
            if isinstance(dictionary, dict):  # FIXME
                for key in dictionary.keys():
                    dictionary[key] = torch.stack(dictionary[key], 0)
            elif isinstance(dictionary, list):
                output[i_o] = torch.stack(dictionary, 0)

            else:
                raise NotImplementedError(f"sequence scan - stacking - type {type(dictionary)}")

    elif isinstance(outs[0][0], torch.Tensor):
        # create tensor structure
        # no output tuple, flatten all in same stack
        # and is very specific to the problem
        # and is awful
        output = []
        for o in outs:
            for tensor in o:  # flatten tuple
                output.append(tensor)

        output = torch.stack(output, 0)

    else:
        raise NotImplementedError(f"sequence scan - Not implemented type {type(outs[0])}")

    return output


def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        # step = tf.cast(step, tf.float32) #Fixme cast
        match = re.match(r'linear\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clamp(step / duration, 0, 1)
            return (1 - mix) * initial + mix * final
        match = re.match(r'warmup\((.+),(.+)\)', string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = torch.clamp(step / warmup, 0, 1)
            return scale * value
        match = re.match(r'exp\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        match = re.match(r'horizon\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clamp(step / duration, 0, 1)
            horizon = (1 - mix) * initial + mix * final
            return 1 - 1 / horizon
        raise NotImplementedError(string)


def lambda_return(reward, value, pcont, bootstrap, lambda_, axis, mask_reject=None): # FIXME needs to be tested
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None: # should be fed with value[-1], the value of the last state of the imagined horizon
        bootstrap = torch.zeros_like(value[-1])
    if mask_reject is not None:
        bootstrap[mask_reject[-1].reshape(bootstrap.shape)] = torch.nan
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    if mask_reject is not None:
        next_values[mask_reject.reshape(next_values.shape)] = torch.nan
    # curr_values = torch.cat([value[[0]], next_values[:-1]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    # if mask_reject is not None and not mask_reject.all():
    #     assert (curr_values[1:][~mask_reject[:-1].reshape(mask_reject.shape[0] - 1, -1)] == next_values[:-1][~mask_reject[:-1].reshape(mask_reject.shape[0] - 1, -1)]).all()
    
    def accumulate(agg, input, pcont, value):
        out = input + pcont * lambda_ * agg
        mask_nan = torch.isnan(out)
        if mask_nan.any():
            out[mask_nan] = value[mask_nan]
        return out
    returns = sequence_scan(accumulate, bootstrap, inputs, pcont, value, reverse=True)
    # NOTE:
    # step -1:  agg = v15
    # step 0:   agg = r14 + gamma * v15
    #               = [1-step return from s14]
    # step 1:   agg = (1 - lambda) * (r13 + gamma * V14) + lambda * (r13 + gamma * r14 + gamma ** 2 * v15)
    #               = (1 - lambda) * [1-step return from s13] + lambda * [2-step return from s13] 
    # step 2:   agg = (1 - lambda) * [1-step return from s12] + (gamma * lambda) * (1 - lambda) * [2-step return from s12] + (gamma * lambda) * lambda * [3-step return from s12]
    # we can mask the untrustworthy parts of "inputs" with the feasibility mask, maybe set them to be NaN
    if mask_reject is not None and mask_reject.any():
        mask_nan_returns = torch.isnan(returns)
        if mask_nan_returns.any():
            assert mask_reject.flatten(1)[mask_nan_returns].all()
        mask_should_be_same_value = mask_reject.flatten(1) & ~mask_nan_returns
        if mask_should_be_same_value.any():
            assert (returns[mask_should_be_same_value] == value[mask_should_be_same_value]).all()
        returns[mask_reject.flatten(1)] = torch.nan
    if axis != 0:
        returns = returns.permute(dims)
    return returns


def action_noise(action, amount, action_space):
    if amount == 0:
        return action
    # amount = tf.cast(amount, action.dtype) # FIXME cast
    if hasattr(action_space, 'n'):
        probs = amount / action.shape[-1] + (1 - amount) * action
        return dists.OneHotDist(probs=probs).sample()
    else:
        return torch.clamp(tdist.Normal(action, amount).sample(), -1, 1)


def pad_dims(tensor, total_dims):
    # Adds dims to end, to match total_dims
    while len(tensor.shape) < total_dims:
        tensor = tensor[..., None]
    return tensor


def dict_to_device(d, device):
    return {k: v.to(device) for k, v in d.items()}


def dict_detach(d):
    return {k: v.detach() for k, v in d.items()}
