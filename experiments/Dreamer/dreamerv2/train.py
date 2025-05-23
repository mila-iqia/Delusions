import collections
import functools
import logging
import os
import pathlib
import sys
import warnings

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tensorboard
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

# this allows common import
sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml
import torch
import random

import agent
import elements
import common

configs = pathlib.Path(sys.argv[0]).parent / 'configs.yaml'
configs = yaml.safe_load(configs.read_text())
config = elements.Config(configs['defaults'])
parsed, remaining = elements.FlagParser(configs=['defaults']).parse_known(
    exit_on_help=False)
for name in parsed.configs:
    config = config.update(configs[name])
config = elements.FlagParser(config).parse(remaining)
logdir = pathlib.Path(config.logdir).expanduser()
config = config.update(
    steps=config.steps // config.action_repeat,
    eval_every=config.eval_every // config.action_repeat,
    log_every=config.log_every // config.action_repeat,
    time_limit=config.time_limit // config.action_repeat,
    prefill=config.prefill // config.action_repeat)

message = 'No GPU found. To actually train on CPU remove this assert.'
assert torch.cuda.is_available(), message  # FIXME

# assert config.precision in (16, 32), config.precision
# assert config.load in ('all', 'wm'), config.precision
# assert config.device in ('cuda', 'cpu'), config.precision

device = config.device

if device != 'cpu':
    torch.set_num_threads(1)


# reproducibility
seed = int(config.seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True  # no apparent impact on speed
torch.backends.cudnn.benchmark = True  # faster, increases memory though.., no impact on seed

# ## seed
# import cv2
#
# cv2.setRNGSeed(seed)
# torch.use_deterministic_algorithms(True) #no difference in performance, or speed


print('Logdir', logdir)
train_replay = common.Replay(logdir / 'train_replay', config.replay_size)
eval_replay = common.Replay(logdir / 'eval_replay', config.time_limit or 1)



step = elements.Counter(train_replay.total_steps)
outputs = [
    elements.TerminalOutput(),
    elements.JSONLOutput(logdir),
    common.TensorBoardOutputPytorch(logdir),
    # common.WandBOutput(config=config, name=str(logdir), project="dreamerv2", entity='esteveste',
    #                    resume=str(logdir).replace('/', '_'))
]

logger = elements.Logger(step, outputs, multiplier=1)
metrics = collections.defaultdict(list)
should_train = elements.Every(config.train_every)
should_log = elements.Every(config.log_every)
should_video_train = elements.Every(config.eval_every)
should_video_eval = elements.Every(5 * config.eval_every)

# save experiment used config
with open(logdir / 'used_config.yaml', 'w') as f:
    f.write('## command line input:\n## ' + ' '.join(sys.argv) + '\n##########\n\n')
    yaml.dump(config, f)


def make_env(mode):
    suite, task = config.task.split('_', 1)
    if suite == 'dmc':
        env = common.DMC(task, config.action_repeat, config.image_size)
        env = common.NormalizeAction(env)
    elif suite == 'atari':
        env = common.Atari(
            task, config.action_repeat, config.image_size, config.grayscale,
            life_done=False, sticky_actions=config.sticky_actions, all_actions=False, seed=seed)
        env = common.OneHotAction(env)
    elif suite == 'retro':
        env = common.Retro(task, config.action_repeat, config.image_size, config.grayscale,
                           life_done=False, sticky_actions=True, all_actions=True, seed=seed)
        env = common.OneHotAction(env)

    elif suite == 'minigrid':
        env = common.Minigrid(task, config.image_size, config.grayscale, seed=seed)
        env = common.OneHotAction(env)

    else:
        raise NotImplementedError(suite)

    if tuple(config.encoder['keys']) == ('flatten',):
        assert tuple(config.decoder['keys']) == ('flatten',), "config: decoder is not flatten"
        env = common.FlattenImageObs(env)

    env = common.TimeLimit(env, config.time_limit)
    env = common.RewardObs(env)
    env = common.ResetObs(env)
    return env


def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    replay_ = dict(train=train_replay, eval=eval_replay)[mode]
    replay_.add(ep)
    logger.scalar(f'{mode}_transitions', replay_.num_transitions)
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    # logger.scalar(f'{mode}_eps', replay_.num_episodes)
    # logger.scalar("memory_usage_kb", int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    should = {'train': should_video_train, 'eval': should_video_eval}[mode]
    if should(step):
        video_log = ep['image']
        if len(ep['image'].shape) == 2: #(flatten)
            shape = (1 if config.grayscale else 3,) + config.image_size #big mess, since logger receiver other format
            video_log = video_log.reshape(-1,*shape).transpose(0, 2, 3, 1)

        logger.video(f'{mode}_policy', video_log)  # B,H,W,C
    logger.write()
    return score, length


print('Create envs.')
train_envs = [make_env('train') for _ in range(config.num_envs)]
eval_envs = [make_env('eval') for _ in range(config.num_envs)]
action_space = train_envs[0].action_space['action']
train_driver = common.Driver(train_envs, device)
train_driver.on_step(lambda _: step.increment())
train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
eval_driver = common.Driver(eval_envs, device)
eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))

prefill = max(0, config.prefill - train_replay.total_steps)
if prefill:
    print(f'Prefill dataset ({prefill} steps).')
    random_agent = common.RandomAgent(action_space)
    train_driver(random_agent, steps=prefill, episodes=1)
    eval_driver(random_agent, episodes=1)
    train_driver.reset()
    eval_driver.reset()

print('Create agent.')

# the agent needs 1. init modules 2. go to device 3. set optimizer
# init agent modules
agnt = agent.Agent(config, logger, action_space, step)

if config.precision == 16:
    common.ENABLE_FP16 = True  # enable fp16 here since only cuda can use fp16
    print("setting fp16")

if config.load =='all': # NOTE: evaluator is injected to the agent thus this would load the evaluator states too
    if (logdir / 'variables.pt').exists():
        print("Load agent")
        agnt.load_state_dict(torch.load(logdir / 'variables.pt'))
elif config.load=='wm':
    if (logdir / 'variables.pt').exists():
        print("Load ONLY WORLD MODEL AGENT")
        # agnt.load_state_dict(torch.load(logdir / 'variables.pt'))

        agnt_state_dict = torch.load(logdir / 'variables.pt', map_location='cpu')

        get_wm_state_dict_keys = lambda m: list(filter(lambda x:'wm' in x,m.keys()))
        get_wm_state_dict = lambda m:{k[3:]:v for k,v in m.items() if k in get_wm_state_dict_keys(m)}

        agnt.wm.load_state_dict(get_wm_state_dict(agnt_state_dict))


agnt = agnt.to(device)

agnt.init_optimizers()

train_dataset = iter(train_replay.dataset(**config.dataset))
eval_dataset = iter(eval_replay.dataset(pin_memory=False, **config.dataset))


def next_batch(iter):
    # casts to fp16 and cuda
    dtype = torch.float16 if common.ENABLE_FP16 else torch.float32  # only on cuda
    out = {k: v.to(device=device, dtype=dtype) for k, v in next(iter).items()}
    return out

agnt.train(next_batch(train_dataset), update_params=False)  # do initial benchmarking pass
torch.cuda.empty_cache()  # clear cudnn bechmarking cache

if not (logdir / 'variables.pt').exists():  # new agent
    config.pretrain and print('Pretrain agent.')
    for _ in range(config.pretrain):
        agnt.train(next_batch(train_dataset))


def train_step(tran):
    flag_record = should_log(step)
    if should_train(step):
        histograms_record_avg = {}
        for step_i in range(config.train_steps):
            _, mets, histograms_record = agnt.train(next_batch(train_dataset))
            # if len(histograms_record):
            #     for name, value in histograms_record.items():
            #         if name not in histograms_record_avg.keys():
            #             histograms_record_avg[name] = [value.reshape(-1)]
            #         else:
            #             histograms_record_avg[name].append(value.reshape(-1))

            [metrics[key].append(value) for key, value in mets.items()]
            # if step_i == config.train_steps - 1:
            #     for name, value in histograms_record_avg.items():
            #         outputs[2]._writer.add_histogram(name, torch.cat(value, -1), step.value)
            #         outputs[2]._writer.flush()
        del histograms_record_avg
    if flag_record:
        for name, values in metrics.items():
            value_mean = np.nanmean(np.array(values, np.float64))
            if not np.isnan(value_mean):
                logger.scalar(name, value_mean)
            metrics[name].clear()
        logger.add(agnt.report(next_batch(train_dataset)), prefix='train')
        logger.write(fps=True)

train_driver.on_step(train_step)

try:
    while step < config.steps:
        logger.write()
        ###
        print('Start evaluation.')
        torch.cuda.empty_cache()
        logger.add(agnt.report(next_batch(eval_dataset)), prefix='eval')
        eval_policy = functools.partial(agnt.policy, mode='eval')
        scores_eval, lengths_eval = eval_driver(eval_policy, episodes=config.eval_eps)

        logger.scalar(f'Performance/scores_eval_mean', np.mean(scores_eval))
        logger.scalar(f'Performance/scores_eval_std', np.std(scores_eval))
        outputs[2]._writer.add_histogram('Performance/scores_eval', scores_eval, step.value)
        outputs[2]._writer.flush()
        del eval_policy
        print(f"[step {step.value:d}]: scores_eval_mean {np.mean(scores_eval):.2g}, scores_eval_std {np.std(scores_eval):.2g}, {scores_eval.shape[0]:d} episodes")
        ###
        print('Start training.')
        torch.cuda.empty_cache()
        scores_train, lengths_train = train_driver(agnt.policy, steps=config.eval_every)
        if scores_train is not None and scores_train.size:
            logger.scalar(f'Performance/scores_train_mean', np.mean(scores_train))
            logger.scalar(f'Performance/scores_train_std', np.std(scores_train))
            # outputs[2]._writer.add_histogram('Performance/scores_train', scores_train, step.value)
            # outputs[2]._writer.flush()
            print(f"[step {step.value:d}]: scores_train_mean {np.mean(scores_train):.2g}, scores_train_std {np.std(scores_train):.2g}, {scores_train.shape[0]:d} episodes")
        torch.save(agnt.state_dict(), logdir / 'variables.pt')
        
except KeyboardInterrupt:
    print("Keyboard Interrupt - saving agent")
    torch.save(agnt.state_dict(), logdir / 'variables.pt')

finally:
    torch.cuda.empty_cache()
    logger.add(agnt.report(next_batch(eval_dataset)), prefix='eval')
    eval_policy = functools.partial(agnt.policy, mode='eval')
    scores_final, lengths_final = eval_driver(eval_policy, episodes=100)
    logger.scalar(f'Performance/scores_final_mean', np.mean(scores_final))
    logger.scalar(f'Performance/scores_final_std', np.std(scores_final))
    outputs[2]._writer.add_histogram('Performance/scores_final', scores_final, step.value)
    outputs[2]._writer.flush()
    print(f"[step {step.value:d}]: scores_final_mean {np.mean(scores_final):.2g}, scores_final_std {np.std(scores_final):.2g}, {scores_final.shape[0]:d} episodes")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

