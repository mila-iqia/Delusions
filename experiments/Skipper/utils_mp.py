import os, json
import time, warnings, datetime, numpy as np

from skipper import create_DQN_Skipper_agent, create_DQN_Skipper_network
from utils import *
from runtime import get_new_env_RDS, get_new_env_SSM, evaluate_agent, save_code_snapshot
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process, Value, Event
from multiprocessing.managers import SyncManager
from cpprb import ReplayBuffer, PrioritizedReplayBuffer
from common.HER import HindsightReplayBuffer
import os, psutil, copy
from tensorboardX import SummaryWriter


from gym.envs.registration import register as gym_register

gym_register(id="RandDistShift-v2", entry_point="RandDistShift:RandDistShift2", reward_threshold=0.95)
gym_register(id="SwordShieldMonster-v2", entry_point="SwordShieldMonster:SwordShieldMonster2", reward_threshold=0.95)


def get_agent(env, args, rb=None, network_policy=None, network_target=None, inference_only=False, silent=False):
    if args.method == "DQN_Skipper":
        agent = create_DQN_Skipper_agent(
            args,
            env=env,
            dim_embed=args.dim_embed,
            num_actions=env.action_space.n,
            device=None,
            hrb=rb,
            network_policy=network_policy,
            network_target=network_target,
            inference_only=inference_only,
            silent=silent,
        )
    else:
        raise NotImplementedError
    return agent


def prepare_experiment(args, config_train):
    if args.game == "RandDistShift":
        env = get_new_env_RDS(args, **config_train)
    elif args.game == "SwordShieldMonster":
        env = get_new_env_SSM(args, **config_train)

    SyncManager.register("SummaryWriter", SummaryWriter)
    SyncManager.register("ReplayBuffer", ReplayBuffer)
    SyncManager.register("PrioritizedReplayBuffer", PrioritizedReplayBuffer)
    SyncManager.register("HindsightReplayBuffer", HindsightReplayBuffer)

    manager = multiprocessing.Manager()

    if torch.cuda.is_available() and not args.force_cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        warnings.warn("global network agent created on cpu")
    if args.method == "DQN_Skipper":
        rb_global = get_cpprb(env, args.size_buffer, prioritized=args.prioritized_replay, hindsight=True, hindsight_strategy=args.hindsight_strategy, num_envs=args.num_envs_train, ctx=manager)
        network_policy_global = create_DQN_Skipper_network(args, env, args.dim_embed, env.action_space.n, device=device, share_memory=True)
    else:
        raise NotImplementedError()

    queue_snapshots = manager.Queue()
    queue_envs_train = manager.Queue(maxsize=16)
    queue_batches_prefetched = multiprocessing.Queue(maxsize=1)

    event_terminate = Event()
    steps_interact, episodes_interact = Value("i", 0), Value("i", 0)
    steps_processed = Value("i", 0)
    signal_explore = Value("b", False)
    signal_pause = Value("b", False)
    path_tf_events = f"tb_records/{env.spec.id}/{args.size_world}x{args.size_world}_traindiff{args.difficulty:g}"
    if args.num_envs_train > 0:
        path_tf_events += f"x{args.num_envs_train:d}"
    path_tf_events += f"/{args.method}/{args.comments}/{args.seed}"
    writer_global = manager.SummaryWriter(path_tf_events)
    save_code_snapshot(path_tf_events)
    with open(os.path.join(path_tf_events, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return (
        network_policy_global,
        rb_global,
        queue_snapshots,
        queue_envs_train,
        queue_batches_prefetched,
        event_terminate,
        steps_interact,
        steps_processed,
        episodes_interact,
        signal_explore,
        signal_pause,
        writer_global,
        path_tf_events,
    )


def prefetcher_batch(queue_batches_prefetched, rb_global, steps_processed, args, event_terminate):
    if args.prioritized_replay:
        schedule_beta_sample_priorities = LinearSchedule(args.steps_max, initial_p=0.4, final_p=1.0)
    while True:
        flag_q_full = queue_batches_prefetched.full()
        if flag_q_full or rb_global.get_stored_size() < args.size_batch:
            if event_terminate.is_set():
                break
            else:
                time.sleep(0.000001)
        else:
            if args.prioritized_replay:
                batch_preload_unprocessed = rb_global.sample(
                    args.size_batch,
                    beta=schedule_beta_sample_priorities.value(steps_processed.value),
                )
            else:
                batch_preload_unprocessed = rb_global.sample(args.size_batch)
            batch_preloaded = process_batch(batch_preload_unprocessed, prioritized=args.prioritized_replay, with_targ=True)
            batch_preload = []
            for item in batch_preloaded:
                if isinstance(item, torch.Tensor):
                    batch_preload.append(item.share_memory_().cuda(non_blocking=True))
                else:
                    batch_preload.append(item)
            queue_batches_prefetched.put(batch_preload)


def generator_env(queue_envs_train, config_train, args):
    if args.num_envs_train > 0:
        envs_train = []
        for idx_env in range(args.num_envs_train):
            if args.game == "RandDistShift":
                env = get_new_env_RDS(args, **config_train)
            elif args.game == "SwordShieldMonster":
                env = get_new_env_SSM(args, **config_train)
            env.idx_env = idx_env
            env.reset()
            env.generate_oracle()
            envs_train.append(env)
    while True:
        flag_q_train_full = queue_envs_train.full()
        if flag_q_train_full:
            time.sleep(0.00001)
        else:
            if args.num_envs_train > 0:
                idx_env = np.random.randint(args.num_envs_train)
                env_train = envs_train[idx_env]
                env_train.reset()
                env_train = copy.deepcopy(env_train)
                queue_envs_train.put(env_train)
            else:
                if args.game == "RandDistShift":
                    env_train = get_new_env_RDS(args, **config_train)
                elif args.game == "SwordShieldMonster":
                    env_train = get_new_env_SSM(args, **config_train)
                env_train.reset()
                queue_envs_train.put(env_train)


@torch.no_grad()
def explorer(
    network_policy_global, rb_global, queue_envs_train, steps_interact, episodes_interact, event_terminate, signal_explore, signal_pause, args, config_train, writer
):
    if args.num_envs_train > 0:
        env = None
        while env is None:
            try:
                env = queue_envs_train.get()
            except:
                time.sleep(0.00001)
    else:
        if args.game == "RandDistShift":
            env = get_new_env_RDS(args, **config_train)
        elif args.game == "SwordShieldMonster":
            env = get_new_env_SSM(args, **config_train)
    env.reset()
    local_hrb = get_cpprb(
        env,
        env.unwrapped.max_steps,
        prioritized=args.prioritized_replay,
        hindsight=True,
        hindsight_strategy=args.hindsight_strategy,
        num_envs=args.num_envs_train,
        rb_pertask_sample=rb_global,
    )
    agent = get_agent(env, args, rb=local_hrb, network_policy=network_policy_global, inference_only=True, silent=True)
    size_submit = 1
    if "minigrid" in args.game.lower() or "distshift" in args.game.lower() or "swordshieldmonster" in args.game.lower():
        type_env = "minigrid"
    else:
        raise NotImplementedError()
    flag_newenvs = "distshift" in args.game.lower()
    print("[EXPLORER] env generation pipeline enabled")
    steps_collected, episodes_collected = 0, 0
    while not event_terminate.is_set():
        return_episode, return_episode_discounted, steps_episode = 0, 0, 0
        obs_curr, done, real_done, flag_reset = env.obs_curr, False, False, False
        steps_interact_curr, episodes_interact_curr = steps_interact.value, episodes_interact.value
        agent.steps_interact = steps_interact_curr
        while not flag_reset:
            if not signal_explore.value or signal_pause.value:
                if writer is not None:
                    writer.flush()
            while not signal_explore.value or signal_pause.value:
                if event_terminate.is_set():
                    return
                else:
                    time.sleep(0.00001)

            epsilon = agent.schedule_epsilon.value(steps_interact.value)
            with torch.autocast("cuda", enabled=False):
                action = agent.decide(
                    obs_curr, epsilon=epsilon, eval=False, env=env, writer=writer, random_walk=args.random_walk, step_record=steps_interact.value
                )
            obs_next, reward, done, info = env.step(action)
            steps_episode += 1
            if type_env == "minigrid":
                real_done = done and not info["overtime"]
            else:
                real_done = done
            if event_terminate.is_set():
                return
            agent.step(
                obs_curr=obs_curr, action=action, reward=reward, done=real_done, obs_next=obs_next, idx_env=env.idx_env, add_to_buffer=True, increment_steps=True
            )
            agent.steps_processed = agent.steps_interact
            steps_collected += 1
            return_episode += reward
            return_episode_discounted += reward * agent.gamma**env.step_count
            obs_curr = obs_next
            flag_reset = real_done or (done and type_env == "minigrid")
        if writer is not None:
            str_info = (
                f"[EXPLORER] seed: {args.seed}, steps_interact: {steps_interact_curr}, episode: {episodes_interact_curr}, "
                f"epsilon: {epsilon: .2f}, return: {return_episode: g}, return_discount: {return_episode_discounted: g}, "
                f"steps_episode: {steps_episode}"
            )
            writer.add_text("Text/info_train", str_info, steps_interact_curr)
        len_trajectory = agent.hrb.episode_rb.get_stored_size()
        num_planning_triggered = int(agent.num_planning_triggered)
        num_planning_triggered_timeout = int(agent.num_planning_triggered_timeout)
        num_waypoints_reached = int(agent.num_waypoints_reached)
        if agent.waypoints_existing is not None:
            num_waypoints_selected = int(agent.proxy_graph_curr["selected"].sum())

        else:
            num_waypoints_selected = None
        agent.on_episode_end(eval=False)
        episodes_collected += 1
        if agent.hrb.get_stored_size() >= size_submit:
            submitted = True
            samples_local = agent.hrb.get_all_transitions()
            agent.hrb.clear()
            size_submitted = samples_local["rew"].shape[0]
            if args.prioritized_replay:
                (
                    batch_obs_curr,
                    batch_action,
                    batch_reward,
                    batch_obs_next,
                    batch_done,
                    batch_obs_targ,
                    batch_obs_targ2,
                    _,
                    _,
                ) = process_batch(
                    samples_local,
                    prioritized=False,
                    with_targ=True,
                    device=agent.device,
                    obs2tensor=minigridobs2tensor,
                    clip_reward=agent.clip_reward)
                with torch.autocast("cuda", enabled=False):
                    ret = agent.calculate_multihead_error(batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_obs_targ2)
                new_priorities = ret[0].detach().cpu().numpy()
                rb_global.rb.add(**samples_local, priorities=new_priorities)
                del ret
            else:
                rb_global.rb.add(**samples_local)
            with episodes_interact.get_lock():
                episodes_interact.value += episodes_collected
            episodes_collected = 0
            with steps_interact.get_lock():
                steps_interact.value += steps_collected
            steps_collected = 0
        else:
            submitted = False
        debug = writer is not None and np.random.rand() < 0.05
        if debug:
            if submitted:
                writer.add_scalar("Other/size_submitted", size_submitted, steps_interact_curr)
            steps_interact_curr, episodes_interact_curr = steps_interact.value, episodes_interact.value
            writer.add_scalar("Experience/len_trajectory", len_trajectory, steps_interact_curr)
            writer.add_scalar("Experience/num_planning_triggered", num_planning_triggered, steps_interact_curr)
            writer.add_scalar("Experience/num_planning_triggered_timeout", num_planning_triggered_timeout, steps_interact_curr)
            writer.add_scalar("Experience/num_waypoints_reached", num_waypoints_reached, steps_interact_curr)
            if num_waypoints_selected is not None:
                writer.add_scalar("Experience/num_waypoints_selected", num_waypoints_selected, steps_interact_curr)
            writer.add_scalar("Experience/return", return_episode, steps_interact_curr)
            writer.add_scalar("Experience/return_discount", return_episode_discounted, steps_interact_curr)
            writer.add_scalar("Experience/episodes", episodes_interact_curr, steps_interact_curr)
            if args.game == "RandDistShift":
                writer.add_scalar("Experience/dist2init", info["dist2init"], steps_interact_curr)
                writer.add_scalar("Experience/dist2goal", info["dist2goal"], steps_interact_curr)
                writer.add_scalar("Experience/dist2init_x", np.abs(info["agent_pos"][0] - info["agent_pos_init"][0]), steps_interact_curr)
            elif args.game == "SwordShieldMonster":
                writer.add_scalar("Experience/sword_acquired", float(info["sword_acquired"]), steps_interact_curr)
                writer.add_scalar("Experience/shield_acquired", float(info["shield_acquired"]), steps_interact_curr)
            writer.add_scalar("Experience/overtime", float(info["overtime"]), steps_interact_curr)
            writer.add_scalar("Experience/dead", float(done and not float(return_episode) and not info["overtime"]), steps_interact_curr)
        if event_terminate.is_set():
            return
        if flag_newenvs or args.num_envs_train > 0:
            env_preloaded = True
            if args.num_envs_train > 0:
                del env
                env = None
                while env is None:
                    try:
                        env = queue_envs_train.get()
                    except:
                        time.sleep(0.00001)
                        env_preloaded = False
            else:
                del env
                try:
                    env = queue_envs_train.get()
                except:
                    if args.game == "RandDistShift":
                        env = get_new_env_RDS(args, **config_train)
                    elif args.game == "SwordShieldMonster":
                        env = get_new_env_SSM(args, **config_train)
                    env.reset()
                    env_preloaded = False
            if debug:
                writer.add_scalar("Other/env_preloaded", float(env_preloaded), steps_interact_curr)


def learner(
    network_policy_global,
    rb_global,
    queue_snapshots,
    steps_interact,
    steps_processed,
    episodes_interact,
    event_terminate,
    signal_explore,
    signal_pause,
    args,
    pid_main,
    config_train,
    queue_batches_prefetched,
    writer,
):
    process_main = psutil.Process(pid_main)
    process_learner = psutil.Process(os.getpid())
    if args.game == "RandDistShift":
        env = get_new_env_RDS(args, **config_train)
    elif args.game == "SwordShieldMonster":
        env = get_new_env_SSM(args, **config_train)
    agent = get_agent(env, args, rb=rb_global, network_policy=network_policy_global, network_target=None)
    step_last_eval, time_last_disp = -args.freq_eval, time.time()
    print("[LEARNER] loop enter")
    agent.steps_interact = steps_interact.value
    steps_processed_last_disp, episode_last_disp, time_last_disp = 0, 0, time.time()
    while True:
        if signal_pause.value:
            signal_explore.value = False
            time.sleep(0.00001)
            continue
        flag_need_update = agent.need_update()
        if flag_need_update:
            with signal_explore.get_lock():
                signal_explore.value = False
        episodes_interact_curr = episodes_interact.value
        agent.steps_interact = steps_interact.value
        if agent.steps_processed - step_last_eval >= args.freq_eval:
            weights = agent.network_policy.state_dict()
            for key in weights.keys():
                weights[key] = weights[key].cpu()
            snapshot_shared = {"weights": weights, "steps_processed": int(agent.steps_processed)}
            queue_snapshots.put(snapshot_shared)
            step_last_eval += args.freq_eval
        if agent.steps_processed >= min(args.steps_stop, args.steps_max) or episodes_interact_curr >= args.episodes_max:
            event_terminate.set()
            break
        if flag_need_update:
            with signal_explore.get_lock():
                signal_explore.value = False
            if queue_batches_prefetched.empty():
                batch_preload = None
            else:
                batch_preload = queue_batches_prefetched.get()
            agent.update_step(batch_processed=batch_preload, writer=writer)
            with steps_processed.get_lock():
                steps_processed.value = agent.steps_processed
            if writer is not None and np.random.rand() < 0.01:
                writer.add_scalar("Other/batch_preloaded", float(batch_preload is not None), agent.steps_processed)
            del batch_preload
        else:
            if signal_explore.value:
                time.sleep(0.00001)
            else:
                with signal_explore.get_lock():
                    signal_explore.value = True
                if episodes_interact_curr - episode_last_disp > 0:
                    time_from_last_disp = time.time() - time_last_disp
                    try:
                        mem = process_main.memory_info().rss / 1073741824
                    except:
                        mem = None
                    if time_from_last_disp > 0:
                        sps = (agent.steps_processed - steps_processed_last_disp) / time_from_last_disp
                        if sps > 0:
                            if mem is not None:
                                try:
                                    mem_learner = 0
                                    for process_child in process_main.children(recursive=True):
                                        if process_child.pid == process_learner.pid:
                                            mem_learner = process_child.memory_info().rss / 1073741824
                                        mem += process_child.memory_info().rss / 1073741824
                                except:
                                    mem = None
                            eta = str(datetime.timedelta(seconds=int((args.steps_stop - agent.steps_processed) / sps)))
                            if steps_processed_last_disp:
                                writer.add_scalar("Other/sps", sps, agent.steps_interact)
                            if mem is not None:
                                print(
                                    "[%d] episode_explored: %d, steps_explored: %d, steps_processed: %d, size_rb: %d, eps: %.2f, mem: %.2f(%.2f)GB, sps: %.2f, eta: %s"
                                    % (
                                        args.seed,
                                        episodes_interact_curr,
                                        steps_interact.value,
                                        agent.steps_processed,
                                        rb_global.rb.get_stored_size(),
                                        agent.schedule_epsilon.value(agent.steps_processed),
                                        mem,
                                        mem_learner,
                                        sps,
                                        eta,
                                    )
                                )
                    if steps_processed_last_disp and mem is not None:
                        writer.add_scalar("Other/RAM", mem, agent.steps_processed)
                    steps_processed_last_disp, episode_last_disp, time_last_disp = agent.steps_processed, episodes_interact_curr, time.time()
                writer.flush()
    if not queue_snapshots.empty():
        print("[LEARNER] waiting for evaluator to finish")
    while not queue_snapshots.empty():
        time.sleep(30)
    print("[LEARNER] finished with empty queue_snapshots")
    queue_batches_prefetched.close()


@torch.no_grad()
def evaluator(config_train, configs_eval, event_terminate, queue, queue_envs_train, steps_interact, signal_pause, args, writer, path_save=None):
    num_episodes = 20
    args = copy.copy(args)
    if args.game == "RandDistShift":
        env_train_generator = lambda: get_new_env_RDS(args, **config_train)
    elif args.game == "SwordShieldMonster":
        env_train_generator = lambda: get_new_env_SSM(args, **config_train)
    env = env_train_generator()
    args.waypoint_strategy = "regenerate_whole_graph"
    agent = get_agent(env, args, rb=None, inference_only=True, silent=True)
    agent.network_policy.eval()
    for module in agent.network_policy.modules():
        module.eval()
    print("[EVALUATOR] agent.device:")
    print(agent.device)

    while True:
        if queue.empty():
            if event_terminate.is_set():
                break
            else:
                if signal_pause.value:
                    signal_pause.value = False
                    print(f"[EVALUATOR] cancelled signal_pause, setting to False")
                time.sleep(0.1)
        else:
            while not queue.empty():
                if event_terminate.is_set():
                    print(f"[EVALUATOR] event_terminate is set but evaluator hasn't finished the jobs yet")
                snapshot_shared = queue.get()
                steps_processed = copy.copy(int(snapshot_shared["steps_processed"]))
                agent.network_policy.load_state_dict(snapshot_shared["weights"])
                del snapshot_shared
                print(f"[EVALUATOR] package received for step {steps_processed:d}")
                agent.steps_interact = steps_processed
                agent.steps_processed = steps_processed
                if steps_interact.value >= args.freq_eval + steps_processed:
                    signal_pause.value = True
                    print(f"[EVALUATOR] announced signal_pause, setting to True")
                if args.method == "DQN_Skipper":
                    evaluate_multihead_minigrid(
                        env,
                        agent,
                        writer,
                        size_batch=32,
                        num_episodes=5,
                        suffix="",
                        step_record=None,
                        env_generator=lambda: get_new_env_RDS(args, **config_train) if args.game == "RandDistShift" else get_new_env_SSM(args, **config_train),
                        queue_envs=None,
                    )
                    writer.flush()
                (
                    returns_mean,
                    returns_std,
                    returns_discounted_mean,
                    returns_discounted_std,
                ) = evaluate_agent(env_train_generator, agent, num_episodes=num_episodes, type_env="minigrid", queue_envs=queue_envs_train)
                print(
                    f"Eval/trainx{num_episodes} @ step {agent.steps_processed:d} - returns_mean: {returns_mean:.2f}, returns_std: {returns_std:.2f}, returns_discounted_mean: {returns_discounted_mean:.2f}, returns_discounted_std: {returns_discounted_std:.2f}"
                )
                writer.add_scalar("Eval/train", returns_mean, agent.steps_processed)
                writer.add_scalar("Eval/train_discount", returns_discounted_mean, agent.steps_processed)
                for config_eval in configs_eval:
                    env_generator = lambda: get_new_env_RDS(args, **config_eval) if args.game == "RandDistShift" else get_new_env_SSM(args, **config_eval)
                    (
                        returns_mean,
                        returns_std,
                        returns_discounted_mean,
                        returns_discounted_std,
                    ) = evaluate_agent(env_generator, agent, num_episodes=num_episodes, type_env="minigrid")
                    diff = np.mean(config_eval["lava_density_range"])
                    print(
                        f"Eval/{diff:g} x{num_episodes} @ step {agent.steps_processed:d} - returns_mean: {returns_mean:.2f}, returns_std: {returns_std:.2f}, returns_discounted_mean: {returns_discounted_mean:.2f}, returns_discounted_std: {returns_discounted_std:.2f}"
                    )
                    writer.add_scalar(f"Eval/{diff:g}", returns_mean, agent.steps_processed)
                    writer.add_scalar(f"Eval/discount_{diff:g}", returns_discounted_mean, agent.steps_processed)
                    writer.flush()
                if path_save is not None and steps_processed >= args.steps_stop:
                    agent.save2disk(path=os.path.join(path_save, f"snapshots_params_eval/{steps_processed:d}/"))
                    print(f"saved the snapshot at step {steps_processed:d} to disk")
                writer.flush()
    print("[EVALUATOR] finished with empty queue_snapshots")


def run_multiprocess(args, config_train, configs_eval):
    pid_main = os.getpid()
    (
        network_policy_global,
        rb_global,
        queue_snapshots,
        queue_envs_train,
        queue_batches_prefetched,
        event_terminate,
        steps_interact,
        steps_processed,
        episodes_interact,
        signal_explore,
        signal_pause,
        writer,
        path_writer,
    ) = prepare_experiment(args, config_train)
    tasks = []

    task_generator_env = Process(name="generator_env", target=generator_env, args=[queue_envs_train, config_train, args])
    task_generator_env.start()

    task = Process(
        name="explorer_0",
        target=explorer,
        args=[
            network_policy_global,
            rb_global,
            queue_envs_train,
            steps_interact,
            episodes_interact,
            event_terminate,
            signal_explore,
            signal_pause,
            args,
            config_train,
            writer,
        ],
    )
    task.start()
    tasks.append(task)

    task = Process(
        name="evaluator",
        target=evaluator,
        args=[config_train, configs_eval, event_terminate, queue_snapshots, queue_envs_train, steps_interact, signal_pause, args, writer, path_writer],
    )
    task.start()
    tasks.append(task)

    task = Process(
        name="learner",
        target=learner,
        args=[
            network_policy_global,
            rb_global,
            queue_snapshots,
            steps_interact,
            steps_processed,
            episodes_interact,
            event_terminate,
            signal_explore,
            signal_pause,
            args,
            pid_main,
            config_train,
            queue_batches_prefetched,
            writer,
        ],
    )
    task.start()
    tasks.append(task)

    args_otherexplorers = copy.deepcopy(args)
    for i in range(1, args.num_explorers):
        task = Process(
            name=f"explorer_{i:g}",
            target=explorer,
            args=[
                network_policy_global,
                rb_global,
                queue_envs_train,
                steps_interact,
                episodes_interact,
                event_terminate,
                signal_explore,
                signal_pause,
                args_otherexplorers,
                config_train,
                None,
            ],
        )
        task.start()
        tasks.append(task)

    task_prefetcher = Process(name="prefetcher", target=prefetcher_batch, args=[queue_batches_prefetched, rb_global, steps_processed, args, event_terminate])
    task_prefetcher.start()

    finished = np.zeros(len(tasks), dtype=bool)
    while not finished.all():
        for idx_task in range(len(tasks)):
            task = tasks[idx_task]
            if not task.is_alive():
                if task.exitcode == 0:
                    finished[idx_task] = True
                    print(f"[utils_mp] {task.name} RIP'ed")
                else:
                    raise RuntimeError(f"[utils_mp] {task.name} exited with code {task.exitcode}")
        time.sleep(15)

    del (
        network_policy_global,
        rb_global,
        queue_snapshots,
        queue_envs_train,
        queue_batches_prefetched,
        event_terminate,
        steps_interact,
        episodes_interact,
        signal_explore,
        writer,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
