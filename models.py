import os, sys

sys.path.insert(1, os.getcwd())
import torch, numpy as np
from utils import init_weights, HistogramConverter, cyclical_schedule
from modules import ResidualBlock, TopKMultiheadAttention
from SwordShieldMonster import x2swordshield

class Encoder_MiniGrid_RDS(torch.nn.Module):
    def __init__(self):
        super(Encoder_MiniGrid_RDS, self).__init__()
        from minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

        self.object_to_idx = OBJECT_TO_IDX
        self.color_to_idx = COLOR_TO_IDX

    @torch.no_grad()
    def forward(self, obs):
        if len(obs.shape) == 3:
            obs = obs[None, :, :, :]
        size_batch = obs.shape[0]
        mask_agent = obs[:, :, :, 0] == self.object_to_idx["agent"]
        colors = obs[mask_agent][:, 1]
        mask_on_lava = colors == self.color_to_idx["yellow"]
        mask_on_goal = colors == self.color_to_idx["green"]
        mask_agent_on_empty = (~mask_on_lava & ~mask_on_goal).reshape(size_batch, 1, 1) & mask_agent
        mask_agent_on_lava = mask_on_lava.reshape(size_batch, 1, 1) & mask_agent
        mask_agent_on_goal = mask_on_goal.reshape(size_batch, 1, 1) & mask_agent
        layout = obs[:, :, :, [0]]
        layout[mask_agent_on_empty] = self.object_to_idx["empty"]
        layout[mask_agent_on_lava] = self.object_to_idx["lava"]
        layout[mask_agent_on_goal] = self.object_to_idx["goal"]
        if layout.dtype != torch.uint8:
            layout = layout.to(torch.uint8)
        return layout, mask_agent


class Decoder_MiniGrid_RDS(torch.nn.Module):
    def __init__(self):
        super(Decoder_MiniGrid_RDS, self).__init__()
        from minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

        self.object_to_idx, self.color_to_idx = OBJECT_TO_IDX, COLOR_TO_IDX

    @torch.no_grad()
    def forward(self, layout, mask_agent):
        size_batch = mask_agent.shape[0]
        if layout.shape[0] == 1:
            layout = layout.repeat(size_batch, 1, 1, 1)
        mask_agent = mask_agent.bool()
        obs = torch.cat([layout, torch.zeros_like(layout)], dim=-1)
        colors = torch.full([size_batch], self.color_to_idx["red"], device=obs.device, dtype=obs.dtype)
        mask_lava = obs[:, :, :, 0] == self.object_to_idx["lava"]
        # NOTE(H): in RDS, the lavas are not by default printed, only the agent changes color when doing shit
        mask_on_lava = torch.logical_and(mask_agent.reshape(size_batch, -1), mask_lava.reshape(size_batch, -1)).any(-1)
        colors[mask_on_lava] = self.color_to_idx["yellow"]
        mask_goal = obs[:, :, :, 0] == self.object_to_idx["goal"]
        mask_on_goal = torch.logical_and(mask_agent.reshape(size_batch, -1), mask_goal.reshape(size_batch, -1)).any(-1)
        colors[mask_on_goal] = self.color_to_idx["green"]
        mask_agent = mask_agent.reshape(size_batch, layout.shape[-3], layout.shape[-2])

        obs[mask_agent] = torch.stack([torch.full([size_batch], self.object_to_idx["agent"], device=obs.device, dtype=obs.dtype), colors], dim=-1)
        assert obs.dtype == torch.uint8
        return obs


class Encoder_MiniGrid_SSM(torch.nn.Module):
    def __init__(self):
        super(Encoder_MiniGrid_SSM, self).__init__()
        from minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

        self.object_to_idx = OBJECT_TO_IDX
        self.color_to_idx = COLOR_TO_IDX

    @torch.no_grad()
    def forward(self, obs):
        if len(obs.shape) == 3:
            obs = obs[None, :, :, :]
        size_batch = obs.shape[0]
        mask_lava = obs[:, :, :, 1] == self.color_to_idx["yellow"]
        mask_sword = obs[:, :, :, 1] == self.color_to_idx["purple"]
        mask_shield = obs[:, :, :, 1] == self.color_to_idx["blue"]
        mask_monster = obs[:, :, :, 1] == self.color_to_idx["green"]
        assert mask_sword.sum() == size_batch
        assert mask_shield.sum() == size_batch
        assert mask_monster.sum() == size_batch
        
        mask_agent = obs[:, :, :, 0] == self.object_to_idx["agent"]
        x_agent = torch.zeros(size_batch, device=obs.device, dtype=torch.long)
        slice_obs_obj = obs[:, :, :, 0]
        x_agent[(slice_obs_obj[mask_sword] == self.object_to_idx["sword"])] += 2
        x_agent[(slice_obs_obj[mask_shield] == self.object_to_idx["shield"])] += 1

        layout = obs[:, :, :, [0]]
        layout[mask_agent] = self.object_to_idx["empty"]
        layout[mask_lava] = self.object_to_idx["lava"]
        layout[mask_sword] = self.object_to_idx["sword"]
        layout[mask_shield] = self.object_to_idx["shield"]
        layout[mask_monster] = self.object_to_idx["monster"]

        mask_agent_out = torch.zeros(size_batch, layout.shape[1], layout.shape[2], 4, device=obs.device, dtype=torch.bool)
        for x_curr in torch.arange(4, device=obs.device):
            mask_x_curr = x_agent == x_curr
            mask_agent_out[:, :, :, x_curr] = mask_x_curr[:, None, None] & mask_agent
        mask_agent_out = mask_agent_out.reshape(size_batch, -1)
        if layout.dtype != torch.uint8:
            layout = layout.to(torch.uint8)
        return layout, mask_agent_out


class Decoder_MiniGrid_SSM(torch.nn.Module):
    def __init__(self):
        super(Decoder_MiniGrid_SSM, self).__init__()
        from minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

        self.object_to_idx, self.color_to_idx = OBJECT_TO_IDX, COLOR_TO_IDX

    @torch.no_grad()
    def forward(self, layout, mask_agent_ijx):
        size_batch = mask_agent_ijx.shape[0]
        mask_agent_ijx = mask_agent_ijx.reshape(size_batch, layout.shape[1], layout.shape[2], 4)
        mask_agent = mask_agent_ijx.any(-1)
        x_agent = mask_agent_ijx.sum([1,  2]).argmax(-1)

        obs = torch.cat([layout, torch.zeros_like(layout)], dim=-1)
        mask_lava = obs[:, :, :, 0] == self.object_to_idx["lava"]
        mask_sword = obs[:, :, :, 0] == self.object_to_idx["sword"]
        mask_shield = obs[:, :, :, 0] == self.object_to_idx["shield"]
        mask_monster = obs[:, :, :, 0] == self.object_to_idx["monster"]
        obs[:, :, :, 1][mask_lava] = self.color_to_idx["yellow"]
        obs[:, :, :, 1][mask_sword] = self.color_to_idx["purple"]
        obs[:, :, :, 1][mask_shield] = self.color_to_idx["blue"]
        obs[:, :, :, 1][mask_monster] = self.color_to_idx["green"]
        
        if layout.shape[0] == 1 and size_batch > 1:
            obs = obs.repeat(size_batch, 1, 1, 1)
            mask_lava = mask_lava.repeat(size_batch, 1, 1)
            mask_sword = mask_sword.repeat(size_batch, 1, 1)
            mask_shield = mask_shield.repeat(size_batch, 1, 1)
            mask_monster = mask_monster.repeat(size_batch, 1, 1)

        for x_curr in range(4):
            sword_acquired, shield_acquired = x2swordshield(x_curr)
            mask_x_curr = x_agent == x_curr
            if mask_x_curr.any() and (not sword_acquired or not shield_acquired):
                mask_sword_x_curr = mask_sword.clone()
                mask_sword_x_curr[~mask_x_curr] = False
                mask_shield_x_curr = mask_shield.clone()
                mask_shield_x_curr[~mask_x_curr] = False
                if not sword_acquired:
                    obs[mask_sword_x_curr, [0]] = self.object_to_idx["empty"]
                if not shield_acquired:
                    obs[mask_shield_x_curr, [0]] = self.object_to_idx["empty"]
        colors_agent = torch.full([size_batch], self.color_to_idx["red"], device=obs.device, dtype=obs.dtype)

        mask_on_sword = torch.logical_and(mask_agent.reshape(size_batch, -1), mask_sword.reshape(size_batch, -1)).any(-1)
        colors_agent[mask_on_sword] = self.color_to_idx["purple"]

        mask_on_shield = torch.logical_and(mask_agent.reshape(size_batch, -1), mask_shield.reshape(size_batch, -1)).any(-1)
        colors_agent[mask_on_shield] = self.color_to_idx["blue"]

        mask_on_lava = torch.logical_and(mask_agent.reshape(size_batch, -1), mask_lava.reshape(size_batch, -1)).any(-1)
        colors_agent[mask_on_lava] = self.color_to_idx["yellow"]

        mask_on_monster = torch.logical_and(mask_agent.reshape(size_batch, -1), mask_monster.reshape(size_batch, -1)).any(-1)
        colors_agent[mask_on_monster] = self.color_to_idx["green"]

        mask_agent = mask_agent.reshape(size_batch, obs.shape[1], obs.shape[2])

        obs[mask_agent] = torch.stack([torch.full([size_batch], self.object_to_idx["agent"], device=obs.device, dtype=obs.dtype), colors_agent], dim=-1)
        
        assert obs.dtype == torch.uint8
        return obs

class CVAE_MiniGrid(torch.nn.Module):
    def __init__(
        self,
        layout_extractor,
        decoder,
        sample_input,
        num_categoricals=6,
        num_categories=2,
        beta=0.00025,
        maximize_entropy=True,
        alpha_KL_balance=0.8,
        activation=torch.nn.ReLU,
        interval_beta=10000,
        batchnorm=False,
        argmax_latents=True,
        argmax_reconstruction=False,
        num_classes_abstract=4,
        prob_random_generation=0.2,
        **kwargs,
    ):
        super(CVAE_MiniGrid, self).__init__(**kwargs)
        self.argmax_latents = bool(argmax_latents)
        self.argmax_reconstruction = bool(argmax_reconstruction)

        self.num_classes_abstract = num_classes_abstract
        self.num_categoricals, self.num_categories = num_categoricals, num_categories
        self.len_code = num_categoricals * num_categories
        self.decoder = decoder
        self.layout_extractor = layout_extractor
        self.prob_random_generation = prob_random_generation

        self.maximize_entropy = maximize_entropy
        self.beta, self.alpha_KL_balance = beta, alpha_KL_balance
        from minigrid import OBJECT_TO_IDX

        self.object_to_idx = OBJECT_TO_IDX
        self.interval_beta = interval_beta
        self.steps_trained = 0
        self.size_input = sample_input.shape[-2]

        self.encoder_context = Embedder_MiniGrid_BOW(
            dim_embed=16, width=sample_input.shape[-3], height=sample_input.shape[-2], channels_obs=sample_input.shape[-1], ebd_pos=False
        )
        self.encoder_obs = Embedder_MiniGrid_BOW(
            dim_embed=16, width=sample_input.shape[-3], height=sample_input.shape[-2], channels_obs=sample_input.shape[-1], ebd_pos=False
        )

        self.compressor = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=self.size_input, stride=1, padding=0),
            torch.nn.Flatten(),
            torch.nn.Linear(32, self.len_code),
        )

        self.decompressor = torch.nn.Sequential(
            torch.nn.Unflatten(1, (self.len_code, 1, 1)),
            torch.nn.ConvTranspose2d(self.len_code, 16, kernel_size=self.size_input, stride=1, padding=0),
        )

        self.fuser = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * self.size_input ** 2, self.num_classes_abstract * self.size_input ** 2),
        )

        tensors_to_mesh = []
        for idx_category in range(self.num_categoricals):
            tensors_to_mesh.append(torch.arange(self.num_categories))
        indices_mesh = torch.meshgrid(*tensors_to_mesh)
        indices_mesh = torch.concatenate([indices.reshape(1, -1) for indices in indices_mesh], 0).permute(1, 0).contiguous()
        self.samples_uniform = torch.nn.functional.one_hot(indices_mesh, num_classes=self.num_categories).float()
        self.minus_log_uniform = float(np.log(self.num_categories))

    def compress_from_obs(self, obs):
        obs_encoded = self.encoder_obs(obs)
        logits_code = self.compressor(obs_encoded).reshape(-1, self.num_categoricals, self.num_categories)
        return logits_code

    # @profile
    def logits_from_code(self, samples, context):
        size_batch = samples.shape[0]
        assert context.shape[0] == size_batch
        samples_decompressed = self.decompressor(samples.float().reshape(size_batch, -1))
        logits_mask_pos_agent = self.fuser(torch.cat([samples_decompressed, context], 1))
        return logits_mask_pos_agent

    def to(self, device):
        super().to(device)
        self.encoder_context.to(device)
        self.encoder_obs.to(device)
        self.compressor.to(device)
        self.decompressor.to(device)
        self.fuser.to(device)

    def parameters(self):
        parameters = []
        parameters += list(self.encoder_context.parameters())
        parameters += list(self.encoder_obs.parameters())
        parameters += list(self.compressor.parameters())
        parameters += list(self.decompressor.parameters())
        parameters += list(self.fuser.parameters())
        return parameters

    @torch.no_grad()
    # @profile
    def mask_from_logits(self, logits_mask_pos_agent, noargmax=False):
        size_batch = logits_mask_pos_agent.shape[0]
        logits_mask_pos_agent = logits_mask_pos_agent.reshape(size_batch, -1)
        assert logits_mask_pos_agent.shape[-1] == self.num_classes_abstract * self.size_input ** 2
        if np.random.rand() < self.prob_random_generation:
            probs_mask_pos_agent = torch.ones_like(logits_mask_pos_agent) / torch.numel(logits_mask_pos_agent)
            mask_agent_pred = torch.distributions.OneHotCategorical(probs=probs_mask_pos_agent).sample()
        else:
            if self.argmax_reconstruction and not noargmax:
                mask_agent_pred = torch.nn.functional.one_hot(logits_mask_pos_agent.argmax(-1), num_classes=self.num_classes_abstract * self.size_input ** 2)
            else:
                mask_agent_pred = torch.distributions.OneHotCategorical(logits=logits_mask_pos_agent).sample()
        mask_agent_pred = mask_agent_pred.reshape(-1, self.size_input, self.size_input, self.num_classes_abstract)
        return mask_agent_pred

    @torch.no_grad()
    # @profile
    def sample_from_uniform_prior(self, obs_curr, num_samples=None, code2exclude=None):
        if self.samples_uniform.device != obs_curr.device:
            self.samples_uniform = self.samples_uniform.to(obs_curr.device)
        samples = self.samples_uniform
        assert samples.shape[0] == self.num_categories ** self.num_categoricals
        if code2exclude is not None:
            assert isinstance(code2exclude, torch.Tensor) and len(code2exclude.shape) == 2
            to_remove = torch.zeros(samples.shape[0], dtype=torch.bool, device=samples.device)
            all_codes = samples.reshape(-1, self.num_categories * self.num_categoricals).bool()
            for idx_row in range(code2exclude.shape[0]):
                to_exclude = code2exclude[idx_row, :].reshape(1, -1).bool()
                coincidence = (to_exclude == all_codes).all(-1)
                to_remove |= coincidence
            samples = samples[~to_remove]
        if num_samples is not None and num_samples < samples.shape[0]:
            indices = torch.randperm(samples.shape[0])[:num_samples]
            samples = samples[indices]
        return samples, self.forward_from_one(obs_curr, samples)

    @torch.no_grad()
    def forward_from_one(self, obs_curr, samples):
        assert obs_curr.shape[0] == 1 and samples is not None
        context = self.encoder_context(obs_curr)
        size_batch = samples.shape[0]
        logits_mask_pos_agent = self.logits_from_code(samples, torch.repeat_interleave(context, size_batch, dim=0))
        mask_agent_pred = self.mask_from_logits(logits_mask_pos_agent)
        return mask_agent_pred.bool()

    @torch.no_grad()
    def fuse_mask_with_obs(self, mask, obs):
        layout, _ = self.layout_extractor(obs)
        return self.decoder(layout, mask)

    @torch.no_grad()
    # @profile
    def generate_from_obs(self, obs, num_samples=None):
        assert num_samples is not None
        size_batch = obs.shape[0]
        if size_batch > 1:
            assert size_batch == num_samples
        code, mask_agent_pred = self.sample_from_uniform_prior(obs, num_samples=num_samples)
        obses_pred = self.fuse_mask_with_obs(mask_agent_pred, obs)
        return code.bool(), obses_pred

    @torch.no_grad()
    # @profile
    def imagine_batch_from_obs(self, obs):
        context = self.encoder_context(obs)
        size_batch = obs.shape[0]
        samples = (
            torch.distributions.OneHotCategorical(
                probs=torch.ones(size_batch, self.num_categoricals, self.num_categories, dtype=torch.float32, device=obs.device)
            )
            .sample()
            .reshape(size_batch, -1)
        )
        logits_mask_pos_agent = self.logits_from_code(samples, context)
        mask_agent_pred = self.mask_from_logits(logits_mask_pos_agent, noargmax=True)
        obses_pred = self.fuse_mask_with_obs(mask_agent_pred, obs)
        return obses_pred

    @torch.no_grad()
    # @profile
    def encode_from_obs(self, obs, no_argmax=False):
        logits_samples = self.compress_from_obs(obs)
        if self.argmax_latents and not no_argmax:
            samples = torch.nn.functional.one_hot(logits_samples.argmax(-1), num_classes=self.num_categories)
        else:
            dist = torch.distributions.OneHotCategorical(logits=logits_samples)
            samples = dist.sample()
        return samples

    @torch.no_grad()
    # @profile
    def decode_to_obs(self, samples, obs, context=None):
        size_batch = obs.shape[0]
        assert samples.shape[0] == size_batch
        if context is None:
            context = self.encoder_context(obs)
        samples = samples.reshape(size_batch, -1)
        logits_mask_pos_agent = self.logits_from_code(samples, context)
        mask_agent_pred = self.mask_from_logits(logits_mask_pos_agent)
        obses_pred = self.fuse_mask_with_obs(mask_agent_pred, obs)
        return obses_pred

    # @profile
    def forward(self, obs_curr, obs_targ=None, samples=None, train=False):
        size_batch = obs_curr.shape[0]
        context = self.encoder_context(obs_curr)
        if samples is None:
            assert obs_targ is not None
            logits_samples = self.compress_from_obs(obs_targ)
            if self.argmax_latents:
                argmax_samples = logits_samples.argmax(-1)
                samples = torch.nn.functional.one_hot(argmax_samples, num_classes=self.num_categories)
                probs_samples = logits_samples.softmax(-1)
                samples = probs_samples + (samples - probs_samples).detach()
            else:
                samples = torch.distributions.OneHotCategoricalStraightThrough(logits=logits_samples).rsample()
        samples = samples.reshape(size_batch, -1)
        logits_mask_pos_agent = self.logits_from_code(samples, context)
        logits_mask_pos_agent = logits_mask_pos_agent.reshape(size_batch, -1)
        if train:
            return logits_samples, logits_mask_pos_agent
        else:
            with torch.no_grad():
                mask_agent_pred = self.mask_from_logits(logits_mask_pos_agent)
            return mask_agent_pred.bool()

    # @profile
    def compute_loss(self, batch_processed, debug=False):
        batch_obs_curr, batch_action, batch_reward, batch_obs_next, batch_done, batch_obs_targ, batch_obs_targ2, weights, batch_idxes = batch_processed
        with torch.no_grad():
            obses_context, obses_chosen = batch_obs_curr, batch_obs_targ

        size_batch = obses_chosen.shape[0]
        layouts_train, masks_agent_train = self.layout_extractor(obses_chosen)
        logits_samples, logits_mask_pos_agent_train = self.forward(obs_curr=obses_context, obs_targ=obses_chosen, train=True)
        logsoftmax_mask_agent = logits_mask_pos_agent_train.log_softmax(-1)
        loss_recon = torch.nn.functional.kl_div(
            input=logsoftmax_mask_agent, target=masks_agent_train.float().reshape(size_batch, -1), log_target=False, reduction="none"
        ).sum(-1)

        loss_align = None

        # maximize the kl-loss
        probs_samples = logits_samples.softmax(-1)
        h1_minus_h2 = probs_samples * ((probs_samples + 1e-7).log_() + self.minus_log_uniform)
        loss_entropy = h1_minus_h2.reshape(size_batch, -1).sum(-1)
        
        loss_conditional_prior = None

        coeff_schedule = cyclical_schedule(step=self.steps_trained, interval=self.interval_beta)
        loss_overall = self.beta * coeff_schedule * loss_entropy + loss_recon

        self.steps_trained += 1

        if not debug:
            return loss_overall, loss_recon, loss_entropy, loss_conditional_prior, loss_align, None, None, None, None, None, None, None
        else:
            with torch.no_grad():
                masks_agent_pred = (
                    torch.nn.functional.one_hot(logits_mask_pos_agent_train.argmax(-1), obses_chosen.shape[1] * obses_chosen.shape[2] * self.num_classes_abstract)
                    .bool()
                    .reshape(size_batch, obses_chosen.shape[-3], obses_chosen.shape[-2], self.num_classes_abstract)
                )
                obses_pred = self.decoder(layouts_train, masks_agent_pred)

                code_chosen = torch.eye(logits_samples.shape[-1], device=logits_samples.device, dtype=torch.long)[logits_samples.argmax(-1)]
                code_pred = self.encode_from_obs(obses_pred)
                ratio_aligned = (code_chosen == code_pred).reshape(size_batch, -1).all(-1).float().mean()

                dist_L1 = torch.abs(obses_pred.float() - obses_chosen.float())
                mask_perfect_recon = dist_L1.reshape(dist_L1.shape[0], -1).sum(-1) == 0
                ratio_perfect_recon = mask_perfect_recon.sum() / mask_perfect_recon.shape[0]
                mask_agent = obses_chosen[:, :, :, 0] == self.object_to_idx["agent"]
                dist_L1_mean = dist_L1.mean()
                dist_L1_nontrivial = dist_L1[mask_agent].mean()
                dist_L1_trivial = dist_L1[~mask_agent].mean()
                uniformity = (probs_samples.mean(0) - 1.0 / self.num_categories).abs_().mean()
                entropy_prior = None
            return (
                loss_overall,
                loss_recon,
                loss_entropy,
                loss_conditional_prior,
                loss_align,
                dist_L1_mean,
                dist_L1_nontrivial,
                dist_L1_trivial,
                uniformity,
                entropy_prior,
                ratio_perfect_recon,
                ratio_aligned,
            )


class Embedder_MiniGrid_BOW(torch.nn.Module):  # adapted from BabyAI 1.1
    def __init__(self, max_value=32, dim_embed=8, channels_obs=2, height=8, width=8, ebd_pos=False):
        super().__init__()
        self.max_value = max_value
        self.dim_embed = dim_embed
        self.width, self.height = width, height
        self.ebd_pos = ebd_pos
        self.channels_obs = channels_obs + int(self.ebd_pos) * 2
        if self.ebd_pos:
            self.meshgrid_x, self.meshgrid_y = torch.meshgrid(torch.arange(self.width), torch.arange(self.height), indexing="ij")
            self.meshgrid_x = self.meshgrid_x.reshape(1, self.width, self.height, 1).contiguous().detach()
            self.meshgrid_y = self.meshgrid_y.reshape(1, self.width, self.height, 1).contiguous().detach()
            self.max_value = max(max_value, self.width, self.height)
        self.embedding = torch.nn.Embedding(self.channels_obs * self.max_value, dim_embed)  # NOTE(H): +2 for X and Y
        if self.channels_obs > 1:
            offset = []
            for index_channel in range(self.channels_obs):
                offset.append(index_channel * self.max_value)
            self.register_buffer("offsets", torch.Tensor(offset).long().reshape(1, 1, 1, -1).contiguous().to(self.embedding.weight.device))

    def to(self, device):
        super().to(device)
        self.embedding.to(device)
        if self.ebd_pos:
            self.meshgrid_x = self.meshgrid_x.to(device)
            self.meshgrid_y = self.meshgrid_y.to(device)
        if self.channels_obs > 1:
            self.offsets = self.offsets.to(device)

    def parameters(self):
        parameters = list(self.embedding.parameters())
        return parameters

    # @profile
    def forward(self, inputs):
        with torch.no_grad():
            if self.ebd_pos:
                inputs = torch.cat(
                    [
                        inputs,
                        self.meshgrid_x.expand(inputs.shape[0], self.width, self.height, 1).detach(),
                        self.meshgrid_y.expand(inputs.shape[0], self.width, self.height, 1).detach(),
                    ],
                    dim=-1,
                )
            else:
                inputs = inputs.long()
            if self.channels_obs > 1:
                inputs += self.offsets
        # permute so it can be fed to conv layers in the encoder
        return self.embedding(inputs.detach()).sum(-2).permute(0, 3, 1, 2).contiguous()


class Encoder_MiniGrid(torch.nn.Module):
    """
    minigrid observation encoder from the Conscious Planning paper
    inputs an observation from the environment and outputs a vector representation of the states
    """
    def __init__(self, dim_embed, obs_sample, norm=True, append_pos=False, activation=torch.nn.ReLU):
        super(Encoder_MiniGrid, self).__init__()
        self.norm = norm
        self.activation = activation
        self.embedder = Embedder_MiniGrid_BOW(
            dim_embed=dim_embed, width=obs_sample.shape[-3], height=obs_sample.shape[-2], channels_obs=obs_sample.shape[-1], ebd_pos=bool(append_pos)
        )
        self.layers = ResidualBlock(len_in=dim_embed, width=None, kernel_size=3, depth=2, stride=1, padding=1, activation=activation)

    def to(self, device):
        super().to(device)
        self.embedder.to(device)
        self.layers.to(device)

    def parameters(self):
        parameters = list(self.embedder.parameters())
        parameters += list(self.layers.parameters())
        return parameters

    # @profile
    def forward(self, obs_minigrid):
        rep_bow = self.embedder(obs_minigrid)
        return self.layers(rep_bow)


class Binder_MiniGrid(torch.nn.Module):
    """
    create a local perception field with state_curr and state_targ
    """

    def __init__(self, sample_input, len_rep, norm=True, activation=torch.nn.ReLU, num_heads=1, size_bottleneck=4, size_field=8, type_arch="CP"):
        super(Binder_MiniGrid, self).__init__()
        self.norm = norm
        dim_embed = sample_input.shape[1]
        self.len_rep = len_rep
        self.len_out = 2 * len_rep
        self.activation = activation
        self.local_perception = "local" in type_arch.lower()
        if self.local_perception:
            self.extractor_fields = torch.nn.Conv2d(dim_embed, len_rep, kernel_size=size_field, stride=1, padding=0)
            self.register_buffer("query", torch.zeros(1, 1, len_rep))
            if size_bottleneck == 0:
                print("BINDER: size_bottleneck == 0, fall back to standard attention")
                self.attn = torch.nn.MultiheadAttention(embed_dim=len_rep, num_heads=num_heads, kdim=len_rep, vdim=len_rep, batch_first=True, dropout=0.0)
            else:
                self.attn = TopKMultiheadAttention(
                    embed_dim=len_rep,
                    num_heads=num_heads,
                    kdim=len_rep,
                    vdim=len_rep,
                    batch_first=True,
                    dropout=0.0,
                    size_bottleneck=size_bottleneck,
                    no_out_proj=num_heads == 1,
                )
            if self.norm:
                self.layer_norm_1 = torch.nn.LayerNorm(len_rep)
                self.layer_norm_2 = torch.nn.LayerNorm(len_rep)
        else:
            self.flattener = torch.nn.Sequential(
                activation(False),
                torch.nn.Flatten(),
                torch.nn.Linear(sample_input.shape[-1] * sample_input.shape[-2] * sample_input.shape[-3], len_rep),
            )

    def to(self, device):
        super().to(device)
        if self.local_perception:
            self.extractor_fields.to(device)
            self.query = self.query.to(device)
            self.attn.to(device)
            if self.norm:
                self.layer_norm_1.to(device)
                self.layer_norm_2.to(device)
        else:
            self.flattener.to(device)

    def parameters(self):
        if self.local_perception:
            parameters = list(self.extractor_fields.parameters())
            if self.norm:
                parameters += list(self.layer_norm_1.parameters())
                parameters += list(self.layer_norm_2.parameters())
            parameters += list(self.attn.parameters())
            # parameters += [self.query]
            return parameters
        else:
            return list(self.flattener.parameters())

    def extract_local_field(self, state):
        size_batch = state.shape[0]
        fields = self.extractor_fields(state).permute(0, 2, 3, 1).reshape(size_batch, -1, self.len_rep)
        if self.norm:
            fields = self.layer_norm_1(fields)
        state_local, _ = self.attn(self.query.expand(size_batch, 1, self.len_rep), fields, fields, need_weights=False)
        if self.norm:
            state_local = self.layer_norm_2(state_local)
        state_local = self.activation()(state_local)
        state_local = state_local.reshape(size_batch, self.len_rep)
        return state_local

    def forward(self, state_curr, state_targ, return_curr=False):
        size_batch = state_curr.shape[0]
        states_stacked_curr_targ = torch.cat([state_curr, state_targ], dim=0)
        if self.local_perception:
            state_local_curr_targ = self.extract_local_field(states_stacked_curr_targ)
        else:
            state_local_curr_targ = self.flattener(states_stacked_curr_targ)
        state_local_curr, state_local_targ = torch.split(state_local_curr_targ, [size_batch, size_batch], dim=0)
        state_binded = torch.cat([state_local_curr, state_local_targ], dim=-1)
        if return_curr:
            return state_binded, state_local_curr
        else:
            return state_binded


class Predictor_MiniGrid(torch.nn.Module):
    """
    on top of the extracted states, this predicts interesting values
    """

    def __init__(
        self,
        num_actions,
        len_input,
        depth=3,
        width=256,
        activation=torch.nn.ReLU,
        norm=True,
        dict_head=[{"len_predict": None, "dist_out": True, "value_min": 0.0, "value_max": 1.0, "atoms": 4, "classify": False}],
        value_min=None,
        value_max=None,
    ):
        super(Predictor_MiniGrid, self).__init__()
        self.len_input = len_input
        self.num_actions = num_actions
        self.dict_head = dict_head
        self.dist_output = bool(dict_head["dist_out"])
        self.norm = norm
        if dict_head["len_predict"] is None:
            self.len_predict = self.num_actions
        else:
            self.len_predict = dict_head["len_predict"]
        if dict_head["dist_out"]:
            assert "value_min" in dict_head and "value_max" in dict_head
            assert "atoms" in dict_head and "classify" in dict_head
            self.histogram_converter = HistogramConverter(value_min=dict_head["value_min"], value_max=dict_head["value_max"], atoms=dict_head["atoms"])
            self.len_output = self.len_predict * dict_head["atoms"]
            self.atoms = dict_head["atoms"]
            self.classify = dict_head["classify"]
            self.value_min, self.value_max = dict_head["value_min"], dict_head["value_max"]
        else:
            self.histogram_converter = None
            self.len_output = self.len_predict
            self.value_min, self.value_max = value_min, value_max

        self.layers = []
        for idx_layer in range(depth):
            len_in, len_out = width, width
            if idx_layer == 0:
                len_in = self.len_input
            if idx_layer == depth - 1:
                len_out = self.len_output
            if idx_layer > 0:
                self.layers.append(activation(True))
            self.layers.append(torch.nn.Linear(len_in, len_out))
        self.layers = torch.nn.Sequential(*self.layers)
        init_weights(self.layers)

    def to(self, device):
        super().to(device)
        self.layers.to(device)
        if self.histogram_converter is not None:
            self.histogram_converter.to(device)

    def parameters(self):
        return list(self.layers.parameters())

    def forward(self, input, action=None, scalarize=False):
        size_batch = input.shape[0]
        predicted = self.layers(input.reshape(size_batch, -1))
        if action is not None:
            assert action.device == predicted.device
            predicted = predicted.reshape(size_batch, self.len_predict, -1)
            predicted = predicted[torch.arange(size_batch, device=predicted.device), action.squeeze()]
        if self.dist_output:
            if action is None:
                predicted = predicted.reshape(size_batch, -1, self.atoms).contiguous()
            else:
                predicted = predicted.reshape(size_batch, self.atoms).contiguous()
            if scalarize:
                with torch.no_grad():
                    if self.classify:
                        return self.histogram_converter.support[predicted.argmax(-1)]
                    else:
                        return self.histogram_converter.from_histogram(predicted, logits=True)
            else:
                return predicted
        else:
            return predicted
