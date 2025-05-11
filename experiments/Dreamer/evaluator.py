import torch

class HistogramConverter(torch.nn.Module):
    def __init__(self, value_min: float=-1.0, value_max: float=1.0, atoms: int=128):
        super(HistogramConverter, self).__init__()
        self.register_buffer("value_min", torch.tensor(value_min))
        self.register_buffer("value_max", torch.tensor(value_max))
        self.atoms = atoms
        self.value_span = value_max - value_min
        const_norm = torch.tensor((self.atoms - 1) / self.value_span)
        self.register_buffer("const_norm", const_norm)
        const_norm_inv = torch.tensor(self.value_span / (self.atoms - 1))
        self.register_buffer("const_norm_inv", const_norm_inv)
        support = torch.arange(self.atoms).float()
        self.register_buffer("support", support)

    def to(self, device):
        super().to(device)
        self.value_min = self.value_min.to(device)
        self.value_max = self.value_max.to(device)
        self.const_norm = self.const_norm.to(device)
        self.const_norm_inv = self.const_norm_inv.to(device)
        self.support = self.support.to(device)

    def parameters(self):
        return []

    @torch.no_grad()
    def to_histogram(self, value: torch.Tensor):
        value = value.clamp(self.value_min, self.value_max)  # NO in-place clipping!!! Do not alter the original
        value_normalized = (value - self.value_min) * self.const_norm  # normalize to [0, atoms - 1] range
        value_normalized.clamp_(0, self.atoms - 1)
        upper, lower = value_normalized.ceil().long(), value_normalized.floor().long()
        upper_weight = value_normalized % 1
        lower_weight = 1 - upper_weight
        dist = torch.zeros(value.shape[0], self.atoms, device=value.device, dtype=value.dtype)
        dist.scatter_add_(-1, lower, lower_weight)
        dist.scatter_add_(-1, upper, upper_weight)
        return dist  # validated with "self.from_histogram(dist, logits=False) - value.squeeze()"

    @torch.no_grad()
    def from_histogram(self, dist: torch.Tensor, logits: bool=True, support_hotswap: torch.Tensor=None):
        if logits:
            dist = torch.nn.functional.softmax(dist, -1)
        if support_hotswap is not None:
            assert support_hotswap.device == dist.device
            assert support_hotswap.shape == (self.atoms,)
            value = dist @ support_hotswap
            return value
        else:
            value_normalized = dist @ self.support
            value = self.value_min + value_normalized * self.const_norm_inv
            return value

class TargetEvaluator(torch.nn.Module):
    def __init__(
        self,
        type_action: str="discrete", # "discrete" or "continuous"
        len_action: int=None,
        num_actions: int=None,
        len_state: int=None,
        len_target: int=None,
        depth: int=4,
        width: int=400,
        activation=torch.nn.ELU,
        atoms: int=16,
        gamma: float=0.999,
        create_targetnet: bool=True,
        interval_sync_targetnet: int=500, # how many loss computations before automatically syncing the targetnet, the default is drawn from DQN
        create_optimizer: bool=True,
        type_optimizer: str="Adam",
        lr: float=0.00025,
        eps: float=1e-5,
        layernorm: bool=True,
        encoder_state: torch.nn.Module=None,
        encoder_target: torch.nn.Module=None,
    ):
        super(TargetEvaluator, self).__init__()
        self.device = "cpu"

        self.encoder_state = encoder_state
        self.encoder_target = encoder_target

        assert gamma <= 1 and gamma >= 0
        self.gamma = gamma

        # histogram output
        assert atoms is not None and atoms >= 2
        self.atoms = atoms
        self.histogram_converter = HistogramConverter(value_min=1, value_max=float(self.atoms), atoms=self.atoms)
        support_discount = self.gamma ** (self.histogram_converter.support + 1)
        support_discount[-1] = 0.0 # correction for the truncation
        self.register_buffer("support_discount", support_discount)

        assert len_state is not None, "len_state must be defined"
        self.len_state = len_state
        assert len_target is not None, "len_target must be defined"
        self.len_target = len_target

        self.type_action = type_action
        if self.type_action == "continuous": # for Q with continuous actions, Q takes the input of both the state and the action and outputs the Q value
            assert len_action is not None, "len_action must be defined for continuous action space"
            self.len_action, self.num_actions = len_action, 1
            self.len_input = len_state + len_action + len_target
            self.len_output = self.atoms
        elif self.type_action == "discrete": # for Q with discrete actions, Q takes the input of the state and outputs the Q value for each action
            assert num_actions is not None, "num_actions must be defined for discrete action space"
            self.len_action, self.num_actions = 1, num_actions
            self.len_input = len_state + len_target
            self.len_output = self.num_actions * self.atoms
        else:
            raise ValueError("type_action must be either 'discrete' or 'continuous'")
        
        self.layernorm = layernorm

        self.layers = []
        for idx_layer in range(depth):
            len_in, len_out = width, width
            if idx_layer == 0:
                len_in = self.len_input
            if idx_layer == depth - 1:
                len_out = self.len_output
            if idx_layer > 0:
                self.layers.append(activation(True))
            if self.layernorm and (idx_layer > 0 or (self.encoder_state is None and self.encoder_target is None)):
                self.layers.append(torch.nn.LayerNorm(len_in))
            self.layers.append(torch.nn.Linear(len_in, len_out))
        self.layers = torch.nn.Sequential(*self.layers)

        self.create_optimizer = bool(create_optimizer)
        if self.create_optimizer:
            self.params_optim = list(self.layers.parameters())
            if self.encoder_state is not None:
                self.params_optim += list(self.encoder_state.parameters())
            if self.encoder_target is not None:
                self.params_optim += list(self.encoder_target.parameters())
            self.params_optim = list(set(self.params_optim)) # remove duplicates
            from common.utils import AdamW_tf
            self.optimizer = AdamW_tf(self.params_optim, lr=lr, eps=eps, weight_decay=1e-2) # NOTE: use Dreamer-style MLPs and optimizer
        else:
            self.optimizer = None

        self.create_targetnet = bool(create_targetnet)
        if self.create_targetnet:
            import copy
            self.layers_target = copy.deepcopy(self.layers)
            for param in self.layers_target.parameters():
                param.requires_grad = False
            self.layers_target.eval()
            for module in self.layers_target.modules():
                module.eval()
            if self.encoder_state is not None:
                self.encoder_state_targetnet = copy.deepcopy(self.encoder_state)
                for param in self.encoder_state_targetnet.parameters():
                    param.requires_grad = False
                self.encoder_state_targetnet.eval()
                for module in self.encoder_state_targetnet.modules():
                    module.eval()
            else:
                self.encoder_state_targetnet = None
            if self.encoder_target is not None:
                self.encoder_target_targetnet = copy.deepcopy(self.encoder_target)
                for param in self.encoder_target_targetnet.parameters():
                    param.requires_grad = False
                self.encoder_target_targetnet.eval()
                for module in self.encoder_target_targetnet.modules():
                    module.eval()
            else:
                self.encoder_target_targetnet = None
            self.sync_targetnet()
            self.interval_sync_targetnet = interval_sync_targetnet
            self.last_sync_targetnet = 0
            self.counter_sync_targetnet = 0
        else:
            self.layers_target = self.layers
            self.state_encoder_target = self.state_encoder
            self.interval_sync_targetnet = None

    def to(self, device):
        super().to(device)
        self.layers.to(device)
        if self.layers_target is not None:
            self.layers_target.to(device)
        if self.histogram_converter is not None:
            self.histogram_converter.to(device)
        if self.encoder_state is not None:
            self.encoder_state.to(device)
        if self.encoder_target is not None:
            self.encoder_target.to(device)
        self.device = list(self.layers.parameters())[0].device

    def parameters(self):
        params = list(self.layers.parameters())
        if self.encoder_state is not None:
            params += list(self.encoder_state.parameters())
        if self.encoder_target is not None:
            params += list(self.encoder_target.parameters())
        return params

    def forward(self, source: torch.Tensor, target: torch.Tensor, action: torch.Tensor=None, type_output: str="logits", use_targetnet: bool=False):
        # this function computes the distribution of the distances from the source to the target
        assert type_output in ["logits", "discount", "distance"]
        assert source.shape[0] == target.shape[0]
        size_batch = source.shape[0]

        with torch.set_grad_enabled(type_output == "logits" and not use_targetnet): # the other types of outputs are just for inference
            if use_targetnet:
                if self.encoder_state is not None:
                    assert self.encoder_state_targetnet is not None
                    source = self.encoder_state_targetnet(source)
                if self.encoder_target is not None:
                    assert self.encoder_target_targetnet is not None
                    target = self.encoder_target_targetnet(target)
            else:
                if self.encoder_state is not None:
                    source = self.encoder_state(source)
                if self.encoder_target is not None:
                    target = self.encoder_target(target)

            layers = self.layers_target if use_targetnet else self.layers
            if self.type_action == "continuous":
                assert action is not None
                assert action.shape[0] == size_batch
                assert action.shape[1] == self.len_action
                input_combined = torch.cat([source.reshape(size_batch, -1), action.reshape(size_batch, -1), target.reshape(size_batch, -1)], dim=-1)
                predicted = layers(input_combined).reshape(size_batch, self.atoms)
            elif self.type_action == "discrete":
                input_combined = torch.cat([source.reshape(size_batch, -1), target.reshape(size_batch, -1)], dim=-1)
                predicted = layers(input_combined).reshape(size_batch, self.num_actions, self.atoms)
                if action is not None:
                    assert action.shape[0] == size_batch
                    assert action.device == predicted.device
                    predicted = predicted[torch.arange(size_batch, device=predicted.device), action.squeeze()]
            else:
                raise ValueError("type_action must be either 'discrete' or 'continuous'")
            if type_output == "logits":
                return predicted
            elif type_output == "discount":
                predicted = self.histogram_converter.from_histogram(predicted, logits=True, support_hotswap=self.support_discount)
                return predicted.detach()
            elif type_output == "distance":
                predicted = self.histogram_converter.from_histogram(predicted, logits=True)
                return predicted.detach()

    def train(self, source_curr: torch.Tensor, action_curr: torch.Tensor, source_next: torch.Tensor, action_next: torch.Tensor, target: torch.Tensor, mask_reached: torch.Tensor, mask_done: torch.Tensor=None, increment_counter: bool=True, update_params: bool=True, flag_debug: bool=False, weights: torch.Tensor=None):
        # source_curr: representation of the current state
        # action_curr: action taken in the current state to get to the next state
        # source_next: representation of the next state
        # action_next: action to be taken in the next state, calculated with the agent's policy
        # target: the representation of the target
        # mask_reached: mask indicating whether the target has been reached at source_next
        # mask_done (optional): mask indicating whether source_next is terminal
        # this function computes the loss for the feasibility evaluator, in the form of KL divergence, without weighting each batch
        # output shape: (bs,)
        assert source_curr.shape[0] == target.shape[0] == action_curr.shape[0] == mask_reached.shape[0] == source_next.shape[0] == action_next.shape[0]
        assert source_curr.device == action_curr.device == target.device == mask_reached.device == source_next.device == action_next.device

        if update_params:
            if self.optimizer is None:
                print("[EVALUATOR]: optimizer not created, cannot update params despite update_params=True")
            else:
                source_curr, action_curr, source_next, action_next, target = source_curr.detach(), action_curr.detach(), source_next.detach(), action_next.detach(), target.detach() # optimizer is only used when not shaping the representations
                self.optimizer.zero_grad(set_to_none=True)

        if self.encoder_state is not None:
            source_curr = self.encoder_state(source_curr)
            source_next = self.encoder_state(source_next)
        if self.encoder_target is not None:
            target = self.encoder_target(target)

        # compute the logits
        logits_curr = self.forward(source_curr, target, action=action_curr, type_output="logits")
        with torch.no_grad():
            distance_next = self.forward(source_next, target, action=action_next, type_output="distance", use_targetnet=True)
            if mask_done is not None:
                assert mask_done.shape[0] == target.shape[0]
                assert mask_done.device == target.device
                distance_next[mask_done] = 1000.0
            distance_next[mask_reached] = 0.0
            target_discount_distance = 1.0 + distance_next
            target_histogram = self.histogram_converter.to_histogram(target_discount_distance.reshape(-1, 1))
        loss = torch.nn.functional.kl_div(torch.log_softmax(logits_curr, -1), target_histogram.detach(), reduction="none").sum(-1)
        
        with torch.no_grad():
            loss_avg = loss.mean()
            if self.create_targetnet and increment_counter:
                self.counter_sync_targetnet += 1
                if self.counter_sync_targetnet - self.last_sync_targetnet >= self.interval_sync_targetnet:
                    print(f"[EVALUATOR]: attempted to sync_targetnet() after {self.counter_sync_targetnet:d} steps of training")
                    self.sync_targetnet()
                    self.last_sync_targetnet = self.counter_sync_targetnet
        
        norm_grad = None
        if update_params:
            if self.optimizer is None:
                print("[EVALUATOR]: optimizer not created, cannot update params despite update_params=True")
            else:
                if weights is not None:
                    assert weights.shape[0] == loss.shape[0]
                    assert weights.device == loss.device
                    loss_combined = (loss * weights.squeeze()).mean()
                else:
                    loss_combined = loss.mean()
                loss_combined.backward()
                norm_grad = torch.nn.utils.clip_grad_norm_(self.params_optim, 10.0)
                self.optimizer.step()

        if flag_debug:
            distance_curr = self.histogram_converter.from_histogram(logits_curr, logits=True)
        else:
            distance_curr = None
        return loss, loss_avg, distance_curr, target_discount_distance, norm_grad

    def sync_targetnet(self):
        # this function copies the weights from the main network to the target network
        if self.create_targetnet:
            for param, param_target in zip(self.layers.parameters(), self.layers_target.parameters()):
                param_target.data.copy_(param.data)
            if self.encoder_state is not None:
                assert self.encoder_state_targetnet is not None
                for param, param_target in zip(self.encoder_state.parameters(), self.encoder_state_targetnet.parameters()):
                    param_target.data.copy_(param.data)
            if self.encoder_target is not None:
                assert self.encoder_target_targetnet is not None
                for param, param_target in zip(self.encoder_target.parameters(), self.encoder_target_targetnet.parameters()):
                    param_target.data.copy_(param.data)
            print("[EVALUATOR]: targetnet synced!")
        else:
            print("[EVALUATOR]: targetnet not created, skipping sync_targetnet()")
