from omnisafe.envs.core import make, support_envs
import torch
import torch.nn.functional as F
from omnisafe.utils.config import Config


def estimate_true_value(agent, 
                    env_id: str, 
                    num_envs: int, 
                    seed: int, 
                    cfgs: Config, 
                    discount: float, 
                    eval_episodes=10):
        """Estimates true Q-value via launching given policy from sampled state until
        the end of an episode. """

        eval_env = make(env_id, num_envs=num_envs, device=cfgs.train_cfgs.device)
        
        true_cvalues, true_rvalues = [], []
        estimate_rvalues, estimate_cvalues = [], []
        for _ in range(eval_episodes):
            obs0, _ = eval_env.reset()

            _, estimate_rvalue, estimate_cvalue, _ = agent.step(obs0)

            obs = obs0

            true_cvalue = 0.0
            true_rvalue = 0.0
            step = 0
            while True:
                act, _, _, _ = agent.step(obs)
                next_obs, r, c, termniated, truncated, info = eval_env.step(act)
                true_cvalue += c * (discount ** step)
                true_rvalue += r * (discount ** step)
                step += 1
                obs = next_obs

                if termniated or truncated:
                    break
            true_cvalues.append(true_cvalue)
            true_rvalues.append(true_rvalue)
            estimate_cvalues.append(estimate_cvalue)
            estimate_rvalues.append(estimate_rvalue)
            print("Estimation took: ", step)

        c_error = torch.mean(torch.stack(true_cvalues) - torch.stack(estimate_cvalues))
        r_error = torch.mean(torch.stack(true_rvalues) - torch.stack(estimate_rvalues))

        true_c = torch.mean(torch.stack(true_cvalues))
        true_r = torch.mean(torch.stack(true_rvalues))
        estimate_c = torch.mean(torch.stack(estimate_cvalues))
        estimate_r = torch.mean(torch.stack(estimate_rvalues))
        print(true_r, estimate_r, r_error)    
        return c_error, true_c, estimate_c, r_error, true_r, estimate_r
        #return torch.mean(torch.stack(true_cvalues) - torch.stack(estimate_cvalues)), torch.mean(torch.stack(true_cvalues)), torch.mean(torch.stack(estimate_cvalues))


class RandomProjection(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RandomProjection, self).__init__()
        self.linear_projection = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear_projection(x)
