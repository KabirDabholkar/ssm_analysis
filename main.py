import os.path

import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import jax.random as jr
import jax.numpy as jnp
from jax import vmap
from config_utils import instantiate
import pandas as pd

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

CONFIG_PATH = "configs"
# CONFIG_NAME = "test"
CONFIG_NAME = "lineargaussian_sweep"


@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def decorated_main(cfg):
    return main(cfg)


def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    teacher = instantiate(cfg.teacher)
    student = instantiate(cfg.student)

    # print(dir(teacher))

    key1, key2, key3 = jr.split(jr.PRNGKey(0), 3)
    keys = jr.split(key2,cfg.train_trials)
    true_params, _ = teacher.initialize(key1)
    sample_many_trials = vmap(teacher.sample,(None,0,None),(0,0))
    train_true_states, train_emissions = sample_many_trials(
            true_params, keys, cfg.num_timesteps
    )
    keys = jr.split(key3, cfg.test_trials)
    test_true_states, test_emissions = sample_many_trials(
        true_params, keys, cfg.num_timesteps
    )
    # print(true_states.shape,emissions.shape)

    # key1, key2, key3 = jr.split(jr.PRNGKey(1), 3)

    keys = jr.split(key2,10)

    # true_params, _ = student.initialize(key1)
    # sample_many_trials = vmap(student.sample,(None,0,None),(0,0))
    # true_states, emissions = sample_many_trials(
    #         true_params, keys, cfg.num_timesteps
    # )
    if cfg.use_teacher_as_student:
        student = teacher
        new_params = true_params
        student.model_name = 'Ground truth'
        cfg.result_save_path = cfg.result_save_path_if_use_teacher_as_student
    else:
        params, props = student.initialize(key1)
        # new_params,new_props = partial(student.fit_sgd,)(params,props,train_emissions)
        new_params,_ = getattr(student, cfg.optimizer.algorithm)(
            params,
            props,
            train_emissions,
            **instantiate(cfg.optimizer.fit_kwargs)
        )
    # def cosmoother():
    # from copy import deepcopy
    # import numpy as np
    # def split_model(model,params,indices):
    #     E = model.emission_dim
    #     indices = np.array(indices)
    #     indices = indices[indices<E]
    #     split_models = [deepcopy(model) for _ in range(len(indices)+1)]
    #     section_sizes = np.diff(np.concatenate([np.zeros(1),indices])).append(E)
    #     print(section_sizes)
    #     # split_models = []
    #
    # split_model(student,new_params,[3])


    results = {}
    results['model_name'] = student.model_name

    #### run on train/test
    marginal_log_prob_many_trials = vmap(student.marginal_log_prob, (None, 0), 0)
    for name,dataset in zip(['train','test'],[train_emissions,test_emissions]):
        loglikehood_scores = marginal_log_prob_many_trials(new_params,dataset)

        results[name+'_loglikelihood_score_mean'] = float(loglikehood_scores.mean())
        results[name+'_loglikelihood_score_SEM'] = float(np.std(loglikehood_scores)/np.sqrt(loglikehood_scores.shape[0]))

    print(results)

    D = pd.DataFrame([results])
    savepath = cfg.result_save_path + '.csv'
    print(savepath)
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))

    D.to_csv(savepath)



    # print('loglikehood_score',loglikehood_score_mean,r"\pm",loglikehood_score_SEM)














# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    decorated_main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
