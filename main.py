import os.path

import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import jax.random as jr
import jax.numpy as jnp
from jax import vmap
from config_utils import instantiate
import pandas as pd

from dynamax.linear_gaussian_ssm import lgssm_smoother, parallel_lgssm_smoother
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


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


    ###### fitting/copying student on/from teacher ########
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

    if cfg.analysis.sample_posterior:
        key,another_key = jr.split(key3, 2)
        vmap_posterior_sample = vmap(student.posterior_sample, (None, None, 0))
        posterior_samples = vmap_posterior_sample(another_key,new_params,train_emissions)

        lgssm_posteriors = vmap(lambda y: student.smoother(params, y))


        fig,ax = plt.subplots()
        ax.plot(posterior_samples[:,:,0].T,posterior_samples[:,:,1].T)
        savepath = cfg.result_save_path + '_posterior_trajectories.png'
        fig.savefig(savepath,dpi=300)
        # print(posterior_samples.shape)


    results = {}
    results['model_name'] = student.model_name

    #### run on train/test
    if cfg.analysis.compute_test_train_loglikelihood:
        marginal_log_prob_many_trials = vmap(student.marginal_log_prob, (None, 0), 0)
        for name,dataset in zip(['train','test'],[train_emissions,test_emissions]):
            loglikehood_scores = marginal_log_prob_many_trials(new_params,dataset)

            results[name+'_loglikelihood_score_mean'] = float(loglikehood_scores.mean())
            results[name+'_loglikelihood_score_SEM'] = float(np.std(loglikehood_scores)/np.sqrt(loglikehood_scores.shape[0]))

    posterior_dict = {}
    #### run on train/test
    if cfg.analysis.compute_decoding:
        for model_name, model_params in zip(['teacher', 'student'], [true_params,new_params]):
            for data_partition_name, dataset in zip(['train', 'test'], [train_emissions, test_emissions]):
                # posterior = vmap(parallel_lgssm_smoother, (None, 0), 0)(model_params, dataset[0])
                def meanandcov(x):
                    posterior = parallel_lgssm_smoother(model_params, x)
                    return (posterior.smoothed_means,posterior.smoothed_covariances)
                posterior = vmap(meanandcov,0,(0,0))(dataset)
                posterior_dict['_'.join([model_name, data_partition_name])] = posterior

        # print([[np.array(p_).shape for p_ in p] for _, p in posterior_dict.items()])
        min_num_trials = min([p[0].shape[0] for _,p in posterior_dict.items()])
        posterior_dict = {k: v[0] for k, v in posterior_dict.items()}
        posterior_dict = {k:v[:min_num_trials] for k,v in posterior_dict.items()}
        posterior_dict = {k: v.reshape(-1,*v.shape[2:]) for k, v in posterior_dict.items()}

        # for posterior_name_from, X in posterior_dict.items():
        #     for posterior_name_to, y in posterior_dict.items():
        #         X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
        #         if hasattr(cfg.decoding, 'preprocess_target'):
        #             y_train = instantiate(cfg.decoding.preprocess_target)(y_train)
        #
        #         model = instantiate(cfg.decoding.regression_model)
        #         model.fit(
        #             X_train,
        #             y_train
        #         )
        #
        #         pred_y_test = getattr(model, cfg.decoding.predict_method)(X_test)
        #
        #         metric = instantiate(cfg.decoding.metric)
        #         score = np.stack([metric(
        #             y_test[i],
        #             pred_y_test[i]
        #         ) for i in range(pred_y_test.shape[0])]).mean()
        #
        #         results['decoding_'+'->'.join([posterior_name_from,posterior_name_to])] = score
    print(results)

    if cfg.analysis.save_results:
        D = pd.DataFrame([results])
        savepath = cfg.result_save_path + '.csv'
        print(savepath)
        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))

        D.to_csv(savepath)

    # print('teacher',true_params)
    # print('student',new_params)

    # print('loglikehood_score',loglikehood_score_mean,r"\pm",loglikehood_score_SEM)




if __name__ == '__main__':
    decorated_main()

