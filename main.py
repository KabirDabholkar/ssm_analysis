import os.path

import jax.numpy
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import jax.random as jr
import jax.numpy as jnp
from jax import vmap
from config_utils import instantiate
import pandas as pd
from utils import get_env_var
from pathlib import Path
import pickle as pkl
from nlb_tools.make_tensors import make_eval_input_tensors, make_train_input_tensors, save_to_h5, \
    make_eval_target_tensors, h5_to_dict
from nlb_tools.evaluation import evaluate
from omegaconf_utils import omegaconf_resolvers
import h5py

from dynamax.linear_gaussian_ssm import lgssm_smoother, parallel_lgssm_smoother
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


CONFIG_PATH = "configs"
# CONFIG_NAME = "test"
CONFIG_NAME = "lineargaussian_sweep"


RESULT_BASE_PATH = Path(get_env_var(variable_name='RESULT_BASE_PATH'))
print('RESULT_BASE_PATH:',RESULT_BASE_PATH)

@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def decorated_main(cfg):
    return main(cfg)


def main(cfg):
    omegaconf_resolvers()

    print(OmegaConf.to_yaml(cfg))

    key1, key2, key3 = jr.split(jr.PRNGKey(0), 3)

    if cfg.data_mode == 'student-teacher':
        teacher = instantiate(cfg.teacher)
        true_params, _ = teacher.initialize(key1, **instantiate(cfg.teacher.initialize_kwargs))
        data_jax = instantiate(cfg.generate_all_data_jax, _convert_='partial')(
            model=teacher,params=true_params, key=key1
        )
        states, emissions = data_jax
        # print(emissions.shape)
        data = instantiate(cfg.numpy_to_xarray_with_breakdownlabels, _convert_='partial')(data=emissions)
    else:
        cfg.teacher_path = cfg.teacher_path_if_data_mode_nlb
        dataset_name = cfg.load_dataset.dataset_name
        bin_size_ms = cfg.load_dataset.bin_size
        binsuf = '' if bin_size_ms == 5 else f'_{bin_size_ms}'
        phase = cfg.load_dataset.phase
        train_save_path = '_'.join(['train_dict', dataset_name, binsuf, phase]) + '.h5'
        eval_save_path = '_'.join(['eval_dict', dataset_name, binsuf, phase]) + '.h5'
        eval_target_save_path = '_'.join(['eval_target_dict', dataset_name, binsuf, phase]) + '.h5'

        train_save_path, eval_save_path, eval_target_save_path = [
            os.path.join(cfg.dataset_path, path) for path in
            [train_save_path, eval_save_path, eval_target_save_path]
        ]
        paths = [train_save_path, eval_save_path] + ([eval_target_save_path] if phase == 'val' else [])
        paths_exist = [os.path.exists(path) for path in paths]
        print(paths_exist)
        print('tensors exist',all(paths_exist))
        if not all(paths_exist):
            dataset = instantiate(cfg.load_dataset.dataset)

            if phase == 'val':
                train_split = 'train'
                eval_split = 'val'
            else:
                train_split = ['train', 'val']
                eval_split = 'test'

            # if hasattr(cfg,'preprocess'):
            #     for item in cfg.preprocess:
            #         getattr(dataset,item.method_name)(*item.args)
            dataset.resample(bin_size_ms)
            print(train_split,eval_split)
            print(train_save_path)
            os.makedirs(os.path.dirname(train_save_path), exist_ok=True)
            train_dict = make_train_input_tensors(dataset, dataset_name, train_split, save_file=True,
                                                  save_path=train_save_path)
            eval_dict = make_eval_input_tensors(dataset, dataset_name, eval_split, save_file=True,
                                                save_path=eval_save_path)
            if phase == 'val':
                print('making target dict')
                target_dict = make_eval_target_tensors(dataset, dataset_name, train_split, eval_split, save_file=True,
                                                       include_psth=True, save_path=eval_target_save_path)
        else:
            def load_h5_to_dict(path):
                D = {}
                with h5py.File(path, "r") as f:
                    D = h5_to_dict(f)
                return D

            train_dict = load_h5_to_dict(train_save_path)
            eval_dict = load_h5_to_dict(eval_save_path)
            if phase == 'val':
                target_dict = load_h5_to_dict(eval_target_save_path)

        train_spikes_heldin = train_dict['train_spikes_heldin'].astype(int)
        train_spikes_heldout = train_dict['train_spikes_heldout'].astype(int)
        print(train_spikes_heldin.shape, train_spikes_heldout.shape)
        data_numpy = np.concatenate([train_spikes_heldin, train_spikes_heldout], axis=-1)
        # print(data_numpy.shape)
        data = instantiate(cfg.numpy_to_xarray_with_breakdownlabels, _convert_='partial')(data=data_numpy)

    data = data.astype('float32')


    student = instantiate(cfg.student)

    # print(dir(teacher))


    # print(true_states.shape,emissions.shape)

    # key1, key2, key3 = jr.split(jr.PRNGKey(1), 3)

    keys = jr.split(key2,10)

    # true_params, _ = student.initialize(key1)
    # sample_many_trials = vmap(student.sample,(None,0,None),(0,0))
    # true_states, emissions = sample_many_trials(
    #         true_params, keys, cfg.num_timesteps
    # )


    ###### fitting/copying student on/from teacher ########
    losses = None
    if cfg.use_teacher_as_student:
        student = teacher
        new_params = true_params
        student.model_name = 'Ground truth'
        cfg.result_save_path = cfg.result_save_path_if_use_teacher_as_student
    else:
        params, props = student.initialize(key1)
        if cfg.run_train:
            # new_params,new_props = partial(student.fit_sgd,)(params,props,train_emissions)
            print(jax.numpy.array(data.select(trials_split='train')).shape)
            new_params,losses = getattr(student, cfg.optimizer.algorithm)(
                params,
                props,
                jax.numpy.array(data.select(trials_split='train')),
                **instantiate(cfg.optimizer.fit_kwargs)
            )

    cfg.result_save_path = str(RESULT_BASE_PATH / cfg.result_save_path)
    if cfg.save_student:
        os.makedirs(os.path.dirname(cfg.result_save_path), exist_ok=True)
        with open(cfg.result_save_path, 'wb') as f:
            pkl.dump({'object': student, 'params': new_params}, f)
    if cfg.load_student:
        with open(cfg.result_save_path, 'rb') as f:
            student_data = pkl.load(f)
        student = student_data['object']
        new_params = student_data['params']

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
    if losses is not None:
        fig,ax = plt.subplots()
        ax.plot(losses)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        savepath = cfg.result_save_path + '_losses.png'
        os.makedirs(os.path.dirname(savepath),exist_ok=True)
        fig.savefig(savepath, dpi=300)

    if cfg.analysis.sample_posterior:
        key,another_key = jr.split(key3, 2)
        vmap_posterior_sample = vmap(student.posterior_sample, (None, None, 0))
        train_emissions = jax.numpy.array(data.select(trials_split='train'))
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
        for data_partition_name in ['train', 'test']:
            dataset = jax.numpy.array(data.select(trials_split=data_partition_name))

            loglikehood_scores = marginal_log_prob_many_trials(new_params,dataset)

            results[data_partition_name+'_loglikelihood_score_mean'] = float(loglikehood_scores.mean())
            results[data_partition_name+'_loglikelihood_score_SEM'] = float(np.std(loglikehood_scores)/np.sqrt(loglikehood_scores.shape[0]))

    posterior_dict = {}
    #### run on train/test
    if cfg.analysis.compute_decoding:
        for model_name, model_params in zip(['teacher', 'student'], [true_params,new_params]):
            for data_partition_name in ['train', 'test']:
                dataset = jax.numpy.array(data.select(trials_split=data_partition_name))
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

        for posterior_name_from, X in posterior_dict.items():
            for posterior_name_to, y in posterior_dict.items():
                X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
                if hasattr(cfg.decoding, 'preprocess_target'):
                    y_train = instantiate(cfg.decoding.preprocess_target)(y_train)

                model = instantiate(cfg.decoding.regression_model)
                model.fit(
                    X_train,
                    y_train
                )

                pred_y_test = getattr(model, cfg.decoding.predict_method)(X_test)

                metric = instantiate(cfg.decoding.metric)
                score = np.stack([metric(
                    y_test[i],
                    pred_y_test[i]
                ) for i in range(pred_y_test.shape[0])]).mean()

                results['decoding_'+'->'.join([posterior_name_from,posterior_name_to])] = score
    print(results)

    if cfg.analysis.plot_matrices:

        fig, axs = plt.subplots(1, 2 + 1, figsize=(12,4))
        for ax,param_set,title in zip(axs,[true_params,params,new_params],['Teacher','Initial','Trained']):
            im = ax.imshow(param_set.dynamics.weights,aspect='equal')
            fig.colorbar(im,ax=ax,shrink=0.5)
            ax.set_title(title)

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        savepath = cfg.result_save_path + '_matrices.png'
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.tight_layout()
        fig.savefig(savepath, dpi=300)

        print(new_params.dynamics.weights.shape)
        true_params_dynamics_eigvals = np.linalg.eigvals(true_params.dynamics.weights)
        new_params_dynamics_eigvals = np.linalg.eigvals(new_params.dynamics.weights)
        params_dynamics_eigvals = np.linalg.eigvals(params.dynamics.weights)
        fig,ax = plt.subplots(1,1)
        import matplotlib.patches as patches
        circle = patches.Circle((0, 0), radius=1, fill=False, edgecolor='black', linewidth=1, linestyle='--')
        # Add the circle to the plot
        ax.add_patch(circle)
        for param_set,name,marker in zip([true_params,params,new_params],['Teacher','Initial','Trained'],['*','s','.']):
            dynamics_eigvals = np.linalg.eigvals(param_set.dynamics.weights)
            # ax.scatter(params_dynamics_eigvals.real, params_dynamics_eigvals.imag, label='Initial weights',marker='sq')
            # ax.scatter(params_dynamics_eigvals.real, params_dynamics_eigvals.imag,label='Initial weights')
            # ax.scatter(new_params_dynamics_eigvals.real,new_params_dynamics_eigvals.imag,label='Trained weights',s=10)
            ax.scatter(dynamics_eigvals.real,dynamics_eigvals.imag,label=name,alpha=0.6,marker=marker)

        # Set the aspect of the plot to be equal
        ax.set_aspect('equal', 'box')
        ax.legend(fontsize=8,framealpha=0.5)
        savepath = cfg.result_save_path + '_eigvals.png'
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=300)

    if cfg.analysis.save_results:
        D = pd.DataFrame([results])
        savepath = cfg.result_save_path + '.csv'
        print(savepath)

        os.makedirs(os.path.dirname(savepath),exist_ok=True)

        D.to_csv(savepath)

    # print('teacher',true_params)
    # print('student',new_params)

    # print('loglikehood_score',loglikehood_score_mean,r"\pm",loglikehood_score_SEM)




if __name__ == '__main__':
    decorated_main()

