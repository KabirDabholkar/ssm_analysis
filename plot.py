import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from typing import Optional
from functools import partial
import os
from utils import make_path_if_not_exist
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


def collater(
        main_dir = 'all_models_validated_finetuning/state5_obs5_eps0.1_emeps0.6_GT',
        sub_dir='models_traintrials2000',
    ):
    #options = ['augmented_mode','augmented_mode_long','augmented_with_shift_mode','augmented_with_shift_repeatwithoutshift_mode','vanilla_sliced_mode']#'vanilla_mode'
    #options = ['vanilla_sliced_mode','sliced_and_augmented_with_small_shifts']
    #options = ['training_vanilla', 'training_augmented_with_shift', 'training_augmented_with_shift_then_vanilla','training_augmented_with_shift_then_vanilla_frozen_te'] # 'training_augmented'
    #options = ['pretrain_vanilla_then_finetuning_emission_vanilla','pretrain_augmented_with_shift_then_finetuning_emission_vanilla']
    # options = ['training_vanilla']
    options = ['']
    main_path = os.path.join(main_dir,sub_dir)
    dir_names = [main_path + opt for opt in options]
    #dir_names += ['models_traintrials700_'+'vanilla_sliced_mode']
    all_DFs = []
    for dir_name in dir_names:
        files = os.listdir(dir_name)
        #print(files,len(files))
        for file_name in files:
            fullpath = os.path.join(dir_name, file_name)
            #print(dir_name,file_name,fullpath,os.path.splitext(fullpath)[1])
            if os.path.splitext(fullpath)[-1]=='.csv':
                DF = pd.read_csv(fullpath,index_col=None)
                all_DFs.append(DF)
    if sub_dir!='':
        for d in os.listdir(main_dir):
            if d[-4:]=='.csv':
                all_DFs += [pd.read_csv(os.path.join(main_dir,d),index_col=None)] #.replace('_validated','')
    DF = pd.concat(all_DFs,ignore_index=True)
    # print(DF['13-shot co-smoothing'])
    return DF,main_path


def plot_scatter_with_lines(
        x: Optional[str],
        y: Optional[str],
        data: pd.DataFrame,
        data_lines: pd.DataFrame,
        save_path,
        hue = None,
        func1 = sns.scatterplot,
        func2 = None,
        sortby = [],
        xlabel=None,
        ylabel=None,
        xlim=[],
        ylim=[],
        hlines=[],
        logscale = {},
        print_corrcoef = False,
        zoom_inset: Optional[dict] = None,
):
    fig,axs = plt.subplots()
    all_axes = [axs]
    if zoom_inset is not None:
        # ax_ins = inset_axes(axs, **zoom_inset)
        ax_ins = axs.inset_axes(**zoom_inset)
        all_axes.append(ax_ins)
    for ax in all_axes:
        data_ = data.sort_values(by=sortby)
        func1(x=x, y=y, hue=hue, data=data_, ax=ax)
        if func2:
            # print(func2)
            # print(data[[x,y,hue]].sort())
            # data_ = data.sort_values(by=hue)
            func2(x=x,y=y,data = data_,ax=ax)
        if data_lines is not None:
            # print('here',data_lines[y].values[0])
            for color,(name_,x_,y_) in zip(['black','blue','red'],data_lines[['model_name',x,y]].values):
                # print(x_,y_)
                # l = ax.axhline(data_lines[y].values[0], ls='dashed', color='black')
                # l = ax.axvline(data_lines[x].values[0], ls='dashed', color='black')
                l = ax.axhline(y_, ls='dashed', color=color,label=name_)
                ax.axvline(x_, ls='dashed', color=color)
        for i,ls in enumerate(hlines):
            ax.axhline(ls,color='C%d'%i,ls='dashed',lw=1)
        handles, labels = ax.get_legend_handles_labels()
        # if data_lines is not None:
        #     handles += [l]
        #     labels  += ['Ground-truth']
    axs.legend(handles,labels,fontsize=7,framealpha=0.3)
    if xlim:
        axs.set_xlim(*xlim)
    if ylim:
        axs.set_ylim(*ylim)
    axs.set_xlabel(x if xlabel is None else xlabel)
    axs.set_ylabel(y if ylabel is None else ylabel)
    if print_corrcoef:
        a,b = data[[x,y]].dropna().values.T
        print(a,b)
        corrcoef = np.corrcoef(a,b)[0,1]
        print('corrcoef',corrcoef)
        ax.set_title('Corrcoef=%.2f'%corrcoef)
    if zoom_inset:
        # ax_ins.set_xlim(*zoom_xlim)
        # ax_ins.set_ylim(*zoom_ylim)
        ax_ins.set_xlabel(None)
        ax_ins.set_ylabel(None)
        legend = ax_ins.legend()
        legend.remove()
        axs.indicate_inset_zoom(ax_ins, edgecolor="black")
    for key,val in logscale.items():
        getattr(axs,key)(val)

    fig.tight_layout()
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    fig.savefig(save_path,dpi=300)
    plt.close()


def get_paired_columns(columns):
    """
    Generate paired columns for train and test based on unique base names.

    Parameters:
    columns (list of str): List of column names.

    Returns:
    list of tuple: List of tuples containing paired columns.
    """
    base_names = set(col.split('_', 1)[1] for col in columns if col.startswith('train_'))
    paired_columns = [(f'train_{base}', f'test_{base}') for base in base_names]
    return paired_columns

def transform_dataframe(df, id_vars, paired_columns):
    """
    Transform the input DataFrame by melting paired columns into a single column with version information.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    id_vars (list of str): List of column names to be retained as identifier variables.
    paired_columns (list of tuple): List of tuples where each tuple contains the pair of column names
                                    for 'train' and 'test' versions.

    Returns:
    pd.DataFrame: Transformed DataFrame with columns [<id_vars>, <variable names>, 'version'].
    """
    # Create a list to store the individual melted DataFrames
    melted_dfs = []

    # Loop through each pair of columns and melt them
    print(paired_columns)
    for train_col, test_col in paired_columns:
        variable_name = train_col.split('_')[1]
        melted_df = df.melt(id_vars=id_vars,
                            value_vars=[train_col, test_col],
                            var_name='version',
                            value_name=variable_name)
        melted_df['version'] = melted_df['version'].str.replace(f'_{variable_name}', '')
        melted_dfs.append(melted_df)
    print(melted_dfs[1].columns)
    # Merge all the melted DataFrames on id_vars and 'version'
    result_df = melted_dfs[0]
    for melted_df in melted_dfs[1:]:
        result_df = result_df.merge(melted_df, on=id_vars + ['version'])

    return result_df

def main():
    #DF = pd.read_csv('plots/collated.csv',index_col=0)

    DF,main_dir = collater(
        # main_dir='all_models_validated_v2/teacher_state5_poisson_partial_eps0.1_length10',
        # sub_dir='models_traintrials500'
        main_dir='results/teacher_LinearGaussianSSM_statedim4_emissiondim5_offdiag_train10',
        # main_dir='results/teacher_GaussianHMM_numstates5_emissiondim5_train10',
        sub_dir=''
    )
    print(DF['model_name'])

    # Melt the DataFrame to combine 'train_C', 'test_C', 'train_D', and 'test_D' into 'C' and 'D' columns
    # select_cols = DF.columns.str.contains('loglikelihood')
    # print(select_cols)

    # DFmelt = transform_dataframe(DF,DF.columns[~select_cols],get_paired_columns(DF.columns[select_cols]))
    # DFmelt = DFmelt[DFmelt!='Ground truth']

    select_GT = DF['model_name']=='Ground truth'
    DF_GT = DF[select_GT]
    DF_GT['state_dim'] = 4
    # DF_GT['num_states'] = 5
    DF = DF[~select_GT]
    # DF_GT = DF[DF['model_name'].str.contains('teacher')]
    print(DF['model_name'])
    DF['state_dim'] = DF['model_name'].str.split('_').str[1].str.split('dim').str[1].astype(int)
    DF = DF.sort_values(by=['state_dim'])
    # DF['num_states'] = DF['model_name'].str.split('_').str[1].str.split('states').str[1].astype(int)
    # DF = DF.sort_values(by=['num_states'])


    plot_configs = [
        {
            'x': 'state_dim',
            # 'x': 'num_states',
            'y': 'test_loglikelihood_score_mean',
            'data'  : DF,
            'func1': partial(sns.lineplot,lw=2,marker='o'),
            'data_lines' : DF_GT,
            'save_path': os.path.join('plots', main_dir, 'test_loglikelihood.png'),
            'logscale':{'set_xscale':'log'},
        },
        {
            # 'x': 'num_states',
            'x': 'state_dim',
            'y': 'train_loglikelihood_score_mean',
            'data': DF,
            'func1': partial(sns.lineplot,lw=2,marker='o'),
            'data_lines': DF_GT,
            'save_path': os.path.join('plots', main_dir, 'train_loglikelihood.png'),
            'logscale': {'set_xscale': 'log'},
        },
        {
            # 'x': 'num_states',
            'x': 'state_dim',
            'y': r'decoding_student_train->teacher_train',
            'data': DF,
            'func1': partial(sns.lineplot, lw=2, marker='o'),
            'data_lines': DF_GT,
            'ylim':(-0.1,1.05),
            'save_path': os.path.join('plots', main_dir, 'decoding_student_train->teacher_train.png'),
            'logscale': {'set_xscale': 'log'},
        },
        {
            # 'x': 'num_states',
            'x': 'state_dim',
            'y': r'decoding_student_test->teacher_test',
            'data': DF,
            'func1': partial(sns.lineplot, lw=2, marker='o'),
            'data_lines': DF_GT,
            'ylim': (-0.1, 1.05),
            'save_path': os.path.join('plots', main_dir, 'decoding_student_test->teacher_test.png'),
            'logscale': {'set_xscale': 'log'},
        },
        {
            # 'x': 'num_states',
            'x': 'state_dim',
            'y': r'decoding_teacher_train->student_train',
            'data': DF,
            'func1': partial(sns.lineplot, lw=2, marker='o'),
            'data_lines': DF_GT,
            'ylim': (-0.1, 1.05),
            'save_path': os.path.join('plots', main_dir, 'decoding_teacher_train->student_train.png'),
            'logscale': {'set_xscale': 'log'},
        },
        {
            # 'x': 'num_states',
            'x': 'state_dim',
            'y': r'decoding_teacher_test->student_test',
            'data': DF,
            'func1': partial(sns.lineplot, lw=2, marker='o'),
            'data_lines': DF_GT,
            'ylim': (-0.1, 1.05),
            'save_path': os.path.join('plots', main_dir, 'decoding_teacher_test->student_test.png'),
            'logscale': {'set_xscale': 'log'},
        },
        # {
        #     'x': 'state_dim',
        #     'y': 'loglikehood_score_mean',
        #     'hue': 'variant',
        #     'data': DFmelt,
        #     'data_lines': DF_GT,
        #     'save_path': os.path.join('plots', main_dir, 'traintest_loglikelihood.png'),
        # }
    ]
    for arg in plot_configs[:]:
        plot_scatter_with_lines(**arg)

if __name__ == '__main__':
    #collater()
    #main_summary()
    main()
