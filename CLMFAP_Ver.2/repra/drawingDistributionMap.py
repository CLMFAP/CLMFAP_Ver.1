import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats
import re
import torch as t
import json


def FileNames(Settings, Mode):
    RootPath = Settings["RootPath"]

    if Mode == "SimArray":
        PretrainModel = Settings["PretrainModel"]
        TargetTask = Settings["TargetTask"]
        Zmetric = Settings["Zmetric"]
        Fd2s = Settings["Fd2s"]

        if Zmetric == "Combine":
            CombineGamma = Settings["CombineGamma"]
            if Fd2s == "LogP":
                LogPValue = Settings["LogPValue"]
                filename = f"{RootPath}ExtractedFeatures/{PretrainModel}/{TargetTask}_{Zmetric}_{CombineGamma}_{Fd2s}_{LogPValue}_SimArray.json"
            else:
                filename = f"{RootPath}ExtractedFeatures/{PretrainModel}/{TargetTask}_{Zmetric}_{CombineGamma}_{Fd2s}_SimArray.json"
        else:
            if Fd2s == "LogP":
                LogPValue = Settings["LogPValue"]
                filename = f"{RootPath}ExtractedFeatures/{PretrainModel}/{TargetTask}_{Zmetric}_{Fd2s}_{LogPValue}_SimArray.json"
            else:
                filename = f"{RootPath}ExtractedFeatures/{PretrainModel}/{TargetTask}_{Zmetric}_{Fd2s}_SimArray.json"
        return filename

    elif Mode == "DiffArray":
        TargetTask = Settings["TargetTask"]
        filename = f"{RootPath}ExtractedFeatures/Properties/{TargetTask}_MinMax_EuclideanDist_DiffArray.json"
        return filename

    else:
        raise KeyError("Wrong Mode Key Given!")


def LoadFile(filename):
    suffix = re.split("\.", filename)[-1]
    if suffix == "npy":
        Content = np.load(filename)
    elif suffix == "npz":
        Content = np.load(filename)
        keys = Content.files
        if len(keys) == 1:
            Content = Content[keys[0]]
        else:
            raise KeyError("Undetermined content key!")
    elif suffix == "pt":
        Content = t.load(filename, map_location="cpu")
        if Content.__class__ == t.Tensor:
            Content = Content.cpu()
        Content = np.array(Content)
    elif suffix == "json":
        with open(filename, "r") as f:
            Content = json.load(f)
    else:
        raise TypeError("Unknown file suffix!")

    return Content


def draw_r_distribution_new(dict_1, dict_2, Settings, model_name='model', result_folder =''):
    dataset = Settings["TargetTask"]
    model = Settings["PretrainModel"]
    Metric = Settings["Zmetric"]
    # prepare data
    x_s = list(dict_1.values())  # Similarity
    x_d = []
    for key in dict_1.keys():
        x_d.append(dict_2.get(key))  # Difference of properties
    # color dict
    colors = {"ESOL": "#F46C3F", "FreeSolv": "#75BDE0"}
    # set Times new Roman
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    h = sns.jointplot(
        x=x_s,
        y=x_d,
        kind="hex",
        color="orange",
        marginal_kws=dict(bins=100, edgecolor=None),
    )
    plt.subplots_adjust(left=0.12, right=0.88, top=0.9, bottom=0.16)
    # get the current positions of the joint ax and the ax for the marginal x
    pos_joint_ax = h.ax_joint.get_position()
    ax = h.fig.add_axes([0.9, pos_joint_ax.y0, 0.02, pos_joint_ax.height])
    plt.colorbar(cax=ax)
    h.ax_joint.set(xlim=(0, 1), ylim=(0, 1))
    h.ax_joint.set_xlabel("Representation Similarity", fontsize=20)
    h.ax_joint.set_ylabel("Property Distance", fontsize=20)
    # h.ax_joint.set_title(
    #     dataset, color="k", fontsize=14, fontweight="medium", x=0.6, y=-0.22
    # )
    h.ax_joint.tick_params(direction="in", width=1)
    # r, p = pearson(x_s, x_d)
    (phantom,) = h.ax_joint.plot([], [], linestyle="", alpha=0)
    # h.ax_joint.legend([phantom],['p={:.3f}, r={:.3f}'.format(p,r)])
    # plt.show()
    # fig.savefig("{model}_{dataset}_{Metric}_distribution_regression.png",dpi=200,format='png')
    h.savefig(
        f"{result_folder}/repra_esol_{model_name}_distribution_regression.png", dpi=200, format="png"
    )