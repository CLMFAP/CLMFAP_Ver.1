import os
import re
import numpy as np
import pandas as pd
import torch as t
import json


def CheckFileExists(Filename):
    if os.path.exists(Filename):
        return True
    else:
        return False


def FileNames(Settings, Mode):
    RootPath = Settings["RootPath"]
    try:
        CodeVersion = Settings["CodeVersion"]
    except:
        CodeVersion = None

    # Original Files
    if Mode == "Dataset":
        DatasetNames = {
            "BBBP": "BBBP_screened.csv",
            "HIV": "HIV_screened.csv",
            "Lipo": "Lipophilicity_screened.csv",
            "BACE": "bace_screened.csv",
            "ClinTox": "clintox_screened.csv",
            "ESOL": "delaney-processed_screened.csv",
            "FreeSolv": "freesolv_screened.csv",
            "MUV": "muv_screened.csv",
            "QM7": "qm7_screened.csv",
            "QM8": "qm8_screened.csv",
            "QM9": "qm9_screened.csv",
            "SIDER": "sider_screened.csv",
            "Tox21": "tox21_screened.csv",
            "Toxcast": "toxcast_data_screened.csv",
        }
        TargetTask = Settings["TargetTask"]
        filename = f"{RootPath}Data/Screened/{DatasetNames[TargetTask]}"
        return filename

    elif Mode == "Feature":
        PretrainModel = Settings["PretrainModel"]
        TargetTask = Settings["TargetTask"]
        filename = f"{RootPath}ExtractedFeatures/{PretrainModel}/{TargetTask}_Features"
        # if PretrainModel == 'MorganFP':
        #     suffix = '.npy'
        # else:
        #     suffix = '.pt'
        suffix = ".pt"
        filename = filename + suffix
        return filename

    elif Mode == "Property":
        TargetTask = Settings["TargetTask"]
        filename = f"{RootPath}ExtractedFeatures/Properties/{TargetTask}_Label.npy"
        return filename

    elif Mode == "Thresholds":
        PretrainModel = Settings["PretrainModel"]
        TargetTask = Settings["TargetTask"]
        filename = (
            f"{RootPath}ExtractedFeatures/{PretrainModel}/{TargetTask}_Thresholds.json"
        )
        return filename

    elif Mode == "Epsilon":
        TargetTask = Settings["TargetTask"]
        filename = f"{RootPath}ExtractedFeatures/Properties/{TargetTask}_Epsilon.json"
        return filename

    # Step 1: Sim and Diff Files

    elif Mode == "FeatureMinMaxEud":
        PretrainModel = Settings["PretrainModel"]
        TargetTask = Settings["TargetTask"]
        filename = f"{RootPath}ExtractedFeatures/{PretrainModel}/{TargetTask}_MinMaxEuclidean_DistanceArray.json"
        return filename

    elif Mode == "FeatureCosineSim":
        PretrainModel = Settings["PretrainModel"]
        TargetTask = Settings["TargetTask"]
        filename = f"{RootPath}ExtractedFeatures/{PretrainModel}/{TargetTask}_CosineSim_SimilarityArray.json"
        return filename

    elif Mode == "SimArray":
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

    # Step 2: Draw RPS Maps

    elif Mode == "RPSMap":
        PretrainModel = Settings["PretrainModel"]
        TargetTask = Settings["TargetTask"]
        Zmetric = Settings["Zmetric"]
        Fd2s = Settings["Fd2s"]

        if Zmetric == "Combine":
            CombineGamma = Settings["CombineGamma"]
            if Fd2s == "LogP":
                LogPValue = Settings["LogPValue"]
                filename = f"{RootPath}{CodeVersion}/Results/RPSMaps/{PretrainModel}_{TargetTask}_{Zmetric}_{CombineGamma}_{Fd2s}_{LogPValue}_FPSMap.png"
            else:
                filename = f"{RootPath}{CodeVersion}/Results/RPSMaps/{PretrainModel}_{TargetTask}_{Zmetric}_{CombineGamma}_{Fd2s}_FPSMap.png"
        else:
            if Fd2s == "LogP":
                LogPValue = Settings["LogPValue"]
                filename = f"{RootPath}{CodeVersion}/Results/RPSMaps/{PretrainModel}_{TargetTask}_{Zmetric}_{Fd2s}_{LogPValue}_FPSMap.png"
            else:
                filename = f"{RootPath}{CodeVersion}/Results/RPSMaps/{PretrainModel}_{TargetTask}_{Zmetric}_{Fd2s}_FPSMap.png"
        return filename
    # Step 3: Calculate Metrics

    elif Mode == "M2Array":
        PretrainModel = Settings["PretrainModel"]
        TargetTask = Settings["TargetTask"]
        Zmetric = Settings["Zmetric"]
        Fd2s = Settings["Fd2s"]

        if Zmetric == "Combine":
            CombineGamma = Settings["CombineGamma"]
            if Fd2s == "LogP":
                LogPValue = Settings["LogPValue"]
                filename = f"{RootPath}{CodeVersion}/Results/M2/{PretrainModel}_{TargetTask}_{Zmetric}_{CombineGamma}_{Fd2s}_{LogPValue}_M2.json"
            else:
                filename = f"{RootPath}{CodeVersion}/Results/M2/{PretrainModel}_{TargetTask}_{Zmetric}_{CombineGamma}_{Fd2s}_M2.json"
        else:
            if Fd2s == "LogP":
                LogPValue = Settings["LogPValue"]
                filename = f"{RootPath}{CodeVersion}/Results/M2/{PretrainModel}_{TargetTask}_{Zmetric}_{Fd2s}_{LogPValue}_M2.json"
            else:
                filename = f"{RootPath}{CodeVersion}/Results/M2/{PretrainModel}_{TargetTask}_{Zmetric}_{Fd2s}_M2.json"
        return filename

    elif Mode == "M1Array":
        PretrainModel = Settings["PretrainModel"]
        TargetTask = Settings["TargetTask"]
        Zmetric = Settings["Zmetric"]
        Fd2s = Settings["Fd2s"]

        if Zmetric == "Combine":
            CombineGamma = Settings["CombineGamma"]
            if Fd2s == "LogP":
                LogPValue = Settings["LogPValue"]
                filename = f"{RootPath}{CodeVersion}/Results/M1/{PretrainModel}_{TargetTask}_{Zmetric}_{CombineGamma}_{Fd2s}_{LogPValue}_M1.json"
            else:
                filename = f"{RootPath}{CodeVersion}/Results/M1/{PretrainModel}_{TargetTask}_{Zmetric}_{CombineGamma}_{Fd2s}_M1.json"
        else:
            if Fd2s == "LogP":
                LogPValue = Settings["LogPValue"]
                filename = f"{RootPath}{CodeVersion}/Results/M1/{PretrainModel}_{TargetTask}_{Zmetric}_{Fd2s}_{LogPValue}_M1.json"
            else:
                filename = f"{RootPath}{CodeVersion}/Results/M1/{PretrainModel}_{TargetTask}_{Zmetric}_{Fd2s}_M1.json"
        return filename

    elif Mode == "M3Array":
        PretrainModel = Settings["PretrainModel"]
        TargetTask = Settings["TargetTask"]
        Zmetric = Settings["Zmetric"]
        Fd2s = Settings["Fd2s"]

        if Zmetric == "Combine":
            CombineGamma = Settings["CombineGamma"]
            if Fd2s == "LogP":
                LogPValue = Settings["LogPValue"]
                filename = f"{RootPath}{CodeVersion}/Results/M3/{PretrainModel}_{TargetTask}_{Zmetric}_{CombineGamma}_{Fd2s}_{LogPValue}_M3.json"
            else:
                filename = f"{RootPath}{CodeVersion}/Results/M3/{PretrainModel}_{TargetTask}_{Zmetric}_{CombineGamma}_{Fd2s}_M3.json"
        else:
            if Fd2s == "LogP":
                LogPValue = Settings["LogPValue"]
                filename = f"{RootPath}{CodeVersion}/Results/M3/{PretrainModel}_{TargetTask}_{Zmetric}_{Fd2s}_{LogPValue}_M3.json"
            else:
                filename = f"{RootPath}{CodeVersion}/Results/M3/{PretrainModel}_{TargetTask}_{Zmetric}_{Fd2s}_M3.json"
        return filename

    elif Mode == "M3C1C4":
        PretrainModel = Settings["PretrainModel"]
        TargetTask = Settings["TargetTask"]
        Zmetric = Settings["Zmetric"]
        Fd2s = Settings["Fd2s"]

        if Zmetric == "Combine":
            CombineGamma = Settings["CombineGamma"]
            if Fd2s == "LogP":
                LogPValue = Settings["LogPValue"]
                filename = f"{RootPath}{CodeVersion}/Results/M3C1C4/{PretrainModel}_{TargetTask}_{Zmetric}_{CombineGamma}_{Fd2s}_{LogPValue}_M3C1C4.json"
            else:
                filename = f"{RootPath}{CodeVersion}/Results/M3C1C4/{PretrainModel}_{TargetTask}_{Zmetric}_{CombineGamma}_{Fd2s}_M3C1C4.json"
        else:
            if Fd2s == "LogP":
                LogPValue = Settings["LogPValue"]
                filename = f"{RootPath}{CodeVersion}/Results/M3C1C4/{PretrainModel}_{TargetTask}_{Zmetric}_{Fd2s}_{LogPValue}_M3C1C4.json"
            else:
                filename = f"{RootPath}{CodeVersion}/Results/M3C1C4/{PretrainModel}_{TargetTask}_{Zmetric}_{Fd2s}_M3C1C4.json"
        return filename

    elif Mode == "MorganFPC1C4":
        PretrainModel = "MorganFP"
        TargetTask = Settings["TargetTask"]
        Zmetric = Settings["Zmetric"]
        Fd2s = Settings["Fd2s"]

        if Zmetric == "Combine":
            CombineGamma = Settings["CombineGamma"]
            if Fd2s == "LogP":
                LogPValue = Settings["LogPValue"]
                filename = f"{RootPath}{CodeVersion}/Results/C1C4/{PretrainModel}_{TargetTask}_{Zmetric}_{CombineGamma}_{Fd2s}_{LogPValue}_C1C4.json"
            else:
                filename = f"{RootPath}{CodeVersion}/Results/C1C4/{PretrainModel}_{TargetTask}_{Zmetric}_{CombineGamma}_{Fd2s}_C1C4.json"
        else:
            if Fd2s == "LogP":
                LogPValue = Settings["LogPValue"]
                filename = f"{RootPath}{CodeVersion}/Results/C1C4/{PretrainModel}_{TargetTask}_{Zmetric}_{Fd2s}_{LogPValue}_C1C4.json"
            else:
                filename = f"{RootPath}{CodeVersion}/Results/C1C4/{PretrainModel}_{TargetTask}_{Zmetric}_{Fd2s}_C1C4.json"
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


TaskPropertyNames = {
    "BACE": ["Class"],
    "BBBP": ["p_np"],
    "ClinTox": ["FDA_APPROVED", "CT_TOX"],
    "ESOL": ["measured log solubility in mols per litre"],
    "FreeSolv": ["y"],
    "Lipo": ["exp"],
    "QM7": ["u0_atom"],
    "QM8": [
        "E1-CC2",
        "E2-CC2",
        "f1-CC2",
        "f2-CC2",
        "E1-PBE0",
        "E2-PBE0",
        "f1-PBE0",
        "f2-PBE0",
        "E1-CAM",
        "E2-CAM",
        "f1-CAM",
        "f2-CAM",
    ],
    "SIDER": [
        "Hepatobiliary disorders",
        "Metabolism and nutrition disorders",
        "Product issues",
        "Eye disorders",
        "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders",
        "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders",
        "Surgical and medical procedures",
        "Vascular disorders",
        "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders",
        "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders",
        "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    ],
    "Tox21": [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ],
}

###############################################


class TaskPropertyExtractor(object):
    def __init__(self, settings):
        super(TaskPropertyExtractor, self).__init__()
        self.settings = settings

    def PropertyExtract(self):
        TaskSetDst = FileNames(self.settings, "Dataset")

        all_properties = self._TargetTaskDatasetFeatureExtract(TaskSetDst)
        all_properties = np.array(all_properties)

        # save properties into file.
        PropertyFileName = FileNames(self.settings, "Property")
        np.save(PropertyFileName, all_properties)

    def _TargetTaskDatasetFeatureExtract(self, TaskDatasetPath):
        TaskName = self.settings["TargetTask"]
        taskpropertyname = TaskPropertyNames[TaskName]

        df = pd.read_csv(TaskDatasetPath)

        all_properties = [[] for i in range(len(taskpropertyname))]

        for i in range(len(taskpropertyname)):
            propertyname = taskpropertyname[i]
            properties = df[propertyname]
            total_num = len(properties)
            for j in range(total_num):
                value = properties[j]
                all_properties[i].append(value)

        # all_properties shape: [TaskNums, total_num]
        return all_properties


###############################################


def draw_scatter_new(dict_1, dict_2, settings, model_name='model', result_folder =''):
    # prepare data
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import Polygon

    model = "IBD-ChemBerta"
    dataset = "ESOL"
    x = list(dict_1.values())
    y = []
    for key in dict_1.keys():
        y.append(dict_2.get(key))
    # set Times new Roman
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    colors = {
        "ESOL": "salmon",
        "FreeSolv": "skyblue",
        "SIDER": "yellowgreen",
        "Tox21": "hotpink",
        "BACE": "paleturquoise",
        "BBBP": "orange",
        "Lipo": "sandybrown",
        "QM7": "plum",
        "QM8": "gold",
        "ClinTox": "skyblue",
        "mytest": "orange",
    }
    # draw figure
    fig = plt.figure(figsize=(8, 6), dpi=200, facecolor="w", edgecolor="k")
    axes = fig.add_axes([0.13, 0.13, 0.8, 0.8])
    # axes = fig.add_axes([0,0,0,0])
    axes.scatter(
        x,
        y,
        marker="o",
        s=12,
        color=colors.get(dataset),
        edgecolors="grey",
    )
    # reference line
    thresholds = settings["Thresholds"]
    if "epsilon_AC" in thresholds.keys():
        epsilon_AC = thresholds["epsilon_AC"]
        epsilon_SH = thresholds["epsilon_SH"]
        delta_AC = thresholds["delta_AC"]
        delta_SH = thresholds["delta_SH"]
        delta_AC_s = 1 - delta_AC
        delta_SH_s = 1 - delta_SH
        axes.axvline(x=delta_AC_s, color="darkgrey", ls="--", lw=0.7, dashes=(5, 1.8))
        axes.axvline(x=delta_SH_s, color="darkgrey", ls="--", lw=0.7, dashes=(5, 1.8))
        vertex = np.array(
            [[delta_AC_s, epsilon_AC], [1, epsilon_AC], [1, 1], [delta_AC_s, 1]]
        )
        polygon = Polygon(vertex, facecolor="grey", edgecolor=None, alpha=0.4)
        axes.add_patch(polygon)
        # axes.set_facecolor('orange')
        # axes.set_alpha(0.4)
        axes.axhline(y=epsilon_AC, color="darkgrey", ls="--", lw=0.7, dashes=(5, 1.8))
        axes.axhline(y=epsilon_SH, color="darkgrey", ls="--", lw=0.7, dashes=(5, 1.8))
        vertex = np.array(
            [[delta_SH_s, epsilon_SH], [0, epsilon_SH], [0, 0], [delta_SH_s, 0]]
        )
        polygon = Polygon(vertex, facecolor="grey", edgecolor=None, alpha=0.4)
        axes.add_patch(polygon)

    else:
        epsilon = thresholds["epsilon"]
        delta_AC = thresholds["delta_AC"]
        delta_SH = thresholds["delta_SH"]
        delta_AC_s = 1 - delta_AC
        delta_SH_s = 1 - delta_SH
        print(delta_AC_s)
        print(delta_SH_s)
        axes.axvline(x=delta_AC_s, color="darkgrey", ls="--", lw=0.7, dashes=(5, 1.8))
        axes.axvline(x=delta_SH_s, color="darkgrey", ls="--", lw=0.7, dashes=(5, 1.8))
        axes.axhline(y=epsilon, color="darkgrey", ls="--", lw=0.7, dashes=(5, 1.8))
        vertex = np.array(
            [[delta_AC_s, epsilon], [1, epsilon], [1, 1], [delta_AC_s, 1]]
        )
        polygon = Polygon(vertex, facecolor="grey", edgecolor=None, alpha=0.4)
        axes.add_patch(polygon)
        vertex = np.array(
            [[delta_SH_s, epsilon], [0, epsilon], [0, 0], [delta_SH_s, 0]]
        )
        polygon = Polygon(vertex, facecolor="grey", edgecolor=None, alpha=0.4)
        axes.add_patch(polygon)
    # axes.axvline(x=0.5,color ='darkgrey',ls='--',lw=0.7,dashes=(5,1.8))
    # axes.axhline(y=0.5,color ='darkgrey',ls='--',lw=0.7,dashes=(5,1.8))

    #######decoration######
    # axes.set_title(dataset,color='k',fontsize=14, fontweight='medium',y=-0.15)
    axes.tick_params(direction="in", width=1)
    # set range and label
    axes.set(xlim=(0, 1), ylim=(0, 1))
    axes.set_xlabel("Representation Similarity", fontsize=20)
    axes.set_ylabel("Property Distance", fontsize=20)
    # axes.text(0.03,0.945,dataset,fontsize=12,bbox=dict(boxstyle='round',facecolor='w',edgecolor='lightgrey'))
    # axes.set_title(dataset, color="k", fontsize=14, fontweight="medium", y=-0.15)
    # axes.legend(fontsize=12)
    # plt.show()
    # set dpi and output *eps/*pdf
    # filename = FileNames(settings, 'RPSMap')
    fig.savefig(f"{result_folder}/repra_esol_{model_name}.png", dpi=200, format="png")
