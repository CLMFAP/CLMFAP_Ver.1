from repra.computing import *
from repra.drawingDistributionMap import *

class Repra:
    def __init__(self, features = [], labels = [], model_name='model', result_folder = '') -> None:
        self.features = np.concatenate(features).tolist()
        self.labels = np.concatenate(labels).tolist()
        self.model_name = model_name
        self.result_folder = result_folder

    def draw(self):

        Settings = {
            "RootPath": "~/codes/MetricForFeatureSpace/",
            "CodeVersion": "MetricExpCodes",
            "Zmetric": "MinMaxEud",
            "Fd2s": "Linear",
            "Ymetric": "MinMaxEud",
        }

        Properties = np.array([self.labels])
        Features = np.array(self.features)

        # print("Labels")
        # print(self.labels)

        # Calculate DiffArray   d(y1,y2)
        # Properties = FillMissingValueByDefault(Properties)
        DiffArray = CalDiffArray(Properties)

        # Calculate MinMaxEud of Features of PretrainModel on TargetTask
        # d(z1,z2)
        DistArray = CalMinMaxEuclidean(Features)
        # print(DistArray)

        # Calculate CosineSim of Features of PretrainModel on TargetTask
        # SimArray = CalCosineSimArray(Features)

        SimArray = CalSimArray(Settings, DistArray)
        # print(SimArray)

        # Calculate Delta and epsilon
        # if Classification

        print(f"Calculating Thresholds:")
        print(f"Calculating epsilon.")
        # epsilon = ClassificationTaskEpsilonCalculating(Properties)
        (delta_AC, epsilon_AC), (delta_SH, epsilon_SH) = RegressionTaskDeltaEpsilonCalculating(
            DiffArray, DistArray
        )
        thresholds = {
            "delta_AC": delta_AC,
            "delta_SH": delta_SH,
            "epsilon_AC": epsilon_AC,
            "epsilon_SH": epsilon_SH,
        }

        print(f"Thresholds: {thresholds}")
        Settings.update({"Thresholds": thresholds})

        # print(self.result_folder)
        # print(self.labels)
        # print(self.features)
        # print(len(DistArray))
        # print(len(SimArray))

        # Draw RPSMap
        # print(f"Drawing RPSMaps")
        draw_scatter_new(SimArray, DiffArray, Settings, model_name=self.model_name, result_folder =self.result_folder)


        Distribution_Settings = {
            "RootPath": "~/codes/MetricForFeatureSpace/",
            "CodeVersion": "MetricExpCodes",
            "PretrainModel": "IBD-ChemBerta",
            "TargetTask": "ESOL",
            "Zmetric": "CosineSim",
            "Fd2s": "Linear",
            "Ymetric": "MinMaxEud",
        }
        Distribution_Settings.update({"Thresholds": thresholds})

        draw_r_distribution_new(SimArray, DiffArray, Distribution_Settings, model_name=self.model_name, result_folder =self.result_folder)

        filename = f"{self.result_folder}/repra_result_{self.model_name}"
        PretrainModel = "MolBERT"
        TargetTask = "ESOL"

        with open(filename, "a") as f:
            f.write("Thresholds \n")
            json.dump(thresholds, f)
            f.write("\n")

        # Calculate M1
        # s_{AD}
        print(f"Calculating M1 value of {PretrainModel} on {TargetTask} dataset.")
        M1 = Metric1Calculating_Array(SimArray, DiffArray, Distribution_Settings)
        print(M1)
        with open(filename, "a") as f:
            f.write("s_AD: \n")
            json.dump(M1, f)
            f.write("\n")

        # Calculate M3
        # s_{IR}
        print(f"Calculating M3 value of {PretrainModel} on {TargetTask} dataset.")
        M3 = Metric3Calculating_Array(SimArray, DiffArray, Distribution_Settings)
        print(M3)
        with open(filename, "a") as f:
            f.write("s_IR: \n")
            json.dump(M3, f)
            f.write("\n")

        print(f"Calculating M3C1C4 value of {PretrainModel} on {TargetTask} dataset.")
        M3C1C4 = Metric3Calculating_Array_C1C4(SimArray, DiffArray, Distribution_Settings)
        print(M3C1C4)
        with open(filename, "a") as f:
            f.write("M3C1C4 result: \n")
            json.dump(M3C1C4, f)
            f.write("\n")
