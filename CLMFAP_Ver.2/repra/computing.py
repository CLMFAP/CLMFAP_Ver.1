import numpy as np
from sklearn.metrics import roc_auc_score
from repra.utils import *


##################################################################
# Basic Calculation Functions
def CalNorm(z):
    # || z ||
    return np.linalg.norm(z)


def CalSigmoid(x, T):
    # Sigmoid function with temperature
    # 1 / (1 + exp(-Tx))
    if T * x >= 0:
        return 1.0 / (1 + np.exp(-T * x))
    else:
        return np.exp(T * x) / (1 + np.exp(T * x))


def CalEuDist(z1, z2):
    # Eu Distance of z1, z2
    # || z1 - z2 ||
    return np.linalg.norm(z1 - z2)


def CalInnerProduct(z1, z2):
    # <z1,z2>
    return float(np.dot(z1, z2))


def CalCosine(z1, z2):
    # cos = <z1, z2> / (||z1|| * ||z2||)
    innerproduct = CalInnerProduct(z1, z2)
    norm = CalNorm(z1) * CalNorm(z2)
    if norm != 0:
        return innerproduct / norm
    else:
        return 0


def CalLogP(x, P):
    # Log_p( ((p^2 -1)/p) * x + 1/p)
    return -(np.log((P - 1) * x / P + 1 / P) / np.log(P))


##################################################################
# Diff Kernels


def CalAbsDiff(y1, y2):
    # d(y1,y2) = |y1-y2|
    return np.abs(y1 - y2)


def CalLogDiff(y1, y2):
    # d(y1,y2) = |log(y1) - log(y2)|
    return np.abs(np.log10(y1) - np.log10(y2))


##################################################################
# Norm Kernels


def MinMaxNormalization(SimArray):
    array = SimArray.values()
    array = list(array)
    array = np.array(array)
    max_value = np.max(array)
    min_value = np.min(array)
    NormSimArray = {}
    for key in SimArray.keys():
        # print(SimArray[key].__class__)
        # print(min_value.__class__)
        # print(min_value)
        value = (SimArray[key] - min_value) / (max_value - min_value)
        value = value.astype("float64")
        NormSimArray.update({key: value})
    print(NormSimArray["0,1"].__class__)
    return NormSimArray


##################################################################
# TransFuncs


def LinearTrans(x, k, b):
    # Linear Transform
    return k * x + b


def LogPTrans(x, P):
    # Log_P Transform
    # from [0,1] to [-1,1]
    return CalLogP(x, P)


##################################################################
def CalMinMaxEuclidean(Z):
    total_num = Z.shape[0]

    # Cal Eud for each pair-wise samples
    Array = {}
    for i in range(total_num):
        for j in range(i + 1, total_num):
            assert i != j
            z1 = Z[i]
            z2 = Z[j]
            dist = CalEuDist(z1, z2)
            Array.update({f"{i},{j}": dist})

    # Using MinMaxNormalization to transform the original Eud into MinMaxEud
    DistArray = MinMaxNormalization(Array)
    # Using DistArray since they are [0,1],d

    return DistArray


def CalCosineSimArray(Z):
    total_num = Z.shape[0]

    SimArray = {}
    for i in range(total_num):
        for j in range(i + 1, total_num):
            assert i != j
            z1 = Z[i]
            z2 = Z[j]
            sim = CalCosine(z1, z2)

            sim.astype("float64")
            SimArray.update({f"{i},{j}": sim})
    return SimArray


##################################################################


def CalSimArray(Settings, dict_features):
    # Calculate D_Z
    # step 1: Load Pre-Calculated File for D_Z
    # step 2: Combine PreCalculated values for D_Z if necessary
    # step 3: Calculate Fd2s
    Zmetric = Settings["Zmetric"]
    if Zmetric == "MinMaxEud":
        # step1
        # filename = FileNames(Settings, 'FeatureMinMaxEud')
        MinMaxEudDistArray = dict_features

        # step3
        Fd2s = Settings["Fd2s"]
        if Fd2s == "Linear":
            allvalues = list(MinMaxEudDistArray.values())
            allkeys = list(MinMaxEudDistArray.keys())
            allvalues = np.array(allvalues)

            transallvalues = 1 - allvalues
            SimArray = {}

            for i in range(len(allkeys)):
                assert allvalues[i] == MinMaxEudDistArray[allkeys[i]]

                transvalue = transallvalues[i]
                SimArray.update({allkeys[i]: transvalue})

        elif Fd2s == "LogP":
            LogPValue = Settings["LogPValue"]

            allvalues = list(MinMaxEudDistArray.values())
            allkeys = list(MinMaxEudDistArray.keys())
            allvalues = np.array(allvalues)
            transallvalues = CalLogP(allvalues, LogPValue)

            SimArray = {}
            for i in range(len(allkeys)):
                assert allvalues[i] == MinMaxEudDistArray[allkeys[i]]

                transvalue = transallvalues[i]
                SimArray.update({allkeys[i]: transvalue})

    elif Zmetric == "CosineSim":
        # step1
        filename = FileNames(Settings, "FeatureCosineSim")
        CosineSimSimArray = LoadFile(filename)

        # step3
        Fd2s = Settings["Fd2s"]
        if Fd2s == "Linear":
            allvalues = list(CosineSimSimArray.values())
            allkeys = list(CosineSimSimArray.keys())
            allvalues = np.array(allvalues)

            transallvalues = (allvalues + 1) / 2
            SimArray = {}

            for i in range(len(allkeys)):
                assert allvalues[i] == CosineSimSimArray[allkeys[i]]

                transvalue = transallvalues[i]
                SimArray.update({allkeys[i]: transvalue})
        else:
            raise NotImplementedError(f"{Zmetric} with {Fd2s} is not implemented.")

    elif Zmetric == "Combine":
        # step1
        filename1 = FileNames(Settings, "FeatureMinMaxEud")
        MinMaxEudDistArray = LoadFile(filename1)
        filename2 = FileNames(Settings, "FeatureCosineSim")
        CosineSimSimArray = LoadFile(filename2)

        # step2
        MinMaxEudAllValues = np.array(list(MinMaxEudDistArray.values()))
        MinMaxEudAllKeys = list(MinMaxEudDistArray.keys())
        CosineSimAllValues = np.array(list(CosineSimSimArray.values()))
        CosineSimAllKeys = list(CosineSimSimArray.keys())

        assert MinMaxEudAllKeys == CosineSimAllKeys

        TransCosineSimAllValues = (-CosineSimAllValues + 1) / 2
        Gamma = Settings["CombineGamma"]
        CombineAllValues = (
            Gamma * MinMaxEudAllValues + (1 - Gamma) * TransCosineSimAllValues
        )

        # step3
        Fd2s = Settings["Fd2s"]
        if Fd2s == "Linear":
            TransCombineAllVlues = 1 - CombineAllValues

            SimArray = {}
            for i in range(len(MinMaxEudAllKeys)):
                assert MinMaxEudAllValues[i] == MinMaxEudDistArray[MinMaxEudAllKeys[i]]
                assert CosineSimAllValues[i] == CosineSimSimArray[CosineSimAllKeys[i]]

                transcombinevalue = TransCombineAllVlues[i]
                SimArray.update({MinMaxEudAllKeys[i]: transcombinevalue})

        elif Fd2s == "LogP":
            LogPValue = Settings["LogPValue"]
            TransCombineAllVlues = CalLogP(CombineAllValues, LogPValue)

            SimArray = {}
            for i in range(len(MinMaxEudAllKeys)):
                assert MinMaxEudAllValues[i] == MinMaxEudDistArray[MinMaxEudAllKeys[i]]
                assert CosineSimAllValues[i] == CosineSimSimArray[CosineSimAllKeys[i]]

                transcombinevalue = TransCombineAllVlues[i]
                SimArray.update({MinMaxEudAllKeys[i]: transcombinevalue})

    return SimArray


def CalDiffArray(Y):
    # Only MinMaxEud is used for properties Y
    Y = Y.T
    DiffArray = CalMinMaxEuclidean(Y)
    return DiffArray


def FillMissingValueByDefault(Y):
    Y = Y.T
    total_num = Y.shape[0]
    element_num = Y.shape[1]
    assert total_num > element_num

    mean_array = []
    # var_array = []
    screened_Y = []

    for ele in range(element_num):
        screened_Y.append([])
        cur_element_vec = Y[:, ele]
        cur_element_wo_mv = []
        for x in cur_element_vec:
            if not np.isnan(x):
                cur_element_wo_mv.append(x)
        cur_element_wo_mv = np.array(cur_element_wo_mv)
        cur_element_mean = cur_element_wo_mv.mean()
        # cur_element_var = cur_element_wo_mv.var()
        mean_array.append(cur_element_mean)
        for x in cur_element_vec:
            if not np.isnan(x):
                screened_Y[ele].append(x)
            else:
                screened_Y[ele].append(cur_element_mean)

    screened_Y = np.array(screened_Y)

    return screened_Y


##################################################################
def ClassificatinoTaskDeltaCalculating(MinMaxEudDistArray, DiffArray):
    # filename = FileNames(Settings, 'FeatureMinMaxEud')
    # MinMaxEudDistArray = LoadFile(filename)
    # filename = FileNames(Settings,'DiffArray')
    # DiffArray = LoadFile(filename)

    Din = []
    Dout = []
    for key in MinMaxEudDistArray.keys():

        dist = MinMaxEudDistArray[key]
        diff = DiffArray[key]
        if diff == 0:
            Din.append(dist)
        else:
            Dout.append(dist)

    Din = np.array(Din)
    Dout = np.array(Dout)

    delta_AC = np.mean(Din)
    delta_SH = np.mean(Dout)

    return delta_AC, delta_SH


def ClassificationTaskEpsilonCalculating(Properties):
    # filename = FileNames(Settings, 'Property')
    # Properties = LoadFile(filename)

    # if (Settings['TargetTask'] == 'Tox21') or (Settings['TargetTask'] == 'Toxcast'):
    #     print(f"Filling Missing Values.")
    #     Properties = FillMissingValueByDefault(Properties)
    #     # print()

    Properties = Properties.T
    total_num = Properties.shape[0]

    # Cal Eud for each pair-wise samples
    Array = {}
    for i in range(total_num):
        for j in range(i + 1, total_num):
            assert i != j
            y1 = Properties[i]
            y2 = Properties[j]
            diff = CalEuDist(y1, y2)
            Array.update({f"{i},{j}": diff})
    # print(Array)
    values = Array.values()
    values = list(values)
    values = np.array(values)
    max_value = np.max(values)
    min_value = np.min(values)
    # print(f"max value: {max_value}")
    # print(f"min value: {min_value}")

    MinMaxEpsilon = (0.5 - min_value) / (max_value - min_value)
    return MinMaxEpsilon


def RegressionTaskDeltaEpsilonCalculating(DiffArray, DistArray):
    # filename = FileNames(Settings, 'DiffArray')
    # DiffArray = LoadFile(filename)
    # filename = FileNames(Settings, 'FeatureMinMaxEud')
    # DistArray = LoadFile(filename)

    dist_value = np.array(list(DistArray.values()))
    dist_50 = np.percentile(dist_value, 50)

    Dnear_dist = []
    Dnear_diff = []
    Dfar_dist = []
    Dfar_diff = []

    for key in DistArray.keys():
        dist = DistArray[key]
        diff = DiffArray[key]
        if dist < dist_50:
            Dnear_diff.append(diff)
            Dnear_dist.append(dist)
        else:
            Dfar_diff.append(diff)
            Dfar_dist.append(dist)

    Dnear_dist = np.array(Dnear_dist)
    Dnear_diff = np.array(Dnear_diff)
    Dfar_dist = np.array(Dfar_dist)
    Dfar_diff = np.array(Dfar_diff)

    delta_AC = np.mean(Dnear_dist)
    delta_SH = np.mean(Dfar_dist)
    epsilon_AC = np.mean(Dnear_diff)
    epsilon_SH = np.mean(Dfar_diff)

    return (delta_AC, epsilon_AC), (delta_SH, epsilon_SH)

    # return DiffArray[max_dist_pair], DiffArray[min_dist_pair]


##################################################################
def AUCCalculating(answer, label):
    assert len(answer) == len(label)
    try:
        result = roc_auc_score(y_true=label, y_score=answer)
    except:
        print(f"No Activity Difference larger than epsilon, return zero.")
        result = 0
    return result


def Metric2Calculating_Array(SimArray, DiffArray, Settings):
    epsilon = Settings["M2Epsilon"]
    total_num = len(SimArray)
    assert len(SimArray) == len(DiffArray)
    print(f"Total data point (pair-wise) of the dataset is {total_num}")
    LabelList = []
    PredList = []
    for key in SimArray.keys():
        sim = SimArray[key]
        diff = DiffArray[key]

        if diff >= epsilon:
            LabelList.append(1)
        else:
            LabelList.append(0)

        PredList.append(1 - sim)

    assert len(LabelList) == len(PredList)
    m2 = AUCCalculating(answer=PredList, label=LabelList)
    return m2


def Metric1Calculating_Array(SimArray, DiffArray, Settings):
    thresholds = Settings["Thresholds"]
    if "epsilon_AC" in thresholds.keys():
        # regression
        epsilon_AC = thresholds["epsilon_AC"]
        epsilon_SH = thresholds["epsilon_SH"]
        delta_AC = thresholds["delta_AC"]
        delta_SH = thresholds["delta_SH"]
        delta_AC_s = 1 - delta_AC
        delta_SH_s = 1 - delta_SH
    else:
        # classification
        epsilon = thresholds["epsilon"]
        delta_AC = thresholds["delta_AC"]
        delta_SH = thresholds["delta_SH"]
        delta_AC_s = 1 - delta_AC
        delta_SH_s = 1 - delta_SH
    total_num = len(SimArray)
    assert len(SimArray) == len(DiffArray)
    print(f"Total data point (pair-wise) of the dataset is {total_num}")
    R1 = []  # SH
    R4 = []  # AC
    C1 = 0
    C4 = 0
    for key in SimArray.keys():
        sim = SimArray[key]
        diff = DiffArray[key]

        if "epsilon_AC" in thresholds.keys():
            # regression
            if (diff > epsilon_AC) & (sim > delta_AC_s):
                # AC
                deviation = min(np.abs(diff - epsilon_AC), np.abs(sim - delta_AC_s))
                R4.append(deviation)
                C4 += 1
            elif (diff < epsilon_SH) & (sim < delta_SH_s):
                # SH
                deviation = min(np.abs(diff - epsilon_SH), np.abs(sim - delta_SH_s))
                R1.append(deviation)
                C1 += 1
        else:
            # classification
            if (diff > epsilon) & (sim > delta_AC_s):
                # AC
                deviation = min(np.abs(diff - epsilon), np.abs(sim - delta_AC_s))
                R4.append(deviation)
                C4 += 1
            elif (diff < epsilon) & (sim < delta_SH_s):
                # SH
                deviation = min(np.abs(diff - epsilon), np.abs(sim - delta_SH_s))
                R1.append(deviation)
                C1 += 1

    # C1 = len(R1)
    # C4 = len(R4)
    assert C1 == len(R1)
    assert C4 == len(R4)
    e = 2 * (C1 + C4) / total_num
    dbar = (np.sum(R1) + np.sum(R4)) / (C1 + C4)
    m1 = e * dbar
    return m1


def Metric3Calculating_Array(SimArray, DiffArray, Settings):
    thresholds = Settings["Thresholds"]
    if "epsilon_AC" in thresholds.keys():
        # regression
        epsilon_AC = thresholds["epsilon_AC"]
        epsilon_SH = thresholds["epsilon_SH"]
        delta_AC = thresholds["delta_AC"]
        delta_SH = thresholds["delta_SH"]
        delta_AC_s = 1 - delta_AC
        delta_SH_s = 1 - delta_SH
    else:
        # classification
        epsilon = thresholds["epsilon"]
        delta_AC = thresholds["delta_AC"]
        delta_SH = thresholds["delta_SH"]
        delta_AC_s = 1 - delta_AC
        delta_SH_s = 1 - delta_SH
    total_num = len(SimArray)
    assert len(SimArray) == len(DiffArray)
    R1 = []  # SH
    R4 = []  # AC
    C1 = 0
    C4 = 0
    for key in SimArray.keys():
        sim = SimArray[key]
        diff = DiffArray[key]

        if "epsilon_AC" in thresholds.keys():
            # regression
            if (diff > epsilon_AC) & (sim > delta_AC_s):
                # AC
                deviation = min(np.abs(diff - epsilon_AC), np.abs(sim - delta_AC_s))
                R4.append(deviation)
                C4 += 1
            elif (diff < epsilon_SH) & (sim < delta_SH_s):
                # SH
                deviation = min(np.abs(diff - epsilon_SH), np.abs(sim - delta_SH_s))
                R1.append(deviation)
                C1 += 1
        else:
            # classification
            if (diff > epsilon) & (sim > delta_AC_s):
                # AC
                deviation = min(np.abs(diff - epsilon), np.abs(sim - delta_AC_s))
                R4.append(deviation)
                C4 += 1
            elif (diff < epsilon) & (sim < delta_SH_s):
                # SH
                deviation = min(np.abs(diff - epsilon), np.abs(sim - delta_SH_s))
                R1.append(deviation)
                C1 += 1

    assert C1 == len(R1)
    assert C4 == len(R4)

    # MorganFPC1C4filename = FileNames(Settings, 'MorganFPC1C4')
    # C1C4 = LoadFile(MorganFPC1C4filename)
    C1C4 = {"C1": 50512, "C4": 36149}
    C1FP = C1C4["C1"]
    C4FP = C1C4["C4"]
    m3 = C1 / C1FP + C4 / C4FP
    return m3


def C1C4Calculating_Array(SimArray, DiffArray, Settings):
    thresholds = Settings["Thresholds"]
    if "epsilon_AC" in thresholds.keys():
        # regression
        epsilon_AC = thresholds["epsilon_AC"]
        epsilon_SH = thresholds["epsilon_SH"]
        delta_AC = thresholds["delta_AC"]
        delta_SH = thresholds["delta_SH"]
        delta_AC_s = 1 - delta_AC
        delta_SH_s = 1 - delta_SH
    else:
        # classification
        epsilon = thresholds["epsilon"]
        delta_AC = thresholds["delta_AC"]
        delta_SH = thresholds["delta_SH"]
        delta_AC_s = 1 - delta_AC
        delta_SH_s = 1 - delta_SH

    # delta = Settings['M3Delta']
    # epsilon = Settings['M3Epsilon']
    total_num = len(SimArray)
    assert len(SimArray) == len(DiffArray)
    R1 = []  # SH
    R4 = []  # AC
    C1 = 0
    C4 = 0
    for key in SimArray.keys():
        sim = SimArray[key]
        diff = DiffArray[key]

        if "epsilon_AC" in thresholds.keys():
            # regression
            if (diff > epsilon_AC) & (sim > delta_AC_s):
                # AC
                deviation = min(np.abs(diff - epsilon_AC), np.abs(sim - delta_AC_s))
                R4.append(deviation)
                C4 += 1
            elif (diff < epsilon_SH) & (sim < delta_SH_s):
                # SH
                deviation = min(np.abs(diff - epsilon_SH), np.abs(sim - delta_SH_s))
                R1.append(deviation)
                C1 += 1
        else:
            # classification
            if (diff > epsilon) & (sim > delta_AC_s):
                # AC
                deviation = min(np.abs(diff - epsilon), np.abs(sim - delta_AC_s))
                R4.append(deviation)
                C4 += 1
            elif (diff < epsilon) & (sim < delta_SH_s):
                # SH
                deviation = min(np.abs(diff - epsilon), np.abs(sim - delta_SH_s))
                R1.append(deviation)
                C1 += 1

    assert C1 == len(R1)
    assert C4 == len(R4)
    C1C4 = {"C1": C1, "C4": C4}
    return C1C4


def Metric3Calculating_Array_C1C4(SimArray, DiffArray, Settings):
    thresholds = Settings["Thresholds"]
    if "epsilon_AC" in thresholds.keys():
        # regression
        epsilon_AC = thresholds["epsilon_AC"]
        epsilon_SH = thresholds["epsilon_SH"]
        delta_AC = thresholds["delta_AC"]
        delta_SH = thresholds["delta_SH"]
        delta_AC_s = 1 - delta_AC
        delta_SH_s = 1 - delta_SH
    else:
        # classification
        epsilon = thresholds["epsilon"]
        delta_AC = thresholds["delta_AC"]
        delta_SH = thresholds["delta_SH"]
        delta_AC_s = 1 - delta_AC
        delta_SH_s = 1 - delta_SH
    total_num = len(SimArray)
    assert len(SimArray) == len(DiffArray)
    R1 = []  # SH
    R4 = []  # AC
    C1 = 0
    C4 = 0
    for key in SimArray.keys():
        sim = SimArray[key]
        diff = DiffArray[key]

        if "epsilon_AC" in thresholds.keys():
            # regression
            if (diff > epsilon_AC) & (sim > delta_AC_s):
                # AC
                deviation = min(np.abs(diff - epsilon_AC), np.abs(sim - delta_AC_s))
                R4.append(deviation)
                C4 += 1
            elif (diff < epsilon_SH) & (sim < delta_SH_s):
                # SH
                deviation = min(np.abs(diff - epsilon_SH), np.abs(sim - delta_SH_s))
                R1.append(deviation)
                C1 += 1
        else:
            # classification
            if (diff > epsilon) & (sim > delta_AC_s):
                # AC
                deviation = min(np.abs(diff - epsilon), np.abs(sim - delta_AC_s))
                R4.append(deviation)
                C4 += 1
            elif (diff < epsilon) & (sim < delta_SH_s):
                # SH
                deviation = min(np.abs(diff - epsilon), np.abs(sim - delta_SH_s))
                R1.append(deviation)
                C1 += 1

    assert C1 == len(R1)
    assert C4 == len(R4)

    # MorganFPC1C4filename = FileNames(Settings, "MorganFPC1C4")
    C1C4 = {"C1": 50512, "C4": 36149}
    # C1C4 = C1C4Calculating_Array(SimArray, DiffArray, Settings)
    C1FP = C1C4["C1"]
    C4FP = C1C4["C4"]
    C1rate = C1 / C1FP
    C4rate = C4 / C4FP
    return {"C1rate": C1rate, "C4rate": C4rate}
