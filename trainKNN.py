import sys
import Model.KnnModel as model 
sys.path.insert(1, '../Data/')
#import DataPreprocessing as dpp
import DataPreprocessing as datacleaner

RightEyeAiman_path = "Data/OpenBCISession_2020-10-11_16-58-40-AIMAN-RIGHT-EYE/OpenBCI-RAW-2020-10-11_16-59-03.txt"
RightEyeAiman_label_path = "DataLables/aimanRightEye"

LeftEyeAiman_path = "Data/OpenBCISession_2020-10-11_16-33-50-AIMAN-LEFT-EYE/OpenBCI-RAW-2020-10-11_16-38-59.txt"
LeftEyeAiman_label_path = "DataLables/aimanLeftEye"


RightFootShikhar_path = "Data/OpenBCISession_2020-10-11_16-26-10-SHIKHAR-RIGHT-FOOT/OpenBCI-RAW-2020-10-11_16-26-50.txt"
RightFootShikhar_label_path = "DataLables/shikharRightFoot"

LeftFootShikhar_path = "Data/OpenBCISession_2020-10-11_16-00-28-SHIKHAR-LEFT-FOOT/OpenBCI-RAW-2020-10-11_16-01-30.txt"
LeftFootShikhar_label_path = "DataLables/shikharLeftFoot"


# RightFoot_observations = dpnew.getObservationSet(RightFootShikhar_path, RightFootShikhar_label_path, 1000, [0,1,2], 'R_FOOT')
# LeftFoot_observations = dpnew.getObservationSet(LeftFootShikhar_path, LeftFootShikhar_label_path, 1000, [0,1,2], 'L_FOOT')
REAiman_observations = datacleaner.getObservationSet(RightEyeAiman_path, RightEyeAiman_label_path, 1000, [0,1,2])
LEAiman_observations = datacleaner.getObservationSet(LeftEyeAiman_path, LeftEyeAiman_label_path, 1000, [0,1,2])



knn = model.KNN(3)
knn.train([REAiman_observations, LEAiman_observations])
knn.saveModel("knnworkingmodel")
knn.loadModel('/Users/shikhar/Desktop/NNModeling/BrainwaveClassifier/knnworkingmodel')
knn.classify(REAiman_observations)


