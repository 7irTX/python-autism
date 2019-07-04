#autism
class Diagnosis_class:
    speaker_ID = None
    features = None
    label = None

    def __init__(self,label,speaker_ID,features, gender):
        self.speaker_ID = speaker_ID
        self.label = label
        self.features = features
        self.gender = gender #1 = male, 2 = female
        