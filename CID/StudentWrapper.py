import torch
from CID.models.util import Reg


class StudentFowardWrapper:
    def __init__(self, student_model, n_cls, uses_kd, hint_layer):
        self.student_model = student_model
        self.n_cls = n_cls
        self.uses_kd = uses_kd
        self.hint_layer_nr = hint_layer
        self.__init_fc()

    def __init_fc(self): 
        ex_data = torch.rand(2, 3, 32, 32)
        feat_s, _ = self.student_model(ex_data, is_feat = True)
        _, Cs_h = feat_s[self.hint_layer_nr].shape
        self.hint_shape = Cs_h
        if self.uses_kd:
            self.model_fc = Reg(Cs_h * 2, self.n_cls)
        else: 
            self.model_fc = self.student_model.fc 
    
    def get_uses_kd(self): 
        return self.uses_kd
    
    def get_model_fc(self): 
        return self.model_fc
    

    def calculate_prediction(self, input, is_feat, preact):
        raise NotImplementedError("TODO")



            
