import torch

from .transformer_IO_resource import SpanResource
from .trasnformer_manager import TransformersManager
from IO_relation import RelModel


def IORelation_load_func(model_path, checkpoint_file, encoder):
    state_dict = torch.load(model_path + checkpoint_file)
    model = RelModel(encoder)
    print(model.load_state_dict(state_dict))
    return model.eval()



class TransformerIORelationResource(SpanResource):
    """
    BoolQ requires spacy NLP to preprocess the questions into statement
    """

    def __init__(self, *args, **kwargs):
        super().__init__(model_name="IO", *args, **kwargs)

    def get_outputs_from_logits(self, logits, tokens, *args, **kwargs):
        wh = ['head', 'tail']
        Batch_size = len(logits)
        #logits_ = logits.reshape(Batch_size, -1, 7, 2)
        Answers = []
        probs = []
        
        for token, logit in zip(tokens, logits):
            logit_ = logit.reshape(-1, 2, 2)
            IDX = logit_.argmax(axis = -1)
            IDX = torch.transpose(IDX, 0, 1)
            answers = {}
            for i, index in enumerate(IDX):
                indices = self.contigous_spans(index)
                answer = []
                for idx in indices:
                    idx_ = torch.cat(( idx, torch.zeros(len(token) - len(idx))))
                    ans = self.model_manager.decode(token[idx_.bool()])
                    #if isinstance(answer, list):
                        # Returning list means span contains answer, override to empty string.
                        # Doing this will save us time for encoding context only
                    
                    answer.append(ans.strip())
                
                answers[wh[i]] = answer
            
            Answers.append(answers)
            
            
            prob = torch.softmax(logit, axis=-1, dtype=torch.float)
            probs.append(
                torch.max(prob, axis = -1)[0].mean().item()
            )
            # get probabilities            

           

       

        outputs = {}
        outputs["answers"] = Answers

        if kwargs.get("return_logits", False):
            outputs["logits"] = [logit.numpy().tolist() for logit in logits]

        if kwargs.get("return_probs", False):
            outputs["probs"] = probs

        if kwargs.get("return_bytes", False):
            outputs["start_bytes"] = start_bytes
            outputs["end_bytes"] = end_bytes

        return outputs


    def contigous_spans(self, a):
        flag = False
        idx = []
        for i in range(len(a)):
            if not flag and a[i] == 1:
                idx.append([i, i] )
                flag = True
            if flag and a[i] == 1:
                idx[-1][-1] = i+1
            if flag and a[i] ==0:
                flag = False
        
        ANS = []
        for x in idx:
            v = torch.zeros(len(a))
            v[x[0]:x[1]] = torch.ones(x[1] - x[0])
            ANS.append(v)
        return ANS




class TransformerIORelationManager(TransformersManager):
    """
    Dummy class for better logging
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manager(cls, *args, **kwargs):
        return super().get_manager(
            load_func=IORelation_load_func,
            head="IORelation",
            *args,
            **kwargs,
        )