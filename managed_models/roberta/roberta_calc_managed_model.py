import numpy as np
import torch

import roberta.span  # noqa: F401
from managed_models.roberta.roberta_managed_model import RobertaManagedModel

# this import is to make sure the span task is registered in fairseq


class RobertaCalcManagedModel(RobertaManagedModel):
    def predict(self, **kwargs):
        """
        predict the given task
        """
        assert (
            "tokens" in kwargs and "head" in kwargs
        ), f"Roberta requires tokens and head for prediction, got {kwargs}"

        if isinstance(kwargs["tokens"], np.ndarray):
            kwargs["tokens"] = torch.from_numpy(kwargs["tokens"])

        assert isinstance(kwargs["tokens"], torch.Tensor)

        with torch.no_grad():
            features = self.model.extract_features(kwargs["tokens"])
            # logits = self.model.predict(**kwargs)
            span_logits = (
                self.model.model.classification_heads["span"](features).detach().cpu()
            )
            opt_logits = (
                self.model.model.classification_heads["operation"](features)
                .detach()
                .cpu()
            )

        return zip(span_logits, opt_logits)
