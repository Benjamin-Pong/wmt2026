from .xcomet_metric import XCOMETMetric
import torch
from typing import Dict, List, Optional, Union
from comet.models.utils import Prediction
import torch.nn.functional as F

class XCOMETMetricLayer(XCOMETMetric):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward function.

        Args:
            input_ids (torch.Tensor): Input sequence.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (Optional[torch.Tensor], optional): Token type ids for
                BERT-like models. Defaults to None.

        Raises:
            Exception: Invalid model word/sent layer if self.{word/sent}_layer are not
                valid encoder model layers .

        Returns:
            Dict[str, torch.Tensor]: Sentence scores and word-level logits (if
                word_level_training = True)
        """
        encoder_out = self.encoder(
            input_ids, attention_mask, token_type_ids=token_type_ids
        )

        # Word embeddings used for the word-level classification task
        if self.word_level:
            if (
                isinstance(self.hparams.word_layer, int)
                and 0 <= self.hparams.word_layer < self.encoder.num_layers
            ):
                last_n_layers = encoder_out["all_layers"][:]

                #normalize each layer
                #last_n_layers = [F.normalize(layer, p=2, dim=-1) for layer in last_n_layers]
                stacked = torch.stack(last_n_layers, dim=0)
                avg = torch.mean(stacked, dim=0)
                #wordemb = encoder_out["all_layers"][self.hparams.word_layer]
                wordemb= avg
            else:
                raise Exception(
                    "Invalid model word layer {}.".format(self.hparams.word_layer)
                )

        # embeddings used for the sentence-level regression task
        if self.layerwise_attention:
            embeddings = self.layerwise_attention(
                encoder_out["all_layers"], attention_mask
            )
        elif (
            isinstance(self.hparams.sent_layer, int)
            and 0 <= self.hparams.sent_layer < self.encoder.num_layers
        ):
            embeddings = encoder_out["all_layers"][self.hparams.sent_layer]
        else:
            raise Exception(
                "Invalid model sent layer {}.".format(self.hparams.word_layer)
            )
        sentemb = embeddings[:, 0, :] # We take the CLS token as sentence-embedding
        
        if self.word_level:
            sentence_output = self.estimator(sentemb)
            word_output = self.hidden2tag(wordemb)
            return Prediction(score=sentence_output.view(-1), logits=word_output)

        return Prediction(score=self.estimator(sentemb).view(-1))
