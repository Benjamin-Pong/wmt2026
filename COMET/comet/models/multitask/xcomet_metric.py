# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
XCOMET Metric
==============
    eXplainable Metric is a multitask metric that performs error span detection along with
    sentence-level regression. It can also be used for QE (reference-free evaluation).
"""

from typing import Dict, List, Optional, Union

import torch
from torch import nn

from comet.models.multitask.unified_metric import UnifiedMetric
from comet.models.utils import Prediction
from comet.modules import FeedForward


class XCOMETMetric(UnifiedMetric):
    """eXplainable COMET is same has Unified Metric but overwrites predict function.
    This way we can control better for the models inference.

    To cast back XCOMET models into UnifiedMetric (and vice-versa) we can simply run
    model.__class__ = UnifiedMetric

    """

    def __init__(
        self,
        nr_frozen_epochs: Union[float, int] = 0.3,
        keep_embeddings_frozen: bool = True,
        optimizer: str = "AdamW",
        warmup_steps: int = 0,
        encoder_learning_rate: float = 1.0e-06,
        learning_rate: float = 3.66e-06,
        layerwise_decay: float = 0.983,
        encoder_model: str = "XLM-RoBERTa-XL",
        pretrained_model: str = "facebook/xlm-roberta-xl",
        sent_layer: Union[str, int] = "mix",
        layer_transformation: str = "sparsemax",
        layer_norm: bool = False,
        word_layer: int = 36,
        loss: str = "mse",
        dropout: float = 0.1,
        batch_size: int = 4,
        train_data: List[str] = [],
        validation_data: List[str] = [],
        hidden_sizes: List[int] = [2560, 1280],
        activations: str = "Tanh",
        final_activation: Optional[str] = None,
        word_level_training: bool = True,
        error_labels: List[str] = ["minor", "major", "critical"],
        loss_lambda: float = 0.055,
        cross_entropy_weights: Optional[List[float]] = [0.08, 0.486, 0.505, 0.533],
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super(UnifiedMetric, self).__init__(
            nr_frozen_epochs=nr_frozen_epochs,
            keep_embeddings_frozen=keep_embeddings_frozen,
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            encoder_learning_rate=encoder_learning_rate,
            learning_rate=learning_rate,
            layerwise_decay=layerwise_decay,
            encoder_model=encoder_model,
            pretrained_model=pretrained_model,
            layer=sent_layer,
            layer_transformation=layer_transformation,
            layer_norm=layer_norm,
            loss=loss,
            dropout=dropout,
            batch_size=batch_size,
            train_data=train_data,
            validation_data=validation_data,
            class_identifier="xcomet_metric",
            load_pretrained_weights=load_pretrained_weights,
            local_files_only=local_files_only,
        )
        self.estimator = FeedForward(
            in_dim=self.encoder.output_units,
            hidden_sizes=self.hparams.hidden_sizes,
            activations=self.hparams.activations,
            dropout=self.hparams.dropout,
            final_activation=self.hparams.final_activation,
        )
        assert error_labels == ["minor", "major", "critical"]
        self.hparams.input_segments = ["mt", "src", "ref"]
        self.word_level = True
        self.encoder.labelset = self.label_encoder
        self.hidden2tag = nn.Linear(self.encoder.output_units, self.num_classes)
        self.input_tags = False  # unused

        # By default 3rd input [mt:src:ref] has 50% weight,
        # 2nd input [mt:ref] 33% and 1st input [mt:src] has 16%
        self.input_weights_spans = torch.tensor([0.1667, 0.3333, 0.5])

        # The final score is a weighted average between different scores.
        # First weight is for [mt:src], second for [mt:ref], third for [mt:src:ref] and
        # last weight is for MQM computed score.
        self.score_weights = [0.12, 0.33, 0.33, 0.22]

        # This is None by default and we will use argmax during decoding yet, to control over
        # precision and recall we can set it to another value.
        self.decoding_threshold = None

        self.init_losses()
        self.save_hyperparameters()

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> Prediction:
        """PyTorch Lightning predict_step

        Args:
            batch (Dict[str, torch.Tensor]): The output of your prepare_sample function
            batch_idx (Optional[int], optional): Integer displaying which batch this is
                Defaults to None.
            dataloader_idx (Optional[int], optional): Integer displaying which
                dataloader this is. Defaults to None.

        Returns:
            Prediction: Model Prediction
        """

        def _compute_mqm_from_spans(error_spans):
            scores = []
            for sentence_spans in error_spans:
                sentence_score = 0
                for annotation in sentence_spans:
                    if annotation["severity"] == "minor":
                        sentence_score += 1
                    elif annotation["severity"] == "major":
                        sentence_score += 5
                    elif annotation["severity"] == "critical":
                        sentence_score += 10

                if sentence_score > 25:
                    sentence_score = 25

                scores.append(sentence_score)

            # Rescale between 0 and 1
            scores = (torch.tensor(scores) * -1 + 25) / 25
            return scores
        def trim(subwords:torch.Tensor, logits:torch.Tensor, mt_offsets:torch.Tensor, input_ids:torch.Tensor, tokenizer):
            '''
            helper function that trims subword_probs,logits, and the actual tokens for qualitative inspection
            outputs subword_probs containing tensors corresponding to mt segment lengths
            outputs logits containing tensors corresponding to mt segment lengths
            '''
            trimmed_subwords: List[List] = [] 
            trimmed_logits: List[List] = []
            tokens: List[str] = []
            token_ids = []
            for i in range(len(mt_offsets)): #recall that each idx in a batch is a system prediction.
                mt_length = len(mt_offsets[i]) #batch dimension =1 
                curr_subword = subwords[i][:mt_length].tolist()
                curr_logits = logits[i][:mt_length].tolist()
                curr_token_ids = input_ids[i][:mt_length].tolist()
                curr_tokens = tokenizer.convert_ids_to_tokens(curr_token_ids)
        
                trimmed_logits.append(curr_logits)
                trimmed_subwords.append(curr_subword)
                tokens.append(curr_tokens)
                token_ids.append(curr_token_ids)
                assert subwords[i][:mt_length].shape == logits[i][:mt_length].shape \
               
                assert len(curr_subword)==len(curr_logits)==len(curr_tokens)==len(curr_token_ids \
                                                                                  )
            return trimmed_subwords, trimmed_logits, tokens, token_ids
        def compute_average_shannon_entropy(curr_span_probabilities, mt_mask, eps: float = 1e-12) -> torch.Tensor:
            '''
            Computes entropy value per subword token, and returns the mean over all tokens
            '''
            log_probs = torch.log(curr_span_probabilities + eps)
            entropy = -torch.sum(curr_span_probabilities * log_probs, dim=-1)
            avg_ent = (entropy*mt_mask).sum(dim=1) / mt_mask.sum(dim=1)
            assert not torch.isnan(avg_ent).any(), f"NaN in avg_conf: {avg_ent}"
            return avg_ent

          
        def compute_average_confidence(subword_probs, mt_mask):
            conf = subword_probs.max(dim=-1).values          # [B, S]
            avg_conf = (conf * mt_mask).sum(dim=1) / mt_mask.sum(dim=1)
            assert not torch.isnan(avg_conf).any(), f"NaN in avg_conf: {avg_conf}"
            return avg_conf
        # XCOMET is suposed to be used with a reference thus 3 different inputs.
        if len(batch) == 3:
            predictions = [self.forward(**input_seq) for input_seq in batch]
            # Regression scores are weighted with self.score_weights
            regression_scores = torch.stack(
                [
                    torch.where(pred.score > 1.0, 1.0, pred.score) * w
                    for pred, w in zip(predictions, self.score_weights[:3])
                ],
                dim=0,
            ).sum(dim=0)
            mt_mask = batch[0]["label_ids"] != -1
            mt_length = mt_mask.sum(dim=1)
            seq_len = mt_length.max()

            # Weighted average of the softmax probs along the different inputs.
            '''
            Compute logits adjustments here
            '''
            subword_probs = [
                nn.functional.softmax(o.logits, dim=2)[:, :seq_len, :] * w
                for w, o in zip(self.input_weights_spans, predictions)
            ]
            subword_probs = torch.sum(torch.stack(subword_probs), dim=0)
            error_spans = self.decode(
                subword_probs, batch[0]["input_ids"], batch[0]["mt_offsets"]
            )
            mqm_scores = _compute_mqm_from_spans(error_spans)
            final_scores = (
                regression_scores
                + mqm_scores.to(regression_scores.device) * self.score_weights[3]
            )
            batch_prediction = Prediction(
                scores=final_scores,
                metadata=Prediction(
                    src_scores=predictions[0].score,
                    ref_scores=predictions[1].score,
                    unified_scores=predictions[2].score,
                    mqm_scores=mqm_scores,
                    error_spans=error_spans,
                    subword_probs = subword_probs,
                    logits = predictions[0].logits

                ),
            )

        # XCOMET if reference is not available we fall back to QE model.
        else:
            model_output = self.forward(**batch[0])
            regression_score = torch.where(
                model_output.score > 1.0, 1.0, model_output.score
            )
            mt_mask = batch[0]["label_ids"] != -1
            mt_length = mt_mask.sum(dim=1)
            seq_len = mt_length.max()
            mt_offsets = batch[0]["mt_offsets"]
            input_ids = batch[0]["input_ids"]
            subword_probs = nn.functional.softmax(model_output.logits, dim=2)[
                :, :seq_len, :
            ]
            logits = model_output.logits[:, :seq_len, :]
            #trimmed_subwords, trimmed_logits, t_tokens, token_ids = trim(subword_probs, logits, mt_offsets, input_ids, self.encoder.tokenizer)
            error_spans = self.decode(
                subword_probs, batch[0]["input_ids"], batch[0]["mt_offsets"]
            )
            mqm_scores = _compute_mqm_from_spans(error_spans)
            final_scores = (
                regression_score * sum(self.score_weights[:3])
                + mqm_scores.to(regression_score.device) * self.score_weights[3]
            )
            mt_mask = mt_mask.unsqueeze(-1).to(logits.dtype)
            trimmed_logits = logits * mt_mask
            trimmed_subword_probs = subword_probs * mt_mask

            avg_confidence = compute_average_confidence(subword_probs, mt_mask)
            avg_entropy = compute_average_shannon_entropy(subword_probs, mt_mask)
           
            batch_prediction = Prediction(
                scores=final_scores,
                metadata=Prediction(
                    src_scores=regression_score,
                    mqm_scores=mqm_scores,
                    avg_confidence=avg_confidence,
                    avg_entropy=avg_entropy,
                    error_spans=error_spans,
                    subword_probs=trimmed_subword_probs,
                    logits=trimmed_logits,
                ),
            )
        return batch_prediction

    def decode(
        self,
        subword_probs: torch.Tensor,
        input_ids: torch.Tensor,
        mt_offsets: torch.Tensor,
    ) -> List[Dict]:
        """Decode error spans from subwords.

        Args:
            subword_probs (torch.Tensor): probabilities of each label for each subword.
            input_ids (torch.Tensor): input ids from the model.
            mt_offsets (torch.Tensor): subword offsets.

        Return:
            List with of dictionaries with text, start, end, severity and a
            confidence score which is the average of the probs for that label.
        """
        decoded_output = []
        for i in range(len(mt_offsets)):
            seq_len = len(mt_offsets[i])
            error_spans, in_span, span = [], False, {}
            for token_id, probs, token_offset in zip(
                input_ids[i, :seq_len], subword_probs[i][:seq_len], mt_offsets[i]
            ):
                if self.decoding_threshold:
                    if torch.sum(probs[1:]) > self.decoding_threshold:
                        probability, label_value = torch.topk(probs[1:], 1)
                        label_value += 1  # offset from removing label 0
                    else:
                        # This is just to ensure same format but at this point
                        # we will only look at label 0 and its prob
                        probability, label_value = torch.topk(probs[0], 1)
                else:
                    probability, label_value = torch.topk(probs, 1)

                # Some torch versions topk returns a shape 1 tensor with only
                # a item inside
                label_value = (
                    label_value.item()
                    if label_value.dim() < 1
                    else label_value[0].item()
                )
                label = self.label_encoder.ids_to_label.get(label_value)
                # Label set:
                # O I-minor I-major
                # Begin of annotation span
                if label.startswith("I") and not in_span:
                    in_span = True
                    span["tokens"] = [
                        token_id,
                    ]
                    span["severity"] = label.split("-")[1]
                    span["offset"] = list(token_offset)
                    span["confidence"] = [
                        probability,
                    ]
                    span['probs']=[probs]

                # Inside an annotation span
                elif label.startswith("I") and in_span:
                    span["tokens"].append(token_id)
                    span["confidence"].append(probability)
                    # Update offset end
                    span["offset"][1] = token_offset[1]
                    span["probs"].append(probs)

                # annotation span finished.
                elif label == "O" and in_span:
                    error_spans.append(span)
                    in_span, span = False, {}

            sentence_output = []
            def compute_average_shannon_entropy(curr_span_probabilities, eps: float = 1e-12) -> torch.Tensor:
                '''
                Computes entropy value per subword token, and returns the mean over all tokens
                '''
                log_probs = torch.log(curr_span_probabilities + eps)
                entropy = -torch.sum(curr_span_probabilities * log_probs, dim=-1)
                return entropy.mean()

            for span in error_spans:
                curr_span_probs = torch.stack(span['probs'])
                entropy = compute_average_shannon_entropy(curr_span_probs)
                
                assert not torch.isnan(entropy).any(), f"NaN in span entropy: {entropy}"
                #print(entropy)
                sentence_output.append(
                    {
                        "text": self.encoder.tokenizer.decode(span["tokens"]),
                        "confidence": torch.concat(span["confidence"]).mean().item(),
                        "entropy": entropy.item(),
                        "severity": span["severity"],
                        "start": span["offset"][0],
                        "end": span["offset"][1],
                    }
                )
            decoded_output.append(sentence_output)
        return decoded_output