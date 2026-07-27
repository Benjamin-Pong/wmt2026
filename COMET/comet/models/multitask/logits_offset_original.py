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
        def logits_adjustment(logits:torch.Tensor):
            '''
            Offset logits using
            '''
            global_error_stats = self.global_error_stats.to(logits.device)
            return logits - (0.8 * torch.log(global_error_stats))
        
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
            adjusted_logits_global = logits_adjustment(model_output.logits.clone())
            subword_probs = nn.functional.softmax(adjusted_logits_global, dim=2)[
                :, :seq_len, :
            ]

            adjusted_logits_global = adjusted_logits_global[:, :seq_len, :]
            
            trimmed_subwords, trimmed_logits, t_tokens, token_ids = trim(subword_probs, adjusted_logits_global, mt_offsets, input_ids, self.encoder.tokenizer)
            error_spans = self.decode(
                subword_probs, batch[0]["input_ids"], batch[0]["mt_offsets"]
            )
            mqm_scores = _compute_mqm_from_spans(error_spans)
            final_scores = (
                regression_score * sum(self.score_weights[:3])
                + mqm_scores.to(regression_score.device) * self.score_weights[3]
            )
            batch_prediction = Prediction(
                scores=final_scores,
                metadata=Prediction(
                    src_scores=regression_score,
                    mqm_scores=mqm_scores,
                    error_spans=error_spans,
                    subword_probs = trimmed_subwords,
                    logits = trimmed_logits,
                    tokens=t_tokens,
                    token_ids = token_ids
                ),
            )
        return batch_prediction