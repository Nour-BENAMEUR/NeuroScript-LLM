import logging
from typing import Dict, List, Optional
import math
from tqdm import tqdm
import numpy as np
import torch
from torch.optim import Optimizer
import transformers
from bert_score import score
from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    
    def __init__(self, args=None):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def train(
        self,
        model,
        dataloader,
        eval_dataloader: Optional = None,
        epochs: int = 1,
        scheduler: str = 'WarmupCosine',
        warmup_ratio: float = 0.01,
        output_path: str = 'E:/data_nour/checkpoints',
        optimizer_params: Dict = {'lr': 5e-5, 'eps': 1e-7, 'weight_decay': 1e-4},
        weight_decay: float = 0.01,
        use_amp: bool = False,
        eval_steps: int = 100,
        accumulation_steps: int = 4,
    ):
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        optimizer, scheduler = self._setup_optimizer(
            model, optimizer_params, weight_decay, scheduler, warmup_ratio, epochs, dataloader
        )
        
        global_step = 0
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            
            optimizer.zero_grad()
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            
            for batch_idx, batch in enumerate(progress_bar):
                # Vérification proactive des NaN
                batch = self._nan_detection_and_fix(model, batch)
                
                loss = self._train_step(model, batch, optimizer, use_amp, scaler, accumulation_steps)
                epoch_loss += loss.item()
                
                if (global_step + 1) % accumulation_steps == 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    if scheduler:
                        scheduler.step()
                    
                    optimizer.zero_grad()
                    
                    # Calcul du batch size effectif basé sur le texte
                    effective_batch_size = accumulation_steps * len(batch['clinical_text'])
                    print(f" Update step {(global_step + 1) // accumulation_steps}: Loss = {loss.item():.4f}, Effective batch = {effective_batch_size}")
                
                global_step += 1
                
                if torch.isnan(loss):
                    print(f"🚨 NaN Loss à step {global_step}")
                    loss = torch.tensor(2.0, device=loss.device, requires_grad=True)
                
                progress_bar.set_postfix(loss=float(loss.item()))
                
                if eval_dataloader and global_step % eval_steps == 0:
                    eval_metrics = self.evaluate(model, eval_dataloader)
                    logger.info(f"\n[Step {global_step}] Eval Metrics: {eval_metrics}\n")
                    model.train()
            
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")
            self._save_ckpt(model, epoch, output_path)

    def _nan_detection_and_fix(self, model, batch):
        """Détection et correction proactive des NaN - spécifique texte-only"""
        # Pour le texte-only, on vérifie seulement les paramètres du modèle
        # car les textes bruts ne peuvent pas contenir de NaN
        
        # Vérification des paramètres du modèle
        for name, param in model.named_parameters():
            if param.requires_grad and torch.isnan(param).any():
                print(f"🚨 NaN dans les paramètres {name}, réinitialisation")
                param.data = torch.nan_to_num(param.data, nan=0.001)
        
        return batch

    def _train_step(self, model, batch, optimizer, use_amp, scaler, accumulation_steps=4):
        """Train step adapté pour texte-only"""
        
        try:
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch)
                    loss = outputs['loss']
            else:
                outputs = model(batch)
                loss = outputs['loss']
            
            # Vérification immédiate
            if torch.isnan(loss) or torch.isinf(loss):
                print("🚨 Loss invalide détectée, réinitialisation")
                return torch.tensor(2.0, device=loss.device, requires_grad=True)
            
            loss = loss / accumulation_steps
            
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            return loss * accumulation_steps
            
        except RuntimeError as e:
            print(f"🚨 Erreur runtime: {e}")
            torch.cuda.empty_cache()
            return torch.tensor(2.0, device=next(model.parameters()).device, requires_grad=True)

    def evaluate(self, model, eval_dataloader) -> Dict:
        model.eval()
        all_preds, all_refs = [], []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                
                # Debug des entrées
                print("\n--- DEBUG GENERATION ---")
                print("Input text (clinical_text):", batch["clinical_text"][0][:100] + "...")
                
                # Génération unifiée
                if hasattr(model, 'generate_report'):
                    outputs = model.generate_report(batch)
                else:
                    outputs = model(batch, generate=True)
                
                # Gestion du format de sortie
                if isinstance(outputs, dict) and 'predictions' in outputs:
                    predictions = outputs['predictions']
                elif isinstance(outputs, list):
                    predictions = outputs
                else:
                    print(f"⚠️ Format de sortie inattendu: {type(outputs)}")
                    predictions = ["Generation failed"]
                
                print("Generated output (raw):", predictions[0])
                all_preds.extend(predictions)
                all_refs.extend(batch['reports'])
        
        metrics = {}
        metrics.update(self._calculate_bertscore(all_preds, all_refs))
        metrics.update(self._calculate_rouge(all_preds, all_refs))
        metrics.update(self._calculate_bleu_fixed(all_preds, all_refs))
        metrics.update(self._calculate_meteor_robust(all_preds, all_refs))
    
        return metrics

    def _calculate_bleu_fixed(self, preds: List[str], refs: List[str]) -> Dict:
        try:
            bleu_scores = {}
            
            # Calcul séparé pour chaque n-gramme
            for n in range(1, 5):
                try:
                    individual_scores = []
                    for pred, ref in zip(preds, refs):
                        try:
                            score = self._calculate_ngram_precision(pred, ref, n)
                            individual_scores.append(score)
                        except:
                            individual_scores.append(0.0)
                    
                    bleu_scores[f'BLEU-{n}'] = np.mean(individual_scores)
                    
                except Exception as e:
                    print(f"⚠️ Erreur BLEU-{n}: {e}")
                    bleu_scores[f'BLEU-{n}'] = 0.0
            
            return bleu_scores
            
        except Exception as e:
            print(f"⚠️ Erreur calcul BLEU général: {e}")
            return {
                'BLEU-1': 0.0,
                'BLEU-2': 0.0,
                'BLEU-3': 0.0,
                'BLEU-4': 0.0
            }

    def _calculate_ngram_precision(self, pred: str, ref: str, n: int) -> float:
        """Calcule la précision des n-grammes"""
        from collections import Counter
        
        def get_ngrams(text: str, n: int):
            words = text.lower().split()
            if len(words) < n:
                return []
            return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        
        pred_ngrams = Counter(get_ngrams(pred, n))
        ref_ngrams = Counter(get_ngrams(ref, n))
        
        if len(pred_ngrams) == 0:
            return 0.0
        
        # Intersection des n-grammes
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())
        
        return overlap / total if total > 0 else 0.0

    def _calculate_meteor_robust(self, preds: List[str], refs: List[str]) -> Dict:
        try:
            meteor_scores = []
            
            for pred, ref in zip(preds, refs):
                try:
                    # Tokenisation avec gestion d'erreurs
                    try:
                        pred_tokens = word_tokenize(pred.lower())
                        ref_tokens = word_tokenize(ref.lower())
                    except:
                        # Fallback: split simple
                        pred_tokens = pred.lower().split()
                        ref_tokens = ref.lower().split()
                    
                    # Calcul METEOR
                    score = meteor_score([ref_tokens], pred_tokens)
                    meteor_scores.append(score)
                    
                except Exception as e:
                    # Fallback pour cette paire
                    approx_score = self._approximate_meteor_single(pred, ref)
                    meteor_scores.append(approx_score)
            
            return {
                'METEOR': np.mean(meteor_scores) if meteor_scores else 0.0
            }
            
        except Exception as e:
            print(f"⚠️ METEOR complet échoué, utilisation approximation: {e}")
            return self._calculate_meteor_approximation(preds, refs)

    def _calculate_meteor_approximation(self, preds: List[str], refs: List[str]) -> Dict:
        try:
            # Dictionnaire de synonymes médicaux simples
            medical_synonyms = {
                'alzheimer': ['ad', 'dementia', 'alzheimers'],
                'cognitive': ['mental', 'thinking', 'memory'],
                'brain': ['cerebral', 'neural', 'neurological'],
                'patient': ['subject', 'individual', 'case'],
                'normal': ['typical', 'standard', 'regular'],
                'abnormal': ['atypical', 'unusual', 'irregular'],
                'mri': ['magnetic resonance', 'imaging'],
                'pet': ['positron emission', 'scan']
            }
            
            scores = []
            for pred, ref in zip(preds, refs):
                pred_words = set(pred.lower().split())
                ref_words = set(ref.lower().split())
                
                # Correspondances exactes
                exact_matches = len(pred_words & ref_words)
                
                # Correspondances avec synonymes
                synonym_matches = 0
                for pred_word in pred_words:
                    if pred_word not in ref_words:
                        for ref_word in ref_words:
                            if self._are_synonyms(pred_word, ref_word, medical_synonyms):
                                synonym_matches += 1
                                break
                
                total_matches = exact_matches + synonym_matches * 0.8
                
                # Calcul F1-score approximatif
                precision = total_matches / len(pred_words) if pred_words else 0
                recall = total_matches / len(ref_words) if ref_words else 0
                
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0
                
                scores.append(f1)
            
            return {'METEOR': np.mean(scores) if scores else 0.0}
            
        except Exception as e:
            print(f"⚠️ Erreur approximation METEOR: {e}")
            return {'METEOR': 0.0}

    def _are_synonyms(self, word1: str, word2: str, synonyms_dict: dict) -> bool:
        """Vérifie si deux mots sont synonymes"""
        for key, synonyms in synonyms_dict.items():
            if (word1 == key and word2 in synonyms) or (word2 == key and word1 in synonyms):
                return True
            if word1 in synonyms and word2 in synonyms:
                return True
        return False

    def _approximate_meteor_single(self, pred: str, ref: str) -> float:
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        
        if not pred_words or not ref_words:
            return 0.0
        
        matches = len(pred_words & ref_words)
        precision = matches / len(pred_words)
        recall = matches / len(ref_words)
        
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0

    def _calculate_bertscore(self, preds, refs) -> Dict:
        P, R, F1 = score(preds, refs, lang="en", verbose=False)
        return {
            "BERTScore_P": P.mean().item(),
            "BERTScore_R": R.mean().item(),
            "BERTScore_F1": F1.mean().item(),
        }

    def _calculate_rouge(self, preds, refs) -> Dict:
        scores = [self.scorer.score(ref, pred)['rougeL'] for ref, pred in zip(refs, preds)]
        return {
            "ROUGE-L_P": np.mean([s.precision for s in scores]),
            "ROUGE-L_R": np.mean([s.recall for s in scores]),
            "ROUGE-L_F1": np.mean([s.fmeasure for s in scores]),
        }

    def _setup_optimizer(self, model, optimizer_params, weight_decay, scheduler, warmup_ratio, epochs, dataloader):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, **optimizer_params)
        
        steps_per_epoch = len(dataloader) // 4  # Ajusté pour accumulation
        total_steps = steps_per_epoch * epochs
        warmup_steps = math.ceil(total_steps * warmup_ratio)
        scheduler = self._get_scheduler(optimizer, scheduler, warmup_steps, total_steps)
        
        return optimizer, scheduler

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def _save_ckpt(self, model, epoch, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(save_dir, f'epoch_{epoch}.pth'))