import torch
import torch.nn as nn
from transformers import T5TokenizerFast, T5Config, T5ForConditionalGeneration

class TextOnlyReportGenerator(nn.Module):
    """Modèle pour étude d'ablation - texte clinique seulement"""
    
    def __init__(
        self,
        t5_model="google/flan-t5-base",
        max_txt_len=512,
    ):
        super().__init__()
        
        # Tokenizer et modèle T5
        self.tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model,
            config=t5_config,
            torch_dtype=torch.float32
        )
        
        self.max_txt_len = max_txt_len
        
        # Configurer les gradients T5 (même stratégie que le modèle multimodal)
        for i, layer in enumerate(self.t5_model.encoder.block):
            layer.requires_grad_(i >= 6)  # Unfreeze last 6/12 layers
            
        for i, layer in enumerate(self.t5_model.decoder.block):
            layer.requires_grad_(i >= 3)  # Unfreeze last 3/12 layers
            
        self._verify_initialization()

    def _verify_initialization(self):
        """Vérification de l'initialisation"""
        print("\n" + "="*50)
        print("🔍 VÉRIFICATION TEXT-ONLY MODEL")
        print("="*50)
        
        t5_params = list(self.t5_model.parameters())
        if t5_params:
            t5_count = sum(p.numel() for p in t5_params)
            print(f"✅ T5 model: {t5_count:,} paramètres")
        
        # Compter les paramètres entraînables
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"📊 Paramètres entraînables: {trainable_params:,} / {total_params:,}")
        
        print("="*50)
        print("✅ MODÈLE TEXTE-SEUL INITIALISÉ")
        print("="*50)

    def _tokenize_texts(self, texts, max_length=256, padding="max_length"):
        """Tokenize une liste de textes"""
        tokens = self.tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return tokens

    def forward(self, samples, generate=False):
        # Récupération des textes bruts
        clinical_texts = samples["clinical_text"]
        reports = samples.get("reports", [""])
        
        # Tokenization dans le modèle
        clinical_tokens = self._tokenize_texts(
            clinical_texts, 
            max_length=self.max_txt_len // 2,
            padding="max_length"
        )
        clinical_input_ids = clinical_tokens['input_ids'].to(self.device)
        clinical_attention_mask = clinical_tokens['attention_mask'].to(self.device)
        
        if generate:
            
            generated_ids = self.t5_model.generate(
                input_ids=clinical_input_ids,
                attention_mask=clinical_attention_mask,
                do_sample=False, 
                max_new_tokens=200,  
                num_beams=3,  
                temperature=0.7,  
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            predictions = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            return {"predictions": predictions}
            
        else:
            # Entraînement - tokenization des rapports cibles
            report_tokens = self._tokenize_texts(
                reports, 
                max_length=self.max_txt_len // 2,
                padding="longest"
            )
            report_input_ids = report_tokens['input_ids'].to(self.device)
            
            labels = report_input_ids.masked_fill(
                report_input_ids == self.tokenizer.pad_token_id, -100
            )
            
            # Forward avec gestion d'erreur 
            try:
                outputs = self.t5_model(
                    input_ids=clinical_input_ids,
                    attention_mask=clinical_attention_mask,
                    labels=labels,
                    return_dict=True
                )
                
                # Vérification et stabilisation de la loss
                if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                    print("🚨 Loss NaN/Inf, utilisation de fallback")
                    return {"loss": torch.tensor(2.0, device=self.device, requires_grad=True)}
                    
                return {"loss": outputs.loss}
                
            except Exception as e:
                print(f"Erreur T5: {e}")
                return {"loss": torch.tensor(2.0, device=self.device, requires_grad=True)}

    @torch.no_grad()
    def generate_report(self, samples, max_new_tokens=300, temperature=0.8):
        """Génération de rapports à partir du texte clinique seulement"""
        clinical_text = samples["clinical_text"]
        
        # Tokenization 
        clinical_tokens = self._tokenize_texts(
            clinical_text, 
            max_length=self.max_txt_len,
            padding="longest"
        )
        clinical_input_ids = clinical_tokens['input_ids'].to(self.device)
        clinical_attention_mask = clinical_tokens['attention_mask'].to(self.device)
        
        # Génération 
        generated_ids = self.t5_model.generate(
            input_ids=clinical_input_ids,
            attention_mask=clinical_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        reports = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return [self._format_report(r) for r in reports]

    def _format_report(self, raw_text: str):
        return raw_text.strip()

    @property
    def device(self):
        return next(self.parameters()).device