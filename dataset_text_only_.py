from collections import defaultdict
import torch
from torch.utils.data import Dataset
import pandas as pd

class TextOnlyDataset(Dataset):
    """Dataset pour l'étude d'ablation - texte clinique seulement"""

    def __init__(self, csv_path):
        super().__init__()
        self.data = pd.read_csv(csv_path)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        
        # Retourne uniquement les textes bruts
        clinical_text = row['clinical_text']
        diagnostic_report = row['diagnostic_report']

        return {
            'clinical_text': clinical_text,
            'reports': diagnostic_report
        }

    def __len__(self):
        return len(self.data)

class TextOnlyCollator:
    def __init__(self):
        pass
        
    def __call__(self, batch):
        inputs = defaultdict(list)
        
        for sample in batch:
            inputs['clinical_text'].append(sample['clinical_text'])
            inputs['reports'].append(sample['reports'])
        
        return inputs