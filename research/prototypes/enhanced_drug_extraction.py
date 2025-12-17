# Enhanced Drug Extraction with NER
# Research Prototype for improving drug identification accuracy

import spacy
import re
import pandas as pd
from transformers import pipeline
from typing import List, Dict, Tuple
import numpy as np

class EnhancedDrugExtractor:
    """
    Enhanced drug extraction using multiple NER approaches and ensemble methods.
    This prototype implements research findings to improve drug identification
    accuracy beyond simple pattern matching.
    """
    
    def __init__(self, use_ensemble=True):
        self.use_ensemble = use_ensemble
        
        # Load models
        try:
            self.spacy_model = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.spacy_model = None
            
        # BioBERT for biomedical NER (research implementation)
        try:
            self.bio_ner = pipeline("ner", 
                model="d4data/biomedical-ner-all",
                tokenizer="d4data/biomedical-ner-all",
                aggregation_strategy="simple"
            )
        except Exception as e:
            print(f"Warning: BioBERT model not available: {e}")
            self.bio_ner = None
        
        # Drug name patterns (enhanced from research)
        self.drug_patterns = [
            r'\b[A-Z][a-z]+(?:ol|in|ine|ate|ide|ium)\b',  # Common drug suffixes
            r'\b[A-Z][a-z]*(?:mab|tinib|prazole|statin|cillin)\b',  # Drug class patterns
            r'\b(?:mg|mcg|ml|tablets?|capsules?|pills?)\b',  # Dosage indicators (context)
        ]
        
        # Load drug knowledge base
        self.known_drugs = self._load_drug_database()
        
        # Confidence weights for ensemble
        self.ensemble_weights = {
            'spacy': 0.3,
            'biobert': 0.4,
            'pattern': 0.2,
            'knowledge': 0.1
        }
    
    def _load_drug_database(self) -> set:
        """Load known drug names from various sources"""
        # In production, this would load from comprehensive drug databases
        common_drugs = {
            'aspirin', 'ibuprofen', 'acetaminophen', 'morphine', 'codeine',
            'warfarin', 'heparin', 'insulin', 'metformin', 'lisinopril',
            'atorvastatin', 'simvastatin', 'omeprazole', 'lansoprazole',
            'amoxicillin', 'ciprofloxacin', 'prednisone', 'furosemide',
            'levothyroxine', 'amlodipine', 'metoprolol', 'hydrochlorothiazide'
        }
        return common_drugs
    
    def extract_drugs(self, text: str) -> List[Dict]:
        """
        Extract drugs using ensemble of multiple methods
        
        Args:
            text: Clinical text to analyze
            
        Returns:
            List of drug mentions with confidence scores
        """
        if not self.use_ensemble:
            return self._extract_pattern_based(text)
        
        # Collect results from all methods
        spacy_results = self._extract_spacy_drugs(text) if self.spacy_model else []
        biobert_results = self._extract_biobert_drugs(text) if self.bio_ner else []
        pattern_results = self._extract_pattern_based(text)
        knowledge_results = self._extract_knowledge_based(text)
        
        # Combine using ensemble approach
        combined_results = self._ensemble_combine(
            spacy_results, biobert_results, pattern_results, knowledge_results
        )
        
        return combined_results
    
    def _extract_spacy_drugs(self, text: str) -> List[Dict]:
        """Extract drugs using SpaCy NER"""
        if not self.spacy_model:
            return []
            
        doc = self.spacy_model(text)
        drugs = []
        
        for ent in doc.ents:
            if ent.label_ in ["CHEMICAL", "DRUG", "ORG"]:  # ORG sometimes catches drug names
                # Additional validation
                if self._is_likely_drug(ent.text):
                    drugs.append({
                        'name': ent.text,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': 0.7,  # Base confidence for SpaCy
                        'method': 'spacy',
                        'entity_type': ent.label_
                    })
        
        return drugs
    
    def _extract_biobert_drugs(self, text: str) -> List[Dict]:
        """Extract drugs using BioBERT NER"""
        if not self.bio_ner:
            return []
        
        try:
            entities = self.bio_ner(text)
            drugs = []
            
            for entity in entities:
                # Filter for chemical/drug entities
                if any(keyword in entity['entity_group'].upper() 
                      for keyword in ['CHEMICAL', 'DRUG']):
                    drugs.append({
                        'name': entity['word'],
                        'start': entity['start'],
                        'end': entity['end'],
                        'confidence': entity['score'],
                        'method': 'biobert',
                        'entity_type': entity['entity_group']
                    })
            
            return drugs
            
        except Exception as e:
            print(f"BioBERT extraction error: {e}")
            return []
    
    def _extract_pattern_based(self, text: str) -> List[Dict]:
        """Extract drugs using pattern matching (baseline method)"""
        drugs = []
        
        for pattern in self.drug_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                drug_name = match.group()
                
                # Skip if it's just a dosage unit
                if drug_name.lower() in ['mg', 'mcg', 'ml', 'tablet', 'capsule', 'pill']:
                    continue
                
                drugs.append({
                    'name': drug_name,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.6,  # Lower confidence for pattern matching
                    'method': 'pattern',
                    'entity_type': 'DRUG_PATTERN'
                })
        
        return drugs
    
    def _extract_knowledge_based(self, text: str) -> List[Dict]:
        """Extract drugs using knowledge base lookup"""
        drugs = []
        text_lower = text.lower()
        
        for drug in self.known_drugs:
            if drug in text_lower:
                # Find all occurrences
                start = 0
                while True:
                    pos = text_lower.find(drug, start)
                    if pos == -1:
                        break
                    
                    # Check word boundaries
                    if (pos == 0 or not text[pos-1].isalpha()) and \
                       (pos + len(drug) >= len(text) or not text[pos + len(drug)].isalpha()):
                        drugs.append({
                            'name': drug,
                            'start': pos,
                            'end': pos + len(drug),
                            'confidence': 0.9,  # High confidence for known drugs
                            'method': 'knowledge',
                            'entity_type': 'KNOWN_DRUG'
                        })
                    
                    start = pos + 1
        
        return drugs
    
    def _ensemble_combine(self, *drug_lists) -> List[Dict]:
        """Combine results from multiple extraction methods"""
        # Group overlapping mentions
        all_drugs = []
        for drug_list in drug_lists:
            all_drugs.extend(drug_list)
        
        # Sort by position
        all_drugs.sort(key=lambda x: x['start'])
        
        # Merge overlapping mentions
        merged_drugs = []
        for drug in all_drugs:
            merged = False
            
            for i, existing in enumerate(merged_drugs):
                # Check for overlap
                if self._mentions_overlap(drug, existing):
                    # Merge mentions
                    merged_drugs[i] = self._merge_mentions(drug, existing)
                    merged = True
                    break
            
            if not merged:
                merged_drugs.append(drug)
        
        # Sort by confidence and remove duplicates
        return sorted(merged_drugs, key=lambda x: x['confidence'], reverse=True)
    
    def _mentions_overlap(self, mention1: Dict, mention2: Dict) -> bool:
        """Check if two mentions overlap significantly"""
        start1, end1 = mention1['start'], mention1['end']
        start2, end2 = mention2['start'], mention2['end']
        
        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:
            return False  # No overlap
        
        overlap_length = overlap_end - overlap_start
        min_length = min(end1 - start1, end2 - start2)
        
        # Consider overlapping if >50% overlap
        return (overlap_length / min_length) > 0.5
    
    def _merge_mentions(self, mention1: Dict, mention2: Dict) -> Dict:
        """Merge two overlapping mentions"""
        # Use mention with higher confidence as base
        if mention1['confidence'] >= mention2['confidence']:
            base, other = mention1, mention2
        else:
            base, other = mention2, mention1
        
        # Combine confidence scores (weighted average)
        method1_weight = self.ensemble_weights.get(mention1['method'], 0.5)
        method2_weight = self.ensemble_weights.get(mention2['method'], 0.5)
        
        combined_confidence = (
            mention1['confidence'] * method1_weight + 
            mention2['confidence'] * method2_weight
        ) / (method1_weight + method2_weight)
        
        return {
            'name': base['name'],
            'start': min(mention1['start'], mention2['start']),
            'end': max(mention1['end'], mention2['end']),
            'confidence': min(1.0, combined_confidence),  # Cap at 1.0
            'method': f"{mention1['method']}+{mention2['method']}",
            'entity_type': base['entity_type'],
            'methods_combined': [mention1['method'], mention2['method']]
        }
    
    def _is_likely_drug(self, text: str) -> bool:
        """Additional validation for drug-like text"""
        # Skip very short or very long strings
        if len(text) < 3 or len(text) > 30:
            return False
        
        # Skip common non-drug words
        non_drug_words = {
            'patient', 'doctor', 'hospital', 'clinic', 'treatment',
            'medicine', 'medication', 'drug', 'pills', 'tablets'
        }
        
        if text.lower() in non_drug_words:
            return False
        
        return True
    
    def evaluate_extraction(self, text: str, ground_truth: List[str]) -> Dict:
        """Evaluate extraction performance against ground truth"""
        extracted = self.extract_drugs(text)
        extracted_names = [drug['name'].lower() for drug in extracted]
        ground_truth_lower = [drug.lower() for drug in ground_truth]
        
        # Calculate metrics
        true_positives = len(set(extracted_names) & set(ground_truth_lower))
        false_positives = len(set(extracted_names) - set(ground_truth_lower))
        false_negatives = len(set(ground_truth_lower) - set(extracted_names))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'extracted': extracted_names,
            'ground_truth': ground_truth_lower
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize extractor
    extractor = EnhancedDrugExtractor(use_ensemble=True)
    
    # Test cases
    test_texts = [
        "Patient experienced nausea after taking atorvastatin 20mg daily.",
        "Started on lisinopril and developed dry cough within 2 weeks.",
        "Allergic reaction to amoxicillin with rash and swelling.",
        "Warfarin interaction with aspirin caused bleeding."
    ]
    
    ground_truths = [
        ["atorvastatin"],
        ["lisinopril"],
        ["amoxicillin"],
        ["warfarin", "aspirin"]
    ]
    
    # Test extraction
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}: {text}")
        
        # Extract drugs
        drugs = extractor.extract_drugs(text)
        print(f"Extracted drugs: {[d['name'] for d in drugs]}")
        
        # Evaluate if ground truth available
        if i < len(ground_truths):
            evaluation = extractor.evaluate_extraction(text, ground_truths[i])
            print(f"Evaluation: P={evaluation['precision']:.2f}, R={evaluation['recall']:.2f}, F1={evaluation['f1_score']:.2f}")
        
        # Show detailed results
        for drug in drugs:
            print(f"  - {drug['name']} (confidence: {drug['confidence']:.2f}, method: {drug['method']})")