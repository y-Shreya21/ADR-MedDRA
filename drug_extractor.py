import re

# Expanded drug database with common medications
KNOWN_DRUGS = [
    # Statins
    "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin", "statin",
    
    # Pain relievers / NSAIDs
    "paracetamol", "acetaminophen", "ibuprofen", "aspirin", "naproxen", "diclofenac",
    "celecoxib", "meloxicam", "indomethacin", "ketorolac",
    
    # Antibiotics
    "amoxicillin", "penicillin", "ciprofloxacin", "azithromycin", "doxycycline",
    "cephalexin", "clindamycin", "metronidazole", "trimethoprim", "sulfamethoxazole",
    
    # Diabetes medications
    "metformin", "insulin", "glipizide", "glyburide", "pioglitazone", "sitagliptin",
    
    # Blood pressure medications
    "lisinopril", "amlodipine", "losartan", "metoprolol", "hydrochlorothiazide",
    "atenolol", "carvedilol", "valsartan", "enalapril",
    
    # Antidepressants
    "sertraline", "fluoxetine", "citalopram", "escitalopram", "paroxetine",
    "venlafaxine", "duloxetine", "bupropion", "mirtazapine",
    
    # Proton pump inhibitors
    "omeprazole", "pantoprazole", "lansoprazole", "esomeprazole", "rabeprazole",
    
    # Antihistamines
    "diphenhydramine", "loratadine", "cetirizine", "fexofenadine", "chlorpheniramine",
    
    # Heart medications
    "warfarin", "digoxin", "furosemide", "spironolactone", "amiodarone",
    
    # Asthma/COPD
    "albuterol", "prednisone", "fluticasone", "budesonide", "montelukast",
    
    # Seizure medications
    "phenytoin", "carbamazepine", "valproic acid", "lamotrigine", "levetiracetam",
    
    # Sleep aids
    "zolpidem", "eszopiclone", "zaleplon", "trazodone", "melatonin",
    
    # Muscle relaxants
    "cyclobenzaprine", "methocarbamol", "carisoprodol", "baclofen",
    
    # Anti-nausea
    "ondansetron", "metoclopramide", "promethazine", "prochlorperazine",
    
    # Thyroid medications
    "levothyroxine", "liothyronine", "methimazole", "propylthiouracil",
    
    # Birth control
    "ethinyl estradiol", "norethindrone", "drospirenone", "levonorgestrel"
]

# Drug name variations and brand names
DRUG_VARIATIONS = {
    "tylenol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "aleve": "naproxen",
    "lipitor": "atorvastatin",
    "zocor": "simvastatin",
    "crestor": "rosuvastatin",
    "nexium": "esomeprazole",
    "prilosec": "omeprazole",
    "prevacid": "lansoprazole",
    "zoloft": "sertraline",
    "prozac": "fluoxetine",
    "lexapro": "escitalopram",
    "paxil": "paroxetine",
    "effexor": "venlafaxine",
    "cymbalta": "duloxetine",
    "wellbutrin": "bupropion",
    "glucophage": "metformin",
    "lasix": "furosemide",
    "coumadin": "warfarin",
    "synthroid": "levothyroxine",
    "ambien": "zolpidem",
    "lunesta": "eszopiclone"
}

def extract_drugs(text: str):
    """
    Extract drug names from clinical text using pattern matching.
    Returns a list of standardized drug names found in the text.
    """
    if not text:
        return []

    text = text.lower()
    found = []

    # Search for known drugs
    for drug in KNOWN_DRUGS:
        pattern = r"\b" + re.escape(drug) + r"\b"
        if re.search(pattern, text):
            found.append(drug.title())

    # Search for brand name variations
    for brand_name, generic_name in DRUG_VARIATIONS.items():
        pattern = r"\b" + re.escape(brand_name) + r"\b"
        if re.search(pattern, text):
            found.append(generic_name.title())

    # Remove duplicates and return
    return list(set(found))

def get_drug_category(drug_name: str):
    """
    Get the therapeutic category of a drug.
    """
    drug_lower = drug_name.lower()
    
    categories = {
        "statin": ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin"],
        "nsaid": ["ibuprofen", "naproxen", "diclofenac", "celecoxib", "meloxicam"],
        "antibiotic": ["amoxicillin", "penicillin", "ciprofloxacin", "azithromycin", "doxycycline"],
        "antidepressant": ["sertraline", "fluoxetine", "citalopram", "escitalopram", "paroxetine"],
        "ppi": ["omeprazole", "pantoprazole", "lansoprazole", "esomeprazole"],
        "ace_inhibitor": ["lisinopril", "enalapril", "ramipril"],
        "beta_blocker": ["metoprolol", "atenolol", "carvedilol"],
        "diabetes_med": ["metformin", "glipizide", "glyburide", "insulin"]
    }
    
    for category, drugs in categories.items():
        if drug_lower in drugs:
            return category
    
    return "other"