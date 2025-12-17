"""
Attribution and metadata for the AI-Powered ADR Analysis Platform.

This module contains attribution information for datasets, models, and other resources
used in this project, along with medical disclaimers and usage guidelines.
"""

# Dataset attributions
CADEC_ATTRIBUTION = """
CADEC (CSIRO Adverse Drug Event Corpus) v2
Citation: Karimi, S., Metke-Jimenez, A., Kemp, M., & Wang, C. (2015). 
CADEC: a corpus of adverse drug event annotations. Journal of biomedical informatics, 55, 73-81.
License: CSIRO Data Licence Agreement
Source: https://data.csiro.au/collection/csiro:10948
"""

MEDDRA_ATTRIBUTION = """
MedDRA® (Medical Dictionary for Regulatory Activities)
The MedDRA® trademark is registered by IFPMA on behalf of ICH.
This research uses MedDRA terminology under appropriate licensing terms.
Version: MedDRA 26.1 (September 2023)
Website: https://www.meddra.org/
"""

# Model attributions
SENTENCE_TRANSFORMERS_ATTRIBUTION = """
Sentence Transformers - all-MiniLM-L6-v2
Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. 
Conference on Empirical Methods in Natural Language Processing, 2019.
Model: sentence-transformers/all-MiniLM-L6-v2
License: Apache-2.0
"""

STREAMLIT_ATTRIBUTION = """
Streamlit Framework
Copyright 2023 Snowflake Inc.
License: Apache-2.0
Website: https://streamlit.io/
"""

# Medical and legal disclaimers
MEDICAL_DISCLAIMER = """
⚠️ MEDICAL DISCLAIMER ⚠️

This AI-Powered ADR Analysis Platform is a RESEARCH PROTOTYPE intended for educational 
and research purposes ONLY. It is NOT intended for clinical use or medical decision-making.

IMPORTANT LIMITATIONS:
• This tool does NOT replace professional medical advice, diagnosis, or treatment
• Results may contain errors and should not be used for patient care decisions  
• The AI model may miss critical drug interactions or adverse effects
• Drug and specialist suggestions are general recommendations only
• Individual patient factors are not considered in the analysis

ALWAYS:
✓ Consult qualified healthcare professionals for medical advice
✓ Seek immediate medical attention for severe or emergency symptoms
✓ Verify all drug information with licensed pharmacists or physicians
✓ Report adverse drug events to official regulatory agencies

The developers, contributors, and affiliated institutions assume NO LIABILITY 
for any medical decisions made based on this tool's output.
"""

RESEARCH_DISCLAIMER = """
📊 RESEARCH USE DISCLAIMER

This software is provided "AS IS" for research and educational purposes.
• Results are not validated for clinical accuracy
• Performance may vary with different types of medical text
• The model training data may not represent all populations equally
• Continuous updates and improvements are needed for clinical application

For research publications using this tool, please cite:
[Your research paper citation when published]
"""

DATA_PRIVACY_NOTICE = """
🔒 DATA PRIVACY NOTICE

• This application processes text locally and does not store personal medical information
• No data is transmitted to external servers during analysis
• Users are responsible for ensuring compliance with applicable privacy laws (HIPAA, GDPR, etc.)
• Remove all personal identifiers before using this tool
• Do not input protected health information (PHI) or personally identifiable information (PII)
"""

USAGE_GUIDELINES = """
📋 RECOMMENDED USAGE GUIDELINES

FOR RESEARCHERS:
• Use for exploring ADR patterns in anonymized clinical text
• Validate results against gold standard datasets
• Consider bias and limitations when interpreting results
• Cite appropriate references when using in publications

FOR EDUCATORS:
• Demonstrate concepts of clinical NLP and medical informatics
• Show examples of AI applications in healthcare
• Discuss limitations and ethical considerations of AI in medicine
• Emphasize the importance of human oversight in medical AI

FOR HEALTHCARE PROFESSIONALS:
• Use only for educational exploration, not patient care
• Understand the tool's limitations before demonstrating to students
• Always emphasize the critical role of clinical judgment
• Discuss the future potential and current limitations of medical AI
"""

def get_full_attribution():
    """Return complete attribution and disclaimer information."""
    return f"""
{MEDICAL_DISCLAIMER}

{RESEARCH_DISCLAIMER}

{DATA_PRIVACY_NOTICE}

{USAGE_GUIDELINES}

--- DATA SOURCES ---
{CADEC_ATTRIBUTION}

{MEDDRA_ATTRIBUTION}

--- TECHNOLOGY STACK ---
{SENTENCE_TRANSFORMERS_ATTRIBUTION}

{STREAMLIT_ATTRIBUTION}
"""

def get_emergency_contacts():
    """Return emergency contact information."""
    return {
        "us_emergency": "911",
        "uk_emergency": "999", 
        "eu_emergency": "112",
        "poison_control_us": "1-800-222-1222",
        "who_contact": "https://www.who.int/emergencies"
    }

def get_regulatory_reporting():
    """Return information about adverse event reporting."""
    return {
        "fda_medwatch": "https://www.fda.gov/safety/medwatch",
        "ema_eudravigilance": "https://www.ema.europa.eu/en/human-regulatory/post-marketing/eudravigilance",
        "who_vigibase": "https://www.who-umc.org/vigibase/",
        "reporting_importance": "Report serious adverse drug reactions to regulatory authorities"
    }