import streamlit as st
from model import MedDRAMatcher
from drug_extractor import extract_drugs
import pandas as pd
import re


# Load data
@st.cache_data
def load_drug_knowledge():
    return pd.read_csv("data/adr_drug_knowledge.csv")

@st.cache_data
def load_medical_specialties():
    return {
        "pain": ["Rheumatologist", "Pain Management Specialist", "Orthopedist", "Neurologist"],
        "muscle": ["Rheumatologist", "Neurologist", "Sports Medicine Doctor", "Physical Medicine & Rehabilitation"],
        "heart": ["Cardiologist", "Internal Medicine", "Emergency Medicine"],
        "stomach": ["Gastroenterologist", "Internal Medicine", "Emergency Medicine"],
        "skin": ["Dermatologist", "Allergist", "Internal Medicine"],
        "liver": ["Hepatologist", "Gastroenterologist", "Internal Medicine"],
        "kidney": ["Nephrologist", "Urologist", "Internal Medicine"],
        "brain": ["Neurologist", "Neurosurgeon", "Psychiatrist"],
        "breathing": ["Pulmonologist", "Allergist", "Emergency Medicine"],
        "vision": ["Ophthalmologist", "Neurologist", "Optometrist"],
        "hearing": ["ENT Specialist", "Audiologist", "Neurologist"],
        "blood": ["Hematologist", "Internal Medicine", "Emergency Medicine"],
        "nausea": ["Gastroenterologist", "Internal Medicine", "Emergency Medicine"],
        "dizziness": ["Neurologist", "ENT Specialist", "Cardiologist"],
        "fatigue": ["Internal Medicine", "Endocrinologist", "Hematologist"]
    }

@st.cache_data
def load_symptom_drugs():
    return {
        "pain": ["Ibuprofen", "Acetaminophen", "Naproxen", "Aspirin", "Diclofenac"],
        "muscle pain": ["Ibuprofen", "Naproxen", "Cyclobenzaprine", "Methocarbamol"],
        "headache": ["Acetaminophen", "Ibuprofen", "Sumatriptan", "Aspirin"],
        "nausea": ["Ondansetron", "Metoclopramide", "Promethazine", "Ginger"],
        "diarrhea": ["Loperamide", "Bismuth subsalicylate", "Probiotics"],
        "constipation": ["Docusate", "Polyethylene glycol", "Senna", "Bisacodyl"],
        "cough": ["Dextromethorphan", "Guaifenesin", "Honey", "Codeine"],
        "fever": ["Acetaminophen", "Ibuprofen", "Aspirin", "Naproxen"],
        "rash": ["Hydrocortisone", "Calamine", "Antihistamines", "Topical steroids"],
        "dizziness": ["Meclizine", "Dimenhydrinate", "Betahistine"]
    }

drug_kb = load_drug_knowledge()
medical_specialties = load_medical_specialties()
symptom_drugs = load_symptom_drugs()

# Page configuration
st.set_page_config(
    page_title="AI-Powered ADR Analysis Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffffff;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .feature-card h4, .feature-card p {
        color: white !important;
    }
    
    /* Main content container styling */
    .main .block-container {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
        padding: 2rem !important;
        border-radius: 10px !important;
        margin-top: 1rem !important;
    }
    
    /* Streamlit containers */
    .stContainer > div {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Empty areas and waiting states */
    .stEmpty, [data-testid="stEmpty"] {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%) !important;
        border-radius: 8px !important;
        padding: 2rem !important;
        min-height: 200px !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    
    /* Override Streamlit's built-in metric styling to use consistent colors */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    [data-testid="metric-container"] > div {
        color: white !important;
    }
    
    [data-testid="metric-container"] label {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.9rem !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: white !important;
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    .result-card {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #ffffff;
    }
    .drug-suggestion {
        background: linear-gradient(135deg, #5ca3f5 0%, #4285f4 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffffff;
        margin: 0.5rem 0;
    }
    .doctor-suggestion {
        background: linear-gradient(135deg, #6bb6ff 0%, #4dabf7 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffffff;
        margin: 0.5rem 0;
    }
    
    /* Ensure all text in suggestion cards is white */
    .drug-suggestion h5, .drug-suggestion p, .drug-suggestion small {
        color: white !important;
    }
    
    .doctor-suggestion h5, .doctor-suggestion p, .doctor-suggestion small {
        color: white !important;
    }
    
    .result-card h4, .result-card p, .result-card code {
        color: white !important;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 AI-Powered ADR Analysis Platform</h1>
    <p>Advanced Adverse Drug Reaction Analysis with MedDRA Standardization, Drug Recommendations & Medical Specialist Suggestions</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return MedDRAMatcher("data/meddra_terms.csv")

matcher = load_model()

# Sidebar for additional features
with st.sidebar:
    st.markdown("### 🔧 Analysis Settings")
    
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Comprehensive Analysis", "Quick ADR Check", "Drug Safety Review"],
        help="Choose the type of analysis you want to perform"
    )
    
    top_k = st.slider("Number of MedDRA matches", 1, 10, 5)
    
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.1)
    
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("MedDRA Terms", "27,000+")
    with col2:
        st.metric("Drug Database", "1,000+")
    
    st.markdown("### 🆘 Emergency")
    st.error("⚠️ For medical emergencies, contact emergency services immediately!")
    
    st.markdown("### 📚 Resources")
    st.info("💡 This tool is for research purposes only. Always consult healthcare professionals.")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter Clinical Description")
    
    # Example texts
    example_options = {
        "Custom Input": "",
        "Muscle Pain Example": "After taking atorvastatin for 3 weeks, I experienced severe muscle pain and weakness in my legs",
        "Stomach Issues": "Started omeprazole last month, now having persistent stomach cramps and nausea",
        "Skin Reaction": "Developed a red itchy rash on my arms after beginning penicillin treatment",
        "Neurological Symptoms": "Taking metformin for diabetes, recently experiencing dizziness and tingling in hands"
    }
    
    selected_example = st.selectbox("Choose an example or enter custom text:", list(example_options.keys()))
    
    if selected_example == "Custom Input":
        text = st.text_area(
            "Describe symptoms, medications, and timeline:",
            placeholder="Example: After taking medication X, I experienced symptoms Y and Z...",
            height=120,
            help="Be specific about medications, symptoms, timing, and severity"
        )
    else:
        text = st.text_area(
            "Describe symptoms, medications, and timeline:",
            value=example_options[selected_example],
            height=120,
            help="You can edit this example or replace it with your own text"
        )

with col2:
    st.markdown("### 🎯 Analysis Features")
    st.markdown("""
    <div class="feature-card">
        <h4>🔍 ADR Detection</h4>
        <p>Identifies potential adverse drug reactions in clinical text</p>
    </div>
    <div class="feature-card">
        <h4>🏷️ MedDRA Mapping</h4>
        <p>Maps symptoms to standardized medical terminology</p>
    </div>
    <div class="feature-card">
        <h4>💊 Drug Suggestions</h4>
        <p>Recommends treatments for identified symptoms</p>
    </div>
    <div class="feature-card">
        <h4>👩‍⚕️ Specialist Referrals</h4>
        <p>Suggests relevant medical specialists</p>
    </div>
    """, unsafe_allow_html=True)

# Analysis section
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    analyze_btn = st.button("🔍 Analyze Clinical Text", width='stretch')

with col2:
    if st.button("🧹 Clear Text", width='stretch'):
        st.rerun()

with col3:
    if st.button("📊 Sample Report", width='stretch'):
        st.info("Sample analysis loaded!")

def suggest_drugs_for_symptoms(symptoms_text):
    """Suggest relevant drugs based on identified symptoms"""
    suggested_drugs = []
    text_lower = symptoms_text.lower()
    
    for symptom, drugs in symptom_drugs.items():
        if symptom in text_lower or any(word in text_lower for word in symptom.split()):
            suggested_drugs.extend(drugs[:2])  # Top 2 drugs per symptom
    
    return list(set(suggested_drugs))[:6]  # Return unique drugs, max 6

def suggest_doctors_for_symptoms(symptoms_text):
    """Suggest medical specialists based on identified symptoms"""
    suggested_doctors = []
    text_lower = symptoms_text.lower()
    
    for keyword, specialists in medical_specialties.items():
        if keyword in text_lower:
            suggested_doctors.extend(specialists[:2])  # Top 2 specialists per keyword
    
    return list(set(suggested_doctors))[:6]  # Return unique specialists, max 6

if analyze_btn:
    if text.strip() == "":
        st.warning("⚠️ Please enter clinical text to analyze")
    else:
        with st.spinner("🔄 Analyzing clinical text..."):
            # ---- Drug extraction ----
            drugs = extract_drugs(text)

            if drugs:
                detected_drug = drugs[0]
                drug_status = "🎯 Detected from text"
            else:
                detected_drug = None
                drug_status = "❌ No drugs detected"

            # ---- ADR → MedDRA ----
            results = matcher.predict(text, top_k)
            
            # Filter by confidence threshold
            filtered_results = [r for r in results if r['score'] >= confidence_threshold]
            
            if not filtered_results:
                st.error(f"❌ No matches found above confidence threshold of {confidence_threshold:.2f}")
                st.info("💡 Try lowering the confidence threshold in the sidebar")
            else:
                # Display results in organized layout
                st.markdown("## 📊 Analysis Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown('<div class="metric-card"><h3>🎯</h3><p>Drug Detection</p><h4>' + drug_status + '</h4></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card"><h3>🏷️</h3><p>MedDRA Matches</p><h4>{len(filtered_results)}</h4></div>', unsafe_allow_html=True)
                with col3:
                    avg_score = sum(r['score'] for r in filtered_results) / len(filtered_results)
                    st.markdown(f'<div class="metric-card"><h3>📈</h3><p>Avg. Confidence</p><h4>{avg_score:.3f}</h4></div>', unsafe_allow_html=True)
                with col4:
                    st.markdown(f'<div class="metric-card"><h3>⚕️</h3><p>Analysis Mode</p><h4>{analysis_mode.split()[0]}</h4></div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Create three columns for comprehensive display
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown("### 🩺 MedDRA Standardization Results")
                    
                    for i, r in enumerate(filtered_results, 1):
                        # Determine confidence level
                        if r['score'] >= 0.8:
                            confidence_color = "#28a745"  # Green
                            confidence_label = "High"
                        elif r['score'] >= 0.6:
                            confidence_color = "#ffc107"  # Yellow
                            confidence_label = "Medium"
                        else:
                            confidence_color = "#fd7e14"  # Orange
                            confidence_label = "Low"
                        
                        # -------- Decide drug display FIRST --------
                        if detected_drug:
                            drug_display = f"**{detected_drug}** ({drug_status})"
                        else:
                            matches = drug_kb[
                                drug_kb["pt_name"].str.lower() == r["pt_name"].lower()
                            ]
                            
                            if not matches.empty:
                                drug_display = matches.iloc[0]["common_drugs"] + " (from knowledge base)"
                            else:
                                drug_display = "No associated drugs found"
                        
                        # -------- Display result card --------
                        st.markdown(f"""
                        <div class="result-card">
                            <h4>#{i} {r['pt_name']}</h4>
                            <p><strong>MedDRA Code:</strong> <code>{r['pt_code']}</code></p>
                            <p><strong>Confidence:</strong> 
                                <span style="color: {confidence_color}; font-weight: bold;">
                                    {r['score']:.4f} ({confidence_label})
                                </span>
                            </p>
                            <p><strong>💊 Associated Drug:</strong> {drug_display}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 💊 Suggested Treatments")
                    suggested_drugs = suggest_drugs_for_symptoms(text)
                    
                    if suggested_drugs:
                        for drug in suggested_drugs:
                            st.markdown(f"""
                            <div class="drug-suggestion">
                                <h5>💊 {drug}</h5>
                                <p><small>Commonly prescribed for similar symptoms</small></p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("💡 No specific drug recommendations available for the described symptoms.")
                    
                    st.markdown("---")
                    st.markdown("**⚠️ Disclaimer:** These are general suggestions. Always consult a healthcare provider before taking any medication.")
                
                with col3:
                    st.markdown("### 👩‍⚕️ Specialist Recommendations")
                    suggested_doctors = suggest_doctors_for_symptoms(text)
                    
                    if suggested_doctors:
                        for doctor in suggested_doctors:
                            st.markdown(f"""
                            <div class="doctor-suggestion">
                                <h5>🩺 {doctor}</h5>
                                <p><small>Specializes in related conditions</small></p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("💡 Consider consulting your primary care physician for initial evaluation.")
                    
                    st.markdown("---")
                    st.markdown("**📞 Emergency:** If experiencing severe symptoms, seek immediate medical attention!")
                
                # Additional analysis section
                st.markdown("---")
                st.markdown("### 📈 Detailed Analysis")
                
                with st.expander("📊 View Detailed Confidence Scores", expanded=False):
                    chart_data = pd.DataFrame({
                        'MedDRA Term': [r['pt_name'] for r in filtered_results],
                        'Confidence Score': [r['score'] for r in filtered_results],
                        'PT Code': [r['pt_code'] for r in filtered_results]
                    })
                    st.bar_chart(chart_data.set_index('MedDRA Term')['Confidence Score'])
                    st.dataframe(chart_data, width='stretch')
                
                with st.expander("🔍 Raw Analysis Data", expanded=False):
                    st.json({
                        'input_text': text,
                        'detected_drugs': drugs,
                        'analysis_mode': analysis_mode,
                        'confidence_threshold': confidence_threshold,
                        'meddra_results': filtered_results,
                        'suggested_treatments': suggested_drugs,
                        'recommended_specialists': suggested_doctors
                    })

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%); border-radius: 10px; margin-top: 2rem; color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    <h4 style="color: white; margin-bottom: 1rem;">🔬 AI-Powered ADR Analysis Platform</h4>
    <p style="color: white; margin-bottom: 0.5rem;"><strong>Research Prototype</strong> • Not for Clinical Decision Making • Always Consult Healthcare Professionals</p>
    <p style="color: rgba(255,255,255,0.9); margin: 0;"><small>Powered by Sentence Transformers • MedDRA Terminology • Machine Learning</small></p>
</div>
""", unsafe_allow_html=True)
