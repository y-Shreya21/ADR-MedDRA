# ADR-MedDRA Standardization Tool

A Streamlit-based web application for mapping Adverse Drug Reactions (ADRs) from clinical text to standardized MedDRA (Medical Dictionary for Regulatory Activities) terms using AI semantic similarity.

## 🎯 Overview

This tool helps healthcare professionals and researchers standardize adverse drug reaction reports by:
- Analyzing clinical text and patient narratives
- Extracting mentioned drugs automatically
- Mapping ADRs to standardized MedDRA Preferred Terms (PTs)
- Providing similarity scores and drug associations

## ✨ Features

- 🔍 **Semantic Search**: Uses sentence transformers for intelligent text similarity matching
- 💊 **Drug Extraction**: Automatically identifies drugs mentioned in clinical text
- 📊 **Top-K Predictions**: Returns multiple MedDRA matches with confidence scores
- 🩺 **Drug-ADR Associations**: Shows known drug-reaction relationships from knowledge base
- 📱 **Web Interface**: Easy-to-use Streamlit interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/amrishnitjsr/ADR-MedDRA.git
cd ADR-MedDRA
```

2. Navigate to the product directory:
```bash
cd product
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open your browser and go to `http://localhost:8501`

## 📁 Project Structure

```
ADR-MedDRA/
├── 📂 product/                     # Production-ready application
│   ├── app.py                     # Streamlit web interface
│   ├── model.py                   # MedDRA matching model
│   ├── drug_extractor.py          # Drug extraction logic
│   ├── attribution.py             # Attribution and disclaimers
│   ├── requirements.txt           # Production dependencies
│   └── data/                      # Production data files
│       ├── meddra_terms.csv       # MedDRA terms database
│       └── adr_drug_knowledge.csv # Drug-ADR associations
│
├── 📂 src/                         # Development source code
│   ├── app.py                     # Development Streamlit app
│   ├── model.py                   # Development model
│   ├── drug_extractor.py          # Development drug extraction
│   └── attribution.py             # Development attribution
│
├── 📂 notebooks/                   # Jupyter notebooks for analysis
│   ├── PT_Extract.ipynb           # Preferred Term extraction analysis
│   └── UNZIPY.ipynb              # Data processing and evaluation
│
├── 📂 results/                     # Analysis results and datasets
│   ├── cadec_adr.csv             # CADEC ADR dataset
│   ├── error_analysis.csv        # Error analysis results
│   ├── final_cadec_meddra_dataset.csv # Final processed dataset
│   └── results.csv               # Model evaluation results
│
├── 📂 data/                        # Core data files
│   ├── meddra_terms.csv          # MedDRA terms database
│   └── adr_drug_knowledge.csv    # Drug-ADR knowledge base
│
├── 📂 dataset/                     # Raw datasets and metadata
│   └── data/CADEC.v2/            # CADEC corpus data
│
├── 📂 docs/                        # Documentation
│   └── PROJECT_STRUCTURE.md       # Detailed structure guide
│
├── 📂 research/                    # Research documentation and findings
│   ├── METHODOLOGY.md             # Research methodology and objectives
│   └── EXPERIMENT_LOG.md          # Detailed experiment logs and results
│
├── 📂 deployment/                  # Deployment configurations and scripts
│   ├── DEPLOYMENT_GUIDE.md        # Comprehensive deployment guide
│   ├── Dockerfile                 # Docker container configuration
│   ├── docker-compose.yml         # Multi-container orchestration
│   ├── deploy.sh                  # Automated deployment script
│   └── nginx.conf                 # Reverse proxy configuration
│
└── requirements.txt                # Development dependencies
```

> 📖 **Detailed Structure Guide**: See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for complete directory documentation.

## 🔧 Usage Example

1. **Input clinical text**:
   ```
   "After taking atorvastatin, I experienced severe pain in calf muscles"
   ```

2. **Get results**:
   - **Detected Drug**: atorvastatin
   - **Top MedDRA Matches**:
     - Muscle pain (PT Code: 10028391, Score: 0.8234)
     - Myalgia (PT Code: 10028411, Score: 0.7892)
     - Muscular weakness (PT Code: 10028372, Score: 0.7456)

## 🔬 Technology Stack

- **Frontend**: Streamlit
- **ML/AI**: 
  - Sentence Transformers (all-MiniLM-L6-v2)
  - Scikit-learn (cosine similarity)
  - PyTorch
- **Data Processing**: Pandas, NumPy
- **Text Processing**: Regular expressions for drug extraction

## 📊 Data Sources

- **MedDRA Terms**: Standardized medical terminology for adverse events
- **CADEC Dataset**: Clinical text with adverse drug reaction annotations
- **Drug Knowledge Base**: Curated drug-ADR associations

## ⚠️ Important Notes

- **Research Use Only**: This is a prototype for research purposes
- **Not for Clinical Decisions**: Not intended for direct clinical use
- **Data Privacy**: Ensure compliance with healthcare data regulations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MedDRA terminology by ICH (International Council for Harmonisation)
- CADEC corpus for adverse drug event annotation
- Sentence Transformers library by UKPLab
- Streamlit for the web framework

## 📧 Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

**Disclaimer**: This tool is for research and educational purposes only. Always consult healthcare professionals for medical decisions.