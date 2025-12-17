# ADR-MedDRA Project Structure

## 📁 Directory Organization

```
ADR-MedDRA/
├── 📂 product/                     # Production-ready application
│   ├── app.py                      # Streamlit web interface
│   ├── model.py                    # MedDRA matching model
│   ├── drug_extractor.py           # Drug extraction logic
│   ├── attribution.py              # Attribution and disclaimers
│   ├── requirements.txt            # Production dependencies
│   └── data/                       # Production data files
│       ├── meddra_terms.csv        # MedDRA terms database
│       └── adr_drug_knowledge.csv  # Drug-ADR associations
│
├── 📂 src/                         # Source code (development)
│   ├── app.py                      # Development version of Streamlit app
│   ├── model.py                    # Development version of model
│   ├── drug_extractor.py           # Development drug extraction
│   └── attribution.py              # Development attribution
│
├── 📂 notebooks/                   # Jupyter notebooks for analysis
│   ├── PT_Extract.ipynb            # Preferred Term extraction analysis
│   └── UNZIPY.ipynb               # Data processing and evaluation
│
├── 📂 results/                     # Analysis results and datasets
│   ├── cadec_adr.csv              # CADEC ADR dataset
│   ├── error_analysis.csv         # Error analysis results
│   ├── final_cadec_meddra_dataset.csv  # Final processed dataset
│   └── results.csv                # Model evaluation results
│
├── 📂 data/                        # Raw and processed data
│   ├── meddra_terms.csv           # MedDRA terms database
│   └── adr_drug_knowledge.csv     # Drug-ADR knowledge base
│
├── 📂 dataset/                     # Raw datasets and metadata
│   ├── data/                      # CADEC corpus data
│   │   └── CADEC.v2/             # CADEC dataset version 2
│   └── metadata/                  # Dataset metadata and documentation
│
├── 📂 docs/                        # Documentation
│   └── PROJECT_STRUCTURE.md       # This file
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
├── 📂 .venv/                       # Python virtual environment
├── 📂 __pycache__/                 # Python cache files
├── 📂 .git/                        # Git version control
├── .gitignore                      # Git ignore rules
├── README.md                       # Project documentation
└── requirements.txt                # Development dependencies
```

## 🎯 Directory Purposes

### `/product/`
- **Purpose**: Production-ready application for deployment
- **Usage**: Deploy this directory to Streamlit Cloud or other hosting platforms
- **Key Files**: Complete Streamlit app with all dependencies and data

### `/src/`
- **Purpose**: Development source code
- **Usage**: Active development and testing of features
- **Key Files**: Development versions of core modules

### `/notebooks/`
- **Purpose**: Jupyter notebooks for data analysis and experimentation
- **Usage**: Research, data exploration, and model evaluation
- **Key Files**: Analysis notebooks and experimental code

### `/results/`
- **Purpose**: Generated results, processed datasets, and analysis outputs
- **Usage**: Store evaluation metrics, processed data, and experiment results
- **Key Files**: CSV files with analysis results and processed datasets

### `/data/`
- **Purpose**: Core data files used by the application
- **Usage**: MedDRA terms, drug knowledge bases, and reference data
- **Key Files**: Database files required for the application to function

### `/dataset/`
- **Purpose**: Raw datasets and original data sources
- **Usage**: Original CADEC corpus and metadata
- **Key Files**: Unprocessed datasets and documentation

### `/docs/`
- **Purpose**: Project documentation and guides
- **Usage**: Technical documentation, API references, and project guides
- **Key Files**: Markdown documentation files

### `/research/`
- **Purpose**: Research methodology, findings, and experimental logs
- **Usage**: Document research process, experiments, and academic findings
- **Key Files**: Methodology documentation, experiment logs, research notes

### `/deployment/`
- **Purpose**: Deployment configurations, scripts, and infrastructure code
- **Usage**: Container definitions, orchestration files, deployment automation
- **Key Files**: Dockerfile, docker-compose.yml, deployment scripts, nginx config

## 🚀 Getting Started

### For Development:
```bash
# Work with source code
cd src/
python -m streamlit run app.py
```

### For Production:
```bash
# Deploy production version
cd product/
streamlit run app.py
```

### For Analysis:
```bash
# Run Jupyter notebooks
jupyter notebook notebooks/
```

## 📋 File Naming Conventions

- **Python files**: `snake_case.py`
- **Notebooks**: `PascalCase.ipynb`
- **Data files**: `snake_case.csv`
- **Documentation**: `UPPERCASE.md`
- **Directories**: `lowercase/`

## 🔄 Workflow

1. **Development**: Work in `/src/` directory
2. **Testing**: Use notebooks in `/notebooks/` for analysis
3. **Results**: Store outputs in `/results/` directory
4. **Production**: Copy stable code to `/product/` for deployment
5. **Documentation**: Update `/docs/` with changes

## 📝 Notes

- The `/product/` directory is the deployment-ready version
- Keep `/src/` and `/product/` synchronized for stable releases
- Use `/notebooks/` for experimental analysis and research
- Store all generated results in `/results/` directory
- Version control ignores cache files and virtual environment