# -*- coding: utf-8 -*-
"""
Medical Terminology Standardization Tools.
Includes Mock Knowledge Base, UMLS API integration, and Unit Conversion.
"""
import os
from typing import Dict, Optional, Tuple

# Try to import UMLS tools
UMLS_AVAILABLE = False
standardize_term_with_umls = None
get_umls_credentials = None

try:
    from tools.tools_umls import standardize_term_with_umls, get_umls_credentials
    UMLS_AVAILABLE = True
except ImportError:
    try:
        from .tools_umls import standardize_term_with_umls, get_umls_credentials
        UMLS_AVAILABLE = True
    except ImportError:
        pass

# Enhanced Knowledge Base for medical terminology standardization
# In a real system, this would query a database or API (e.g., UMLS, BioPortal, ICD-10, ATC, LOINC)
MOCK_KB = {
    "Disease": {
        # Respiratory diseases
        "急性支气管炎": {"code": "J20.9", "system": "ICD-10", "name": "Acute bronchitis, unspecified"},
        "支气管炎": {"code": "J40", "system": "ICD-10", "name": "Bronchitis, not specified as acute or chronic"},
        "肺部感染": {"code": "J98.4", "system": "ICD-10", "name": "Other disorders of lung"},
        "肺炎": {"code": "J18.9", "system": "ICD-10", "name": "Pneumonia, unspecified organism"},
        # Cardiovascular diseases
        "高血压": {"code": "I10", "system": "ICD-10", "name": "Essential (primary) hypertension"},
        "高血压病": {"code": "I10", "system": "ICD-10", "name": "Essential (primary) hypertension"},
        "冠心病": {"code": "I25.9", "system": "ICD-10", "name": "Chronic ischemic heart disease, unspecified"},
        # Metabolic diseases
        "糖尿病": {"code": "E11", "system": "ICD-10", "name": "Type 2 diabetes mellitus"},
        "2型糖尿病": {"code": "E11", "system": "ICD-10", "name": "Type 2 diabetes mellitus"},
        "1型糖尿病": {"code": "E10", "system": "ICD-10", "name": "Type 1 diabetes mellitus"},
        # Other common diseases
        "感冒": {"code": "J00", "system": "ICD-10", "name": "Acute nasopharyngitis [common cold]"},
        "流感": {"code": "J11.1", "system": "ICD-10", "name": "Influenza with other respiratory manifestations"},
    },
    "Drug": {
        # Antibiotics
        "阿莫西林": {"code": "J01CA04", "system": "ATC", "name": "Amoxicillin"},
        "青霉素": {"code": "J01CE01", "system": "ATC", "name": "Benzylpenicillin"},
        "头孢": {"code": "J01D", "system": "ATC", "name": "Cephalosporins"},
        # Anti-inflammatory
        "阿司匹林": {"code": "B01AC06", "system": "ATC", "name": "Acetylsalicylic acid"},
        "布洛芬": {"code": "M01AE01", "system": "ATC", "name": "Ibuprofen"},
        # Antidiabetic
        "二甲双胍": {"code": "A10BA02", "system": "ATC", "name": "Metformin"},
        "胰岛素": {"code": "A10A", "system": "ATC", "name": "Insulins and analogues"},
        # Antihypertensive
        "卡托普利": {"code": "C09AA01", "system": "ATC", "name": "Captopril"},
        "硝苯地平": {"code": "C08CA01", "system": "ATC", "name": "Nifedipine"},
    },
    "Test": {
        # Blood tests
        "WBC": {"code": "6690-2", "system": "LOINC", "name": "Leukocytes [#/volume] in Blood"},
        "白细胞": {"code": "6690-2", "system": "LOINC", "name": "Leukocytes [#/volume] in Blood"},
        "白细胞计数": {"code": "6690-2", "system": "LOINC", "name": "Leukocytes [#/volume] in Blood"},
        "血红蛋白": {"code": "718-7", "system": "LOINC", "name": "Hemoglobin [Mass/volume] in Blood"},
        "Hb": {"code": "718-7", "system": "LOINC", "name": "Hemoglobin [Mass/volume] in Blood"},
        "血糖": {"code": "2339-0", "system": "LOINC", "name": "Glucose [Mass/volume] in Blood"},
        "葡萄糖": {"code": "2339-0", "system": "LOINC", "name": "Glucose [Mass/volume] in Blood"},
        "胆固醇": {"code": "2093-3", "system": "LOINC", "name": "Cholesterol [Mass/volume] in Blood"},
        "总胆固醇": {"code": "2093-3", "system": "LOINC", "name": "Cholesterol [Mass/volume] in Blood"},
        "血小板": {"code": "777-3", "system": "LOINC", "name": "Platelets [#/volume] in Blood"},
        "血小板计数": {"code": "777-3", "system": "LOINC", "name": "Platelets [#/volume] in Blood"},
    },
    "Symptom": {
        # Common symptoms (using SNOMED CT codes as example)
        "发热": {"code": "386661006", "system": "SNOMED-CT", "name": "Fever"},
        "咳嗽": {"code": "49727002", "system": "SNOMED-CT", "name": "Cough"},
        "头痛": {"code": "25064002", "system": "SNOMED-CT", "name": "Headache"},
        "胸痛": {"code": "29857009", "system": "SNOMED-CT", "name": "Chest pain"},
    }
}

# ============================================================================
# 医学检验单位转换规则库 (Enhanced Unit Conversion Rules)
# ============================================================================
# 转换公式来源：
# - 国际单位制 (SI) 标准
# - 《临床检验诊断学》教材
# - LOINC 数据库参考单位
# ============================================================================

UNIT_CONVERSION_RULES = {
    # ========================================================================
    # 血糖相关 (Glucose)
    # ========================================================================
    # 血糖: mg/dL → mmol/L
    # 分子量: 180.16 g/mol, 转换系数: 1 mg/dL = 0.0555 mmol/L (1/18.016)
    "glucose": {
        "from": ["mg/dL", "mg/dl", "mg/100ml", "mg%"],
        "to": "mmol/L",
        "factor": 1 / 18.016,  # 精确值: 1/18.016 ≈ 0.05551
        "keywords": ["血糖", "葡萄糖", "glucose", "sugar", "GLU", "FPG", "空腹血糖", 
                    "餐后血糖", "随机血糖", "血糖测定"]
    },
    
    # ========================================================================
    # 血脂相关 (Lipids)
    # ========================================================================
    # 总胆固醇: mg/dL → mmol/L
    # 分子量: 386.65 g/mol, 转换系数: 1 mg/dL = 0.02586 mmol/L
    "cholesterol": {
        "from": ["mg/dL", "mg/dl", "mg/100ml"],
        "to": "mmol/L",
        "factor": 1 / 38.665,  # 精确值: 10/386.65 ≈ 0.02586
        "keywords": ["胆固醇", "总胆固醇", "cholesterol", "TC", "CHOL", "T-CHO"]
    },
    # 甘油三酯: mg/dL → mmol/L
    # 平均分子量: 885 g/mol (甘油三酯分子量变化较大，取平均值)
    "triglycerides": {
        "from": ["mg/dL", "mg/dl", "mg/100ml"],
        "to": "mmol/L",
        "factor": 1 / 88.5,  # 10/885 ≈ 0.0113
        "keywords": ["甘油三酯", "三酰甘油", "triglyceride", "TG", "TRIG"]
    },
    # HDL胆固醇: mg/dL → mmol/L (与总胆固醇相同)
    "hdl": {
        "from": ["mg/dL", "mg/dl", "mg/100ml"],
        "to": "mmol/L",
        "factor": 1 / 38.665,
        "keywords": ["高密度脂蛋白", "HDL", "HDL-C", "好胆固醇"]
    },
    # LDL胆固醇: mg/dL → mmol/L (与总胆固醇相同)
    "ldl": {
        "from": ["mg/dL", "mg/dl", "mg/100ml"],
        "to": "mmol/L",
        "factor": 1 / 38.665,
        "keywords": ["低密度脂蛋白", "LDL", "LDL-C", "坏胆固醇"]
    },
    
    # ========================================================================
    # 血常规相关 (Complete Blood Count)
    # ========================================================================
    # 白细胞 (已经是×10³/μL或K/μL格式，1:1转换)
    # 注意：这个规则必须在wbc之前，以确保优先匹配
    "wbc_k": {
        "from": ["×10³/μL", "10^3/uL", "K/uL", "K/μL", "千/μL", "×10^3/μL", "10^3/μL",
                 "x10^3/uL", "x10^3/μL", "10*3/uL", "10E3/uL"],
        "to": "×10^9/L",
        "factor": 1.0,  # 1 ×10³/μL = 1 ×10^9/L (因为 10³/μL = 10³/10⁻⁶L = 10⁹/L)
        "keywords": ["白细胞", "WBC", "leukocyte", "white blood cell", "白细胞计数"]
    },
    # 白细胞: /mm³ → 10^9/L (单个细胞计数 → 十亿/升)
    "wbc": {
        "from": ["/mm3", "/ul", "/μL", "/mm³", "cells/mm3", "cells/μL", "cells/uL"],
        "to": "×10^9/L",
        "factor": 1 / 1000.0,  # 1 /mm³ = 1 /μL = 0.001 ×10^9/L
        "keywords": ["白细胞", "WBC", "leukocyte", "white blood cell", "白细胞计数"]
    },
    # 红细胞 (已经是×10^6/μL或M/μL格式，1:1转换)
    "rbc_m": {
        "from": ["×10^6/μL", "10^6/uL", "M/uL", "M/μL", "百万/μL", "10^6/μL",
                 "x10^6/uL", "x10^6/μL", "10*6/uL", "10E6/uL"],
        "to": "×10^12/L",
        "factor": 1.0,  # 1 ×10^6/μL = 1 ×10^12/L (因为 10⁶/μL = 10⁶/10⁻⁶L = 10¹²/L)
        "keywords": ["红细胞", "RBC", "erythrocyte", "red blood cell", "红细胞计数"]
    },
    # 红细胞: /mm³ → 10^12/L (单个细胞计数 → 万亿/升)
    "rbc": {
        "from": ["/mm3", "/ul", "/μL", "/mm³", "cells/mm3"],
        "to": "×10^12/L",
        "factor": 1 / 1000000.0,  # 1 /mm³ = 10^-6 ×10^12/L
        "keywords": ["红细胞", "RBC", "erythrocyte", "red blood cell", "红细胞计数"]
    },
    # 血小板 (已经是×10³/μL或K/μL格式，1:1转换)
    "platelet_k": {
        "from": ["×10³/μL", "10^3/uL", "K/uL", "K/μL", "10^3/μL",
                 "x10^3/uL", "x10^3/μL", "10*3/uL", "10E3/uL"],
        "to": "×10^9/L",
        "factor": 1.0,  # 1 ×10³/μL = 1 ×10^9/L
        "keywords": ["血小板", "platelet", "PLT", "血小板计数", "thrombocyte"]
    },
    # 血小板: /mm³ → 10^9/L
    "platelet": {
        "from": ["/mm3", "/ul", "/μL", "/mm³", "cells/mm3"],
        "to": "×10^9/L",
        "factor": 1 / 1000.0,
        "keywords": ["血小板", "platelet", "PLT", "血小板计数", "thrombocyte"]
    },
    # 血红蛋白: g/dL → g/L
    "hemoglobin": {
        "from": ["g/dL", "g/dl", "g/100ml", "g%"],
        "to": "g/L",
        "factor": 10.0,  # 1 g/dL = 10 g/L
        "keywords": ["血红蛋白", "Hb", "hemoglobin", "HGB", "Hgb", "血色素"]
    },
    # 红细胞压积/血细胞比容: % → L/L (小数)
    "hematocrit": {
        "from": ["%"],
        "to": "L/L",
        "factor": 0.01,  # 45% = 0.45 L/L
        "keywords": ["红细胞压积", "血细胞比容", "hematocrit", "HCT", "PCV", "Hct"]
    },
    
    # ========================================================================
    # 肾功能相关 (Renal Function)
    # ========================================================================
    # 肌酐: mg/dL → μmol/L
    # 分子量: 113.12 g/mol, 转换系数: 1 mg/dL = 88.4 μmol/L
    "creatinine": {
        "from": ["mg/dL", "mg/dl"],
        "to": "μmol/L",
        "factor": 88.4,  # 10000/113.12 ≈ 88.4
        "keywords": ["肌酐", "creatinine", "CREA", "Cr", "血肌酐", "SCr"]
    },
    # 尿素氮 (BUN): mg/dL → mmol/L
    # 分子量 (尿素): 60.06 g/mol, BUN报告的是氮的质量
    # 氮分子量: 28, 转换: 1 mg/dL BUN = 0.357 mmol/L 尿素
    "bun": {
        "from": ["mg/dL", "mg/dl"],
        "to": "mmol/L",
        "factor": 1 / 2.8,  # ≈ 0.357
        "keywords": ["尿素氮", "BUN", "blood urea nitrogen", "尿素"]
    },
    # 尿酸: mg/dL → μmol/L
    # 分子量: 168.11 g/mol, 转换系数: 1 mg/dL = 59.48 μmol/L
    "uric_acid": {
        "from": ["mg/dL", "mg/dl"],
        "to": "μmol/L",
        "factor": 59.48,  # 10000/168.11 ≈ 59.48
        "keywords": ["尿酸", "uric acid", "UA", "血尿酸"]
    },
    
    # ========================================================================
    # 肝功能相关 (Liver Function)
    # ========================================================================
    # 胆红素: mg/dL → μmol/L
    # 分子量: 584.66 g/mol, 转换系数: 1 mg/dL = 17.1 μmol/L
    "bilirubin": {
        "from": ["mg/dL", "mg/dl"],
        "to": "μmol/L",
        "factor": 17.1,  # 10000/584.66 ≈ 17.1
        "keywords": ["胆红素", "总胆红素", "直接胆红素", "间接胆红素", "bilirubin", 
                    "TBIL", "DBIL", "IBIL", "T-Bil", "D-Bil"]
    },
    # 白蛋白: g/dL → g/L
    "albumin": {
        "from": ["g/dL", "g/dl", "g/100ml"],
        "to": "g/L",
        "factor": 10.0,
        "keywords": ["白蛋白", "albumin", "ALB", "血清白蛋白"]
    },
    # 总蛋白: g/dL → g/L
    "total_protein": {
        "from": ["g/dL", "g/dl", "g/100ml"],
        "to": "g/L",
        "factor": 10.0,
        "keywords": ["总蛋白", "total protein", "TP", "血清总蛋白"]
    },
    
    # ========================================================================
    # 电解质相关 (Electrolytes)
    # ========================================================================
    # 钠: mEq/L → mmol/L (1:1，钠是一价离子)
    "sodium": {
        "from": ["mEq/L", "meq/L", "meq/l"],
        "to": "mmol/L",
        "factor": 1.0,  # 一价离子: 1 mEq = 1 mmol
        "keywords": ["钠", "sodium", "Na", "Na+", "血钠"]
    },
    # 钾: mEq/L → mmol/L (1:1，钾是一价离子)
    "potassium": {
        "from": ["mEq/L", "meq/L", "meq/l"],
        "to": "mmol/L",
        "factor": 1.0,
        "keywords": ["钾", "potassium", "K", "K+", "血钾"]
    },
    # 氯: mEq/L → mmol/L (1:1，氯是一价离子)
    "chloride": {
        "from": ["mEq/L", "meq/L", "meq/l"],
        "to": "mmol/L",
        "factor": 1.0,
        "keywords": ["氯", "chloride", "Cl", "Cl-", "血氯"]
    },
    # 钙: mg/dL → mmol/L
    # 分子量: 40.08 g/mol, 转换系数: 1 mg/dL = 0.25 mmol/L
    "calcium": {
        "from": ["mg/dL", "mg/dl"],
        "to": "mmol/L",
        "factor": 1 / 4.0,  # 10/40.08 ≈ 0.25
        "keywords": ["钙", "calcium", "Ca", "Ca2+", "血钙", "总钙"]
    },
    # 钙: mEq/L → mmol/L (2:1，钙是二价离子)
    "calcium_meq": {
        "from": ["mEq/L", "meq/L"],
        "to": "mmol/L",
        "factor": 0.5,  # 二价离子: 1 mEq = 0.5 mmol
        "keywords": ["钙", "calcium", "Ca", "Ca2+"]
    },
    # 磷: mg/dL → mmol/L
    # 分子量: 30.97 g/mol, 转换系数: 1 mg/dL = 0.323 mmol/L
    "phosphorus": {
        "from": ["mg/dL", "mg/dl"],
        "to": "mmol/L",
        "factor": 1 / 3.1,  # 10/30.97 ≈ 0.323
        "keywords": ["磷", "phosphorus", "P", "血磷", "无机磷"]
    },
    # 镁: mg/dL → mmol/L
    # 分子量: 24.31 g/mol, 转换系数: 1 mg/dL = 0.411 mmol/L
    "magnesium": {
        "from": ["mg/dL", "mg/dl"],
        "to": "mmol/L",
        "factor": 1 / 2.43,  # 10/24.31 ≈ 0.411
        "keywords": ["镁", "magnesium", "Mg", "Mg2+", "血镁"]
    },
    # 镁: mEq/L → mmol/L (2:1，镁是二价离子)
    "magnesium_meq": {
        "from": ["mEq/L", "meq/L"],
        "to": "mmol/L",
        "factor": 0.5,
        "keywords": ["镁", "magnesium", "Mg"]
    },
    
    # ========================================================================
    # 心肌标志物 (Cardiac Markers)
    # ========================================================================
    # 肌钙蛋白: ng/mL → μg/L (1:1)
    "troponin": {
        "from": ["ng/mL", "ng/ml"],
        "to": "μg/L",
        "factor": 1.0,  # 1 ng/mL = 1 μg/L
        "keywords": ["肌钙蛋白", "troponin", "cTnI", "cTnT", "TnI", "TnT", 
                    "心肌肌钙蛋白"]
    },
    # 肌酸激酶: U/L (通常不需要转换，但有时报告为μkat/L)
    "ck": {
        "from": ["U/L", "IU/L"],
        "to": "μkat/L",
        "factor": 1 / 60.0,  # 1 U = 1/60 μkat
        "keywords": ["肌酸激酶", "CK", "creatine kinase", "CPK", "CK-MB"]
    },
    # BNP: pg/mL → ng/L (1:1)
    "bnp": {
        "from": ["pg/mL", "pg/ml"],
        "to": "ng/L",
        "factor": 1.0,
        "keywords": ["脑钠肽", "BNP", "NT-proBNP", "B型钠尿肽"]
    },
    
    # ========================================================================
    # 凝血功能 (Coagulation)
    # ========================================================================
    # 纤维蛋白原: mg/dL → g/L
    "fibrinogen": {
        "from": ["mg/dL", "mg/dl"],
        "to": "g/L",
        "factor": 0.01,  # 1 mg/dL = 0.01 g/L
        "keywords": ["纤维蛋白原", "fibrinogen", "FIB", "Fbg"]
    },
    # D-二聚体: μg/mL → mg/L (1:1)
    "d_dimer": {
        "from": ["μg/mL", "ug/mL", "μg/ml", "ug/ml"],
        "to": "mg/L",
        "factor": 1.0,
        "keywords": ["D-二聚体", "D-dimer", "DDimer", "D二聚体"]
    },
    # D-二聚体: ng/mL → μg/L (1:1) 或 mg/L (÷1000)
    "d_dimer_ng": {
        "from": ["ng/mL", "ng/ml"],
        "to": "μg/L",
        "factor": 1.0,
        "keywords": ["D-二聚体", "D-dimer"]
    },
    
    # ========================================================================
    # 甲状腺功能 (Thyroid Function)
    # ========================================================================
    # T4 (甲状腺素): μg/dL → nmol/L
    # 分子量: 776.87 g/mol, 转换系数: 1 μg/dL = 12.87 nmol/L
    "t4": {
        "from": ["μg/dL", "ug/dL", "μg/dl", "ug/dl"],
        "to": "nmol/L",
        "factor": 12.87,  # 10000/776.87 ≈ 12.87
        "keywords": ["T4", "甲状腺素", "thyroxine", "总T4", "FT4", "游离T4"]
    },
    # T3 (三碘甲状腺原氨酸): ng/dL → nmol/L
    # 分子量: 650.98 g/mol, 转换系数: 1 ng/dL = 0.0154 nmol/L
    "t3": {
        "from": ["ng/dL", "ng/dl"],
        "to": "nmol/L",
        "factor": 0.0154,  # 100/650.98 ≈ 0.0154
        "keywords": ["T3", "三碘甲状腺原氨酸", "triiodothyronine", "总T3", "FT3", "游离T3"]
    },
    
    # ========================================================================
    # 激素相关 (Hormones)
    # ========================================================================
    # 皮质醇: μg/dL → nmol/L
    # 分子量: 362.46 g/mol, 转换系数: 1 μg/dL = 27.59 nmol/L
    "cortisol": {
        "from": ["μg/dL", "ug/dL", "μg/dl", "ug/dl"],
        "to": "nmol/L",
        "factor": 27.59,  # 10000/362.46 ≈ 27.59
        "keywords": ["皮质醇", "cortisol", "CORT", "血皮质醇"]
    },
    # 睾酮: ng/dL → nmol/L
    # 分子量: 288.42 g/mol, 转换系数: 1 ng/dL = 0.0347 nmol/L
    "testosterone": {
        "from": ["ng/dL", "ng/dl"],
        "to": "nmol/L",
        "factor": 0.0347,  # 100/288.42 ≈ 0.0347
        "keywords": ["睾酮", "testosterone", "T", "血睾酮"]
    },
    # 雌二醇: pg/mL → pmol/L
    # 分子量: 272.38 g/mol, 转换系数: 1 pg/mL = 3.67 pmol/L
    "estradiol": {
        "from": ["pg/mL", "pg/ml"],
        "to": "pmol/L",
        "factor": 3.67,  # 1000/272.38 ≈ 3.67
        "keywords": ["雌二醇", "estradiol", "E2", "血雌二醇"]
    },
    
    # ========================================================================
    # 铁代谢 (Iron Metabolism)
    # ========================================================================
    # 铁: μg/dL → μmol/L
    # 分子量: 55.85 g/mol, 转换系数: 1 μg/dL = 0.179 μmol/L
    "iron": {
        "from": ["μg/dL", "ug/dL", "μg/dl", "ug/dl"],
        "to": "μmol/L",
        "factor": 0.179,  # 10/55.85 ≈ 0.179
        "keywords": ["铁", "iron", "Fe", "血清铁", "血铁"]
    },
    # 铁蛋白: ng/mL → μg/L (1:1)
    "ferritin": {
        "from": ["ng/mL", "ng/ml"],
        "to": "μg/L",
        "factor": 1.0,
        "keywords": ["铁蛋白", "ferritin", "SF", "血清铁蛋白"]
    },
    
    # ========================================================================
    # 炎症标志物 (Inflammatory Markers)
    # ========================================================================
    # C反应蛋白: mg/L (通常已是标准单位，但有时报告为mg/dL)
    "crp": {
        "from": ["mg/dL", "mg/dl"],
        "to": "mg/L",
        "factor": 10.0,  # 1 mg/dL = 10 mg/L
        "keywords": ["C反应蛋白", "CRP", "C-reactive protein", "超敏CRP", "hs-CRP"]
    },
    
    # ========================================================================
    # 血气分析 (Blood Gas)
    # ========================================================================
    # 氧分压: mmHg → kPa
    "pO2": {
        "from": ["mmHg", "Torr"],
        "to": "kPa",
        "factor": 0.133,  # 1 mmHg = 0.133 kPa
        "keywords": ["氧分压", "PO2", "pO2", "paO2", "动脉血氧分压"]
    },
    # 二氧化碳分压: mmHg → kPa
    "pCO2": {
        "from": ["mmHg", "Torr"],
        "to": "kPa",
        "factor": 0.133,
        "keywords": ["二氧化碳分压", "PCO2", "pCO2", "paCO2", "动脉血二氧化碳分压"]
    },
    
    # ========================================================================
    # 体温 (Temperature)
    # ========================================================================
    # 华氏度 → 摄氏度
    "temperature_f_to_c": {
        "from": ["°F", "F", "华氏度"],
        "to": "°C",
        "factor": lambda x: (x - 32) * 5 / 9,  # 特殊转换函数
        "keywords": ["体温", "temperature", "temp", "发热"]
    },
    # 摄氏度 → 华氏度 (反向转换)
    "temperature_c_to_f": {
        "from": ["°C", "C", "摄氏度"],
        "to": "°F",
        "factor": lambda x: x * 9 / 5 + 32,
        "keywords": []  # 不设关键词，避免误匹配，需显式调用
    },
    
    # ========================================================================
    # 体重/身高 (Weight/Height)
    # ========================================================================
    # 体重: lb → kg
    "weight_lb_to_kg": {
        "from": ["lb", "lbs", "磅"],
        "to": "kg",
        "factor": 0.4536,  # 1 lb = 0.4536 kg
        "keywords": ["体重", "weight"]
    },
    # 身高: inch → cm
    "height_in_to_cm": {
        "from": ["in", "inch", "inches", "英寸"],
        "to": "cm",
        "factor": 2.54,  # 1 inch = 2.54 cm
        "keywords": ["身高", "height"]
    },
}

def standardize_term(
    term: str, 
    category: str, 
    use_umls: bool = True,
    fallback_to_umls: bool = True
) -> Dict[str, str]:
    """
    Map a term to a standard code based on category.
    
    This function performs terminology standardization by mapping medical terms
    to standard coding systems:
    - Diseases: ICD-10 codes
    - Drugs: ATC codes
    - Tests: LOINC codes
    - Symptoms: SNOMED-CT codes
    
    Strategy (Priority Order):
    1. First tries UMLS API (if use_umls=True and credentials available)
    2. If UMLS fails or not available, falls back to local knowledge base (MOCK_KB)
       unless UMLS_ONLY=true is set
    3. Returns empty dict if all methods fail
    
    Args:
        term (str): The term to look up (e.g., "阿莫西林", "2型糖尿病").
        category (str): The category (Disease, Drug, Test, Symptom, etc.).
        use_umls (bool): Whether to use UMLS API first (default: True).
        fallback_to_umls (bool): Deprecated parameter, kept for compatibility. 
                                UMLS is now tried first if use_umls=True.
        
    Returns:
        Dict[str, str]: Dictionary with 'code', 'system', 'name' (standard name), 'source'.
                        Returns empty dict if not found.
    """
    if not term or not category:
        return {}
    
    # Normalize the lookup term: remove spaces, convert to lowercase for matching
    lookup_term = term.strip()
    lookup_term_lower = lookup_term.lower()
    
    # Check if we should only use UMLS (no fallback to local KB)
    umls_only = os.environ.get("UMLS_ONLY", "false").lower() == "true"
    verbose = os.environ.get("UMLS_VERBOSE", "false").lower() == "true"
    
    # Strategy 1: Try UMLS API first (if enabled and available)
    if use_umls and UMLS_AVAILABLE:
        # Check if UMLS credentials are available
        api_key, username, password = get_umls_credentials()
        if api_key or (username and password):
            if verbose:
                print(f"[UMLS] 尝试标准化术语: {lookup_term} (类别: {category})")
            try:
                # On first call, try to force refresh authentication to get detailed error
                if not hasattr(standardize_term, '_first_call_done'):
                    from tools.tools_umls import clear_umls_cache, get_umls_ticket
                    clear_umls_cache()
                    # Try to get ticket with force refresh to see detailed error
                    ticket = get_umls_ticket(force_refresh=True)
                    if not ticket:
                        print("[UMLS] 警告: UMLS认证失败，请检查API key是否正确")
                    standardize_term._first_call_done = True
                
                umls_result = standardize_term_with_umls(lookup_term, category, verbose=verbose)
                if umls_result and umls_result.get("code"):
                    # Mark as from UMLS
                    umls_result["source"] = "umls"
                    if verbose:
                        print(f"[UMLS] ✓ 成功: {lookup_term} -> {umls_result.get('system')} {umls_result.get('code')}")
                    return umls_result
                else:
                    if verbose:
                        print(f"[UMLS] ✗ 未找到匹配结果: {lookup_term}")
                    # If UMLS_ONLY is true, don't fallback to local KB
                    if umls_only:
                        return {}
            except Exception as e:
                # Log error
                error_msg = f"[UMLS] 异常: {e}"
                print(error_msg)
                if verbose:
                    import traceback
                    print(f"[UMLS] 详细错误: {traceback.format_exc()}")
                # If UMLS_ONLY is true, don't fallback to local KB
                if umls_only:
                    return {}
        else:
            if verbose:
                print(f"[UMLS] 警告: 未找到UMLS凭证 (UMLS_API_KEY 或 UMLS_USERNAME/UMLS_PASSWORD)")
            # If UMLS_ONLY is true and no credentials, return empty
            if umls_only:
                return {}
    elif use_umls and not UMLS_AVAILABLE:
        if verbose:
            print(f"[UMLS] 警告: UMLS工具不可用 (tools_umls模块未导入)")
        if umls_only:
            return {}
    
    # Strategy 2: Fallback to local knowledge base if UMLS failed or not available
    # Skip if UMLS_ONLY is true
    if umls_only:
        return {}
    
    if category in MOCK_KB:
        kb = MOCK_KB[category]
        
        # Strategy 2.1: Exact match
        if lookup_term in kb:
            result = kb[lookup_term].copy()
            result["source"] = "local_kb"
            return result
        
        # Strategy 2.2: Case-insensitive exact match
        for key, val in kb.items():
            if key.lower() == lookup_term_lower:
                result = val.copy()
                result["source"] = "local_kb"
                return result
        
        # Strategy 2.3: Substring matching (fuzzy match)
        for key, val in kb.items():
            key_lower = key.lower()
            if key_lower in lookup_term_lower or lookup_term_lower in key_lower:
                result = val.copy()
                result["source"] = "local_kb"
                return result
        
        # Strategy 2.4: Remove common suffixes/prefixes and try again
        suffixes = ["症", "病", "炎", "感染", "综合征", "症候群"]
        prefixes = ["急性", "慢性", "严重", "轻度", "中度", "重度"]
        
        # Try without suffixes
        for suffix in suffixes:
            if lookup_term.endswith(suffix):
                base_term = lookup_term[:-len(suffix)]
                if base_term in kb:
                    result = kb[base_term].copy()
                    result["source"] = "local_kb"
                    return result
        
        # Try without prefixes
        for prefix in prefixes:
            if lookup_term.startswith(prefix):
                base_term = lookup_term[len(prefix):]
                if base_term in kb:
                    result = kb[base_term].copy()
                    result["source"] = "local_kb"
                    return result
    
    return {}

# ============================================================================
# 医学标准单位白名单 - 这些单位已经是临床标准，不需要转换
# ============================================================================
# 这些单位在临床实践中广泛使用，不应该被 LLM 错误地转换成其他单位
STANDARD_UNITS_NO_CONVERSION = {
    # 眼科检查单位 - 角度和角速度在眼动检查中是标准单位
    "°", "度", "deg", "degree", "degrees",
    "°/s", "度/s", "度/秒", "deg/s", "°/sec",
    # 血液学单位 - pg, fL 是红细胞指标的标准单位
    "pg",       # 平均红细胞血红蛋白含量 MCH (27-34 pg)
    "fL", "fl", # 平均红细胞体积 MCV (82-100 fL)
    # 百分比 - 很多血常规指标使用百分比
    "%",
    # 国际单位
    "IU/mL", "IU/ml", "IU/L",
    # 抗体滴度 - 通常以比值表示
    "ratio", "比值",
    # 已经是 SI 单位或医学标准单位
    "mmol/L", "μmol/L", "nmol/L", "pmol/L",
    "g/L", "mg/L", "μg/L", "ng/L",
    "×10^9/L", "×10^12/L", "10^9/L", "10^12/L",
    "×10³/μL", "×10^6/μL",
    "L/L",  # 血细胞比容
    "rad", "rad/s",  # 已经是弧度
}

# 单位别名映射表 - 用于标准化不同写法的单位
UNIT_ALIASES = {
    # mg/dL 变体
    "mg/dl": "mg/dL", "MG/DL": "mg/dL", "mg/100ml": "mg/dL", "mg%": "mg/dL",
    # g/dL 变体
    "g/dl": "g/dL", "G/DL": "g/dL", "g/100ml": "g/dL", "g%": "g/dL", "gm/dL": "g/dL",
    # μg/dL 变体
    "ug/dL": "μg/dL", "ug/dl": "μg/dL", "mcg/dL": "μg/dL", "mcg/dl": "μg/dL",
    # ng/dL 变体
    "ng/dl": "ng/dL", "NG/DL": "ng/dL",
    # ng/mL 变体
    "ng/ml": "ng/mL", "NG/ML": "ng/mL",
    # pg/mL 变体
    "pg/ml": "pg/mL", "PG/ML": "pg/mL",
    # μg/mL 变体
    "ug/mL": "μg/mL", "ug/ml": "μg/mL", "mcg/mL": "μg/mL", "mcg/ml": "μg/mL",
    # mmol/L 变体
    "mmol/l": "mmol/L", "MMOL/L": "mmol/L", "mM": "mmol/L",
    # μmol/L 变体
    "umol/L": "μmol/L", "umol/l": "μmol/L", "μmol/l": "μmol/L",
    # nmol/L 变体
    "nmol/l": "nmol/L", "NMOL/L": "nmol/L",
    # pmol/L 变体
    "pmol/l": "pmol/L", "PMOL/L": "pmol/L",
    # mEq/L 变体
    "meq/L": "mEq/L", "meq/l": "mEq/L", "MEQ/L": "mEq/L",
    # /mm³ 变体
    "/mm3": "/mm³", "/uL": "/μL", "/ul": "/μL", "cells/mm3": "/mm³",
    "/μl": "/μL", "cells/μL": "/μL", "cells/uL": "/μL",
    # ×10³/μL 变体
    "10^3/uL": "×10³/μL", "10^3/μL": "×10³/μL", "K/uL": "×10³/μL", 
    "K/μL": "×10³/μL", "千/μL": "×10³/μL", "x10^3/uL": "×10³/μL",
    "10*3/uL": "×10³/μL", "10E3/uL": "×10³/μL",
    # ×10^6/μL 变体
    "10^6/uL": "×10^6/μL", "10^6/μL": "×10^6/μL", "M/uL": "×10^6/μL",
    "M/μL": "×10^6/μL", "百万/μL": "×10^6/μL", "x10^6/uL": "×10^6/μL",
    # ×10^9/L 变体
    "10^9/L": "×10^9/L", "10*9/L": "×10^9/L", "10E9/L": "×10^9/L",
    "G/L": "×10^9/L", "x10^9/L": "×10^9/L",
    # ×10^12/L 变体
    "10^12/L": "×10^12/L", "10*12/L": "×10^12/L", "T/L": "×10^12/L",
    # g/L 变体
    "g/l": "g/L", "G/l": "g/L", "gm/L": "g/L",
    # U/L 变体
    "u/L": "U/L", "u/l": "U/L", "IU/L": "U/L", "IU/l": "U/L",
    # μkat/L 变体
    "ukat/L": "μkat/L", "ukat/l": "μkat/L",
    # kPa 变体
    "kpa": "kPa", "KPA": "kPa",
    # 温度变体
    "F": "°F", "f": "°F", "华氏": "°F",
    "C": "°C", "c": "°C", "摄氏": "°C",
}


def _normalize_unit_string(unit: str) -> str:
    """
    标准化单位字符串
    
    Args:
        unit: 原始单位字符串
        
    Returns:
        标准化后的单位字符串
    """
    unit = unit.strip()
    # 查找别名映射
    return UNIT_ALIASES.get(unit, unit)


def _smart_round(value: float, original_precision: int = None) -> float:
    """
    智能四舍五入，根据数值大小选择合适的精度
    
    Args:
        value: 要四舍五入的数值
        original_precision: 原始值的精度（小数位数）
        
    Returns:
        四舍五入后的数值
    """
    if value == 0:
        return 0.0
    
    abs_value = abs(value)
    
    # 非常小的值：保留4位有效数字
    if abs_value < 0.001:
        # 使用科学计数法的精度
        from math import floor, log10
        if abs_value > 0:
            magnitude = floor(log10(abs_value))
            return round(value, -magnitude + 3)
        return value
    # 小于1：保留4位小数
    elif abs_value < 1:
        return round(value, 4)
    # 1-100：保留2位小数
    elif abs_value < 100:
        return round(value, 2)
    # 100-10000：保留1位小数
    elif abs_value < 10000:
        return round(value, 1)
    # 大于10000：四舍五入到整数
    else:
        return round(value)


def normalize_unit(
    test_name: str, 
    value: float, 
    unit: str,
    use_umls: bool = True,
    use_llm: bool = True,
    loinc_code: Optional[str] = None,
    return_source: bool = False,
    prefer_loinc: bool = None
) -> Tuple[Optional[float], Optional[str], ...]:
    """
    将医学检验值转换为标准单位 (SI单位或医学常用标准单位)
    
    支持的转换包括：
    - 血糖: mg/dL → mmol/L
    - 胆固醇: mg/dL → mmol/L
    - 血常规: /mm³ → ×10^9/L 或 ×10^12/L
    - 血红蛋白: g/dL → g/L
    - 肾功能: mg/dL → μmol/L
    - 肝功能: mg/dL → μmol/L
    - 电解质: mEq/L → mmol/L
    - 温度: °F → °C
    - 等等...
    
    转换策略（优先级）：
    1. 本地规则库（精确、快速）
    2. LLM Fallback（当本地规则未覆盖时）
    
    Args:
        test_name (str): 检验项目名称 (如"血糖", "WBC", "Hb")
        value (float): 数值
        unit (str): 原单位 (如"mg/dL", "/ul", "°F")
        use_umls (bool): 是否使用UMLS API (默认True，但实际上LOINC不提供单位转换)
        use_llm (bool): 是否使用LLM作为fallback (默认True)
        loinc_code (Optional[str]): LOINC编码（如果已知）
        return_source (bool): 是否返回转换来源 (默认False)
        prefer_loinc (bool): 是否优先LOINC (从环境变量LOINC_FIRST获取，默认False)
        
    Returns:
        如果return_source=False: Tuple[float, str] - (转换后的值, 标准单位)
        如果return_source=True: Tuple[float, str, str] - (转换后的值, 标准单位, 来源)
        来源可以是 "local" (本地规则), "llm" (LLM转换), None (未转换)
    """
    if not test_name or not unit:
        if return_source:
            return value, unit, None
        return value, unit
    
    # 确保 value 是数值类型
    try:
        value = float(value)
    except (ValueError, TypeError):
        # 非数值（如"尚未分析"）直接返回原值
        if return_source:
            return value, unit, None
        return value, unit
    
    test_name_lower = test_name.lower()
    # 标准化单位字符串
    unit_stripped = unit.strip()
    unit_normalized = _normalize_unit_string(unit_stripped)
    verbose = os.environ.get("UMLS_VERBOSE", "false").lower() == "true"
    
    # ========================================================================
    # 检查是否是医学标准单位（不需要转换）
    # ========================================================================
    # 如果单位已经是临床标准单位，直接返回原值，不进行 LLM fallback
    if unit_stripped in STANDARD_UNITS_NO_CONVERSION or unit_normalized in STANDARD_UNITS_NO_CONVERSION:
        if verbose:
            print(f"[量纲统一] 跳过转换: {test_name} ({value} {unit}) - 已经是医学标准单位")
        if return_source:
            return value, unit, "standard"  # 标记为已经是标准单位
        return value, unit
    
    def try_local_rules():
        """尝试使用本地规则进行转换"""
        for rule_key, rule in UNIT_CONVERSION_RULES.items():
            keywords = rule.get("keywords", [])
            from_units = rule.get("from", [])
            
            # 检查检验名称是否匹配任何关键词
            matches_keyword = False
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # 双向匹配：关键词在名称中，或名称在关键词中
                if keyword_lower in test_name_lower or test_name_lower in keyword_lower:
                    matches_keyword = True
                    break
            
            if not matches_keyword:
                continue
            
            # 检查单位是否匹配（同时检查原始单位和标准化后的单位）
            unit_matches = False
            for from_unit in from_units:
                from_unit_normalized = _normalize_unit_string(from_unit)
                if (unit_normalized == from_unit_normalized or 
                    unit.strip() == from_unit or 
                    unit_normalized == from_unit):
                    unit_matches = True
                    break
            
            if not unit_matches:
                continue
            
            # 执行转换
                factor = rule["factor"]
            try:
                if callable(factor):
                    # 特殊转换函数（如温度）
                    normalized_value = factor(value)
                else:
                    normalized_value = value * factor
                
                # 智能四舍五入
                normalized_value = _smart_round(normalized_value)
                
                if verbose:
                    print(f"[本地规则] ✓ {test_name}: {value} {unit} → {normalized_value} {rule['to']}")
                
                return normalized_value, rule["to"], "local"
            except Exception as e:
                if verbose:
                    print(f"[本地规则] 转换计算错误: {e}")
                continue
        
        return None, None, None
    
    def try_llm_conversion():
        """尝试使用LLM进行转换"""
        if not use_llm:
            return None, None, None
        
        try:
            # 尝试导入 LLM 转换函数
            try:
                from tools.tools_standardization_llm import normalize_unit_with_llm as llm_normalize
                import asyncio
            except ImportError:
                if verbose:
                    print("[LLM] 无法导入LLM转换模块")
                return None, None, None
            
            if verbose:
                print(f"[LLM] 尝试使用LLM转换: {test_name} ({value} {unit})")
            
            # 运行异步函数
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                norm_val, norm_unit, source = loop.run_until_complete(
                    llm_normalize(test_name, value, unit, verbose=verbose)
                )
            finally:
                loop.close()
            
            if source and norm_unit != unit:
                if verbose:
                    print(f"[LLM] ✓ 转换成功: {value} {unit} → {norm_val} {norm_unit}")
                return norm_val, norm_unit, "llm"
            elif verbose:
                print(f"[LLM] 未找到转换规则或无需转换")
                
        except Exception as e:
            if verbose:
                print(f"[LLM] 转换失败: {e}")
        
        return None, None, None
    
    # 策略：优先使用本地规则（更准确、更快）
    # 1. 本地规则
    norm_val, norm_unit, source = try_local_rules()
    if source:
        if return_source:
            return norm_val, norm_unit, source
        return norm_val, norm_unit
    
    # 2. LLM Fallback
    if use_llm:
        if verbose:
            print(f"[量纲统一] 本地规则未匹配，尝试LLM转换")
        norm_val, norm_unit, source = try_llm_conversion()
        if source:
            if return_source:
                return norm_val, norm_unit, source
            return norm_val, norm_unit
    
    # 未找到匹配的转换规则，返回原值
    if verbose:
        print(f"[量纲统一] 未找到转换规则: {test_name} ({unit})")
    
    if return_source:
        return value, unit, None
    return value, unit


def get_conversion_info(test_name: str, unit: str) -> Optional[Dict]:
    """
    获取指定检验项目和单位的转换信息（不执行转换）
    
    Args:
        test_name: 检验项目名称
        unit: 原单位
        
    Returns:
        转换信息字典，包含 from_unit, to_unit, factor, rule_key
        如果未找到匹配规则返回 None
    """
    test_name_lower = test_name.lower()
    unit_normalized = _normalize_unit_string(unit.strip())
    
    for rule_key, rule in UNIT_CONVERSION_RULES.items():
        keywords = rule.get("keywords", [])
        from_units = rule.get("from", [])
        
        # 检查名称匹配
        matches_keyword = any(
            keyword.lower() in test_name_lower or test_name_lower in keyword.lower()
            for keyword in keywords
        )
        
        if not matches_keyword:
            continue
        
        # 检查单位匹配
        for from_unit in from_units:
            from_unit_normalized = _normalize_unit_string(from_unit)
            if (unit_normalized == from_unit_normalized or 
                unit.strip() == from_unit):
                factor = rule["factor"]
                factor_str = "函数" if callable(factor) else str(factor)
                return {
                    "rule_key": rule_key,
                    "from_unit": from_unit,
                    "to_unit": rule["to"],
                    "factor": factor_str,
                    "keywords": keywords
                }
    
    return None
