# -*- coding: utf-8 -*-
"""
医学检验单位转换测试
验证本地规则库的完整性和计算准确性
"""

import os
import sys

# 设置导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

os.environ["UMLS_VERBOSE"] = "false"
os.environ["USE_UMLS"] = "false"


def test_basic_conversions():
    """测试基础转换功能"""
    print("=" * 80)
    print("【测试 1】基础单位转换")
    print("=" * 80)
    
    from tools.tools_standardization import normalize_unit
    
    # 测试用例格式: (检验名称, 原值, 原单位, 预期值, 预期单位, 允许误差%)
    test_cases = [
        # 血糖
        ("血糖", 180, "mg/dL", 9.99, "mmol/L", 1),
        ("Glucose", 126, "mg/dL", 6.99, "mmol/L", 1),
        ("空腹血糖", 100, "mg/dL", 5.55, "mmol/L", 1),
        
        # 血脂
        ("总胆固醇", 200, "mg/dL", 5.17, "mmol/L", 1),
        ("TC", 240, "mg/dL", 6.21, "mmol/L", 1),
        ("甘油三酯", 150, "mg/dL", 1.69, "mmol/L", 2),
        ("TG", 200, "mg/dL", 2.26, "mmol/L", 2),
        ("HDL", 50, "mg/dL", 1.29, "mmol/L", 1),
        ("LDL", 130, "mg/dL", 3.36, "mmol/L", 1),
        
        # 血常规
        ("白细胞", 8000, "/mm³", 8.0, "×10^9/L", 1),
        ("WBC", 12000, "/mm3", 12.0, "×10^9/L", 1),
        ("白细胞计数", 5.5, "×10³/μL", 5.5, "×10^9/L", 1),
        ("红细胞", 4500000, "/mm³", 4.5, "×10^12/L", 1),
        ("RBC", 5.0, "×10^6/μL", 5.0, "×10^12/L", 1),
        ("血小板", 250000, "/mm³", 250, "×10^9/L", 1),
        ("PLT", 180, "×10³/μL", 180, "×10^9/L", 1),
        ("血红蛋白", 15, "g/dL", 150, "g/L", 1),
        ("Hb", 12.5, "g/dL", 125, "g/L", 1),
        ("红细胞压积", 45, "%", 0.45, "L/L", 1),
        
        # 肾功能
        ("肌酐", 1.0, "mg/dL", 88.4, "μmol/L", 1),
        ("Creatinine", 1.5, "mg/dL", 132.6, "μmol/L", 1),
        ("尿素氮", 15, "mg/dL", 5.36, "mmol/L", 2),
        ("BUN", 20, "mg/dL", 7.14, "mmol/L", 2),
        ("尿酸", 7.0, "mg/dL", 416.36, "μmol/L", 1),
        
        # 肝功能
        ("总胆红素", 1.0, "mg/dL", 17.1, "μmol/L", 1),
        ("TBIL", 2.0, "mg/dL", 34.2, "μmol/L", 1),
        ("白蛋白", 4.0, "g/dL", 40, "g/L", 1),
        ("总蛋白", 7.0, "g/dL", 70, "g/L", 1),
        
        # 电解质
        ("钠", 140, "mEq/L", 140, "mmol/L", 0.1),
        ("钾", 4.0, "mEq/L", 4.0, "mmol/L", 0.1),
        ("钙", 10, "mg/dL", 2.5, "mmol/L", 1),
        ("磷", 3.5, "mg/dL", 1.13, "mmol/L", 2),
        ("镁", 2.0, "mg/dL", 0.82, "mmol/L", 2),
        
        # 温度
        ("体温", 98.6, "°F", 37.0, "°C", 1),
        ("temperature", 104, "F", 40.0, "°C", 1),
    ]
    
    passed = 0
    failed = 0
    
    print(f"\n{'检验项目':<15} {'原值':>10} {'原单位':<12} {'转换值':>12} {'预期值':>10} {'单位':<12} {'状态':<6}")
    print("-" * 90)
    
    for test_name, value, unit, expected_value, expected_unit, tolerance in test_cases:
        norm_val, norm_unit, source = normalize_unit(test_name, value, unit, 
                                                     use_llm=False, return_source=True)
        
        # 检查单位是否匹配
        unit_match = norm_unit == expected_unit
        
        # 检查值是否在允许误差范围内
        if expected_value != 0:
            error_percent = abs(norm_val - expected_value) / expected_value * 100
            value_match = error_percent <= tolerance
        else:
            value_match = abs(norm_val - expected_value) < 0.01
        
        if unit_match and value_match:
            status = "✓"
            passed += 1
        else:
            status = "✗"
            failed += 1
        
        print(f"{test_name:<15} {value:>10} {unit:<12} {norm_val:>12.2f} {expected_value:>10.2f} {norm_unit:<12} {status:<6}")
    
    print("-" * 90)
    print(f"通过: {passed}/{passed+failed}, 失败: {failed}/{passed+failed}")
    print()
    
    return passed, failed


def test_unit_aliases():
    """测试单位别名识别"""
    print("=" * 80)
    print("【测试 2】单位别名识别")
    print("=" * 80)
    
    from tools.tools_standardization import normalize_unit, _normalize_unit_string
    
    # 测试单位标准化
    alias_tests = [
        ("mg/dl", "mg/dL"),
        ("MG/DL", "mg/dL"),
        ("g/dl", "g/dL"),
        ("ug/dL", "μg/dL"),
        ("mcg/dL", "μg/dL"),
        ("ng/ml", "ng/mL"),
        ("umol/L", "μmol/L"),
        ("/mm3", "/mm³"),
        ("/uL", "/μL"),
        ("10^3/uL", "×10³/μL"),
        ("K/uL", "×10³/μL"),
        ("meq/L", "mEq/L"),
        ("F", "°F"),
    ]
    
    print(f"\n{'原单位':<15} {'标准化后':<15} {'状态':<6}")
    print("-" * 40)
    
    passed = 0
    for original, expected in alias_tests:
        result = _normalize_unit_string(original)
        status = "✓" if result == expected else "✗"
        if status == "✓":
            passed += 1
        print(f"{original:<15} {result:<15} {status:<6}")
    
    print("-" * 40)
    print(f"通过: {passed}/{len(alias_tests)}")
    print()
    
    # 测试实际转换是否能识别别名
    print("验证别名单位能正确转换:")
    print("-" * 60)
    
    alias_conversion_tests = [
        ("血糖", 100, "mg/dl"),  # 小写
        ("血糖", 100, "MG/DL"),  # 大写
        ("WBC", 8000, "/uL"),    # uL 变体
        ("WBC", 8, "K/uL"),      # K/uL 变体
        ("肌酐", 1.0, "mg/dl"),  # 小写
    ]
    
    for test_name, value, unit in alias_conversion_tests:
        norm_val, norm_unit, source = normalize_unit(test_name, value, unit, 
                                                     use_llm=False, return_source=True)
        status = "✓" if source == "local" else "✗"
        print(f"  {test_name}: {value} {unit} → {norm_val} {norm_unit} (来源: {source}) {status}")
    
    print()


def test_precision():
    """测试计算精度"""
    print("=" * 80)
    print("【测试 3】计算精度验证")
    print("=" * 80)
    
    from tools.tools_standardization import normalize_unit, _smart_round
    
    # 测试智能四舍五入
    print("\n智能四舍五入测试:")
    round_tests = [
        (0.00012345, 0.0001235),  # 很小的数
        (0.5678, 0.5678),         # 小于1
        (12.3456, 12.35),         # 1-100
        (123.456, 123.5),         # 100-10000
        (12345.6, 12346),         # 大于10000
    ]
    
    print(f"  {'输入值':<15} {'输出值':<15} {'预期值':<15} {'状态':<6}")
    for input_val, expected in round_tests:
        result = _smart_round(input_val)
        status = "✓" if abs(result - expected) < 0.0001 else "✗"
        print(f"  {input_val:<15} {result:<15} {expected:<15} {status:<6}")
    
    # 测试精确转换计算
    print("\n精确转换计算:")
    precision_tests = [
        # 血糖: 分子量 180.16, 1 mg/dL = 10/180.16 mmol/L = 0.05551 mmol/L
        ("血糖", 180.16, "mg/dL", 10.0, "mmol/L", "1 mg/dL = 0.05551 mmol/L"),
        # 胆固醇: 分子量 386.65, 1 mg/dL = 10/386.65 mmol/L = 0.02586 mmol/L
        ("胆固醇", 386.65, "mg/dL", 10.0, "mmol/L", "1 mg/dL = 0.02586 mmol/L"),
        # 肌酐: 分子量 113.12, 1 mg/dL = 10000/113.12 μmol/L = 88.4 μmol/L
        ("肌酐", 1.0, "mg/dL", 88.4, "μmol/L", "1 mg/dL = 88.4 μmol/L"),
        # 胆红素: 分子量 584.66, 1 mg/dL = 10000/584.66 μmol/L = 17.1 μmol/L
        ("胆红素", 1.0, "mg/dL", 17.1, "μmol/L", "1 mg/dL = 17.1 μmol/L"),
    ]
    
    print(f"\n  {'检验项目':<10} {'原值':>10} {'原单位':<10} {'转换值':>10} {'预期值':>10} {'单位':<10} {'公式说明':<30}")
    for test_name, value, unit, expected_value, expected_unit, formula in precision_tests:
        norm_val, norm_unit, _ = normalize_unit(test_name, value, unit, 
                                                use_llm=False, return_source=True)
        error = abs(norm_val - expected_value)
        status = "✓" if error < 0.1 else "✗"
        print(f"  {test_name:<10} {value:>10} {unit:<10} {norm_val:>10.2f} {expected_value:>10.2f} {norm_unit:<10} {formula:<30} {status}")
    
    print()


def test_get_conversion_info():
    """测试获取转换信息功能"""
    print("=" * 80)
    print("【测试 4】转换信息查询")
    print("=" * 80)
    
    from tools.tools_standardization import get_conversion_info
    
    queries = [
        ("血糖", "mg/dL"),
        ("WBC", "/mm3"),
        ("Hemoglobin", "g/dL"),
        ("肌酐", "mg/dL"),
        ("钠", "mEq/L"),
        ("体温", "°F"),
        ("未知检验", "unknown_unit"),  # 应该返回 None
    ]
    
    print(f"\n{'检验项目':<15} {'原单位':<12} {'目标单位':<12} {'转换因子':<15} {'匹配规则':<15}")
    print("-" * 75)
    
    for test_name, unit in queries:
        info = get_conversion_info(test_name, unit)
        if info:
            print(f"{test_name:<15} {unit:<12} {info['to_unit']:<12} {info['factor']:<15} {info['rule_key']:<15}")
        else:
            print(f"{test_name:<15} {unit:<12} {'--':<12} {'--':<15} {'未找到':<15}")
    
    print()


def test_coverage():
    """测试规则覆盖范围"""
    print("=" * 80)
    print("【测试 5】规则覆盖范围统计")
    print("=" * 80)
    
    from tools.tools_standardization import UNIT_CONVERSION_RULES
    
    # 按类别统计
    categories = {
        "血糖": ["glucose"],
        "血脂": ["cholesterol", "triglycerides", "hdl", "ldl"],
        "血常规": ["wbc", "wbc_k", "rbc", "rbc_m", "platelet", "platelet_k", "hemoglobin", "hematocrit"],
        "肾功能": ["creatinine", "bun", "uric_acid"],
        "肝功能": ["bilirubin", "albumin", "total_protein"],
        "电解质": ["sodium", "potassium", "chloride", "calcium", "calcium_meq", "phosphorus", "magnesium", "magnesium_meq"],
        "心肌标志物": ["troponin", "ck", "bnp"],
        "凝血功能": ["fibrinogen", "d_dimer", "d_dimer_ng"],
        "甲状腺功能": ["t4", "t3"],
        "激素": ["cortisol", "testosterone", "estradiol"],
        "铁代谢": ["iron", "ferritin"],
        "炎症标志物": ["crp"],
        "血气分析": ["pO2", "pCO2"],
        "体温": ["temperature_f_to_c", "temperature_c_to_f"],
        "体重身高": ["weight_lb_to_kg", "height_in_to_cm"],
    }
    
    print(f"\n{'类别':<15} {'规则数':<10} {'包含规则':<60}")
    print("-" * 90)
    
    total_rules = 0
    for category, rule_keys in categories.items():
        # 检查哪些规则存在
        existing = [k for k in rule_keys if k in UNIT_CONVERSION_RULES]
        total_rules += len(existing)
        print(f"{category:<15} {len(existing):<10} {', '.join(existing):<60}")
    
    print("-" * 90)
    print(f"总计: {total_rules} 条转换规则")
    print(f"实际规则数: {len(UNIT_CONVERSION_RULES)} 条")
    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("         医学检验单位转换测试套件")
    print("=" * 80 + "\n")
    
    # 测试1: 基础转换
    passed, failed = test_basic_conversions()
    
    # 测试2: 单位别名
    test_unit_aliases()
    
    # 测试3: 计算精度
    test_precision()
    
    # 测试4: 转换信息查询
    test_get_conversion_info()
    
    # 测试5: 规则覆盖范围
    test_coverage()
    
    # 总结
    print("=" * 80)
    print("         测试总结")
    print("=" * 80)
    print(f"\n基础转换测试: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过！本地规则库工作正常。")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查转换规则。")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
