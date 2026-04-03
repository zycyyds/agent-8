# -*- coding: utf-8 -*-
"""
测试量纲统一修复效果
验证医学标准单位不会被错误转换
"""
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 禁用 LLM fallback 进行基础测试
os.environ["UMLS_VERBOSE"] = "true"

from tools.tools_standardization import normalize_unit, STANDARD_UNITS_NO_CONVERSION

def test_standard_units():
    """测试标准单位不会被转换"""
    print("=" * 60)
    print("测试 1: 医学标准单位白名单")
    print("=" * 60)
    
    print(f"\n白名单中的单位 ({len(STANDARD_UNITS_NO_CONVERSION)} 个):")
    for unit in sorted(STANDARD_UNITS_NO_CONVERSION):
        print(f"  - {unit}")
    
    print("\n✓ 白名单加载成功")
    return True

def test_ophthalmology_units():
    """测试眼科检查单位不被转换"""
    print("\n" + "=" * 60)
    print("测试 2: 眼科检查角度/角速度单位")
    print("=" * 60)
    
    test_cases = [
        # (检验项目名, 值, 单位, 期望不转换)
        ("SPV水平峰值", -7, "度/s", True),
        ("SPV水平峰值", -13, "°/s", True),
        ("主观视觉水平试验-标准差", 2.2, "°", True),
        ("主观视觉垂直试验-翻滚角", 45.0, "度", True),
        ("前庭双温评价-角速度", 10, "度/秒", True),
    ]
    
    all_passed = True
    for name, value, unit, should_not_convert in test_cases:
        result = normalize_unit(name, value, unit, use_llm=False, return_source=True)
        norm_val, norm_unit, source = result
        
        # 检查是否保持原单位
        converted = (norm_unit != unit)
        
        if should_not_convert and not converted:
            status = "✓ PASS"
            detail = f"保持原单位 {unit}"
        elif should_not_convert and converted:
            status = "✗ FAIL"
            detail = f"被错误转换: {unit} → {norm_unit}"
            all_passed = False
        else:
            status = "?"
            detail = f"{unit} → {norm_unit}"
        
        print(f"  {status} {name}: {value} {unit} → {norm_val} {norm_unit} ({detail})")
    
    return all_passed

def test_blood_cell_units():
    """测试血细胞指标单位不被错误转换"""
    print("\n" + "=" * 60)
    print("测试 3: 血细胞指标标准单位 (pg, fL, %)")
    print("=" * 60)
    
    test_cases = [
        # (检验项目名, 值, 单位, 期望不转换)
        ("平均红细胞血红蛋白含量", 32.3, "pg", True),    # MCH
        ("平均红细胞血红蛋白含量", 66, "pg", True),      # MCH 异常值
        ("平均红细胞体积", 89.0, "fL", True),            # MCV
        ("淋巴细胞百分比", 29.4, "%", True),
        ("中性粒细胞百分比", 62.4, "%", True),
    ]
    
    all_passed = True
    for name, value, unit, should_not_convert in test_cases:
        result = normalize_unit(name, value, unit, use_llm=False, return_source=True)
        norm_val, norm_unit, source = result
        
        converted = (norm_unit != unit)
        
        if should_not_convert and not converted:
            status = "✓ PASS"
            detail = f"保持原单位 {unit}"
        elif should_not_convert and converted:
            status = "✗ FAIL"
            detail = f"被错误转换: {unit} → {norm_unit}"
            all_passed = False
        else:
            status = "?"
            detail = f"{unit} → {norm_unit}"
        
        print(f"  {status} {name}: {value} {unit} → {norm_val} {norm_unit} ({detail})")
    
    return all_passed

def test_non_numeric_values():
    """测试非数值数据被正确跳过"""
    print("\n" + "=" * 60)
    print("测试 4: 非数值数据处理")
    print("=" * 60)
    
    test_cases = [
        ("SPV水平峰值", "尚未分析", "度/s"),
        ("检测项目", "阴性", ""),
        ("血型", "A型", ""),
    ]
    
    all_passed = True
    for name, value, unit in test_cases:
        result = normalize_unit(name, value, unit, use_llm=False, return_source=True)
        norm_val, norm_unit, source = result
        
        # 非数值应该保持原样
        if norm_val == value:
            status = "✓ PASS"
            detail = "正确保持原值"
        else:
            status = "✗ FAIL"
            detail = f"值被修改: {value} → {norm_val}"
            all_passed = False
        
        print(f"  {status} {name}: '{value}' → '{norm_val}' ({detail})")
    
    return all_passed

def test_valid_conversions():
    """测试有效的单位转换仍然正常工作"""
    print("\n" + "=" * 60)
    print("测试 5: 有效的单位转换（确保没有破坏正常功能）")
    print("=" * 60)
    
    test_cases = [
        # (检验项目名, 值, 原单位, 期望转换后单位, 期望值范围)
        ("血糖", 100, "mg/dL", "mmol/L", (5.5, 5.6)),
        ("血红蛋白", 14.5, "g/dL", "g/L", (144, 146)),
        ("肌酐", 1.0, "mg/dL", "μmol/L", (88, 89)),
    ]
    
    all_passed = True
    for name, value, from_unit, expected_unit, expected_range in test_cases:
        result = normalize_unit(name, value, from_unit, use_llm=False, return_source=True)
        norm_val, norm_unit, source = result
        
        # 检查单位是否转换正确
        unit_correct = (norm_unit == expected_unit)
        # 检查数值是否在合理范围内
        value_correct = (expected_range[0] <= norm_val <= expected_range[1])
        
        if unit_correct and value_correct:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_passed = False
        
        print(f"  {status} {name}: {value} {from_unit} → {norm_val:.2f} {norm_unit} (期望: {expected_range[0]}-{expected_range[1]} {expected_unit})")
    
    return all_passed

def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("   量纲统一修复验证测试")
    print("=" * 70)
    
    results = []
    
    results.append(("医学标准单位白名单", test_standard_units()))
    results.append(("眼科检查角度单位", test_ophthalmology_units()))
    results.append(("血细胞指标单位", test_blood_cell_units()))
    results.append(("非数值数据处理", test_non_numeric_values()))
    results.append(("有效单位转换", test_valid_conversions()))
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("   测试结果汇总")
    print("=" * 70)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过！修复有效。")
    else:
        print("\n⚠️ 部分测试失败，请检查修复代码。")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
