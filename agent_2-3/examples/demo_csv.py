# -*- coding: utf-8 -*-
"""
CSV处理演示脚本

展示如何使用medical_data_cleaner处理CSV文件。
"""
import asyncio
import os
import sys

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 添加medical_agent_system目录
agent_system_dir = os.path.dirname(parent_dir)
sys.path.insert(0, agent_system_dir)


async def demo_csv_analysis():
    """演示CSV文件分析"""
    from tools.csv_reader import get_csv_info_for_agent
    from tools.data_type_detector import detect_input_type
    
    # 示例CSV文件路径
    csv_file = "/path/to/your/data.csv"
    
    print("=" * 60)
    print("1. 数据类型检测")
    print("=" * 60)
    
    detection_result = detect_input_type(csv_file)
    print(f"检测到类型: {detection_result['type']}")
    print(f"处理流程: {detection_result['processing_stages']}")
    
    print("\n" + "=" * 60)
    print("2. CSV文件详细分析")
    print("=" * 60)
    
    csv_info = get_csv_info_for_agent(csv_file)
    print(csv_info)


async def demo_csv_processing():
    """演示CSV文件处理"""
    from processors.csv_processor import CSVProcessor
    
    csv_file = "/path/to/your/data.csv"
    
    print("\n" + "=" * 60)
    print("3. CSV文件处理（仅处理前5行作为演示）")
    print("=" * 60)
    
    processor = CSVProcessor(
        use_llm=True,  # 使用LLM进行智能标准化
        use_umls=True,  # 使用UMLS API
        verbose=True    # 详细输出
    )
    
    result = await processor.process_file(
        file_path=csv_file,
        # 指定需要标准化的列
        standardization_columns=["test_name", "org_name", "ab_name"],
        # 指定需要量纲统一的列
        value_columns=["dilution_value"],
        # 单位列映射
        unit_columns={"dilution_value": "dilution_text"},
        # 类别映射
        category_mapping={
            "test_name": "Test",
            "org_name": "Test",  # 微生物也归类为检验项目
            "ab_name": "Drug"   # 抗生素归类为药品
        },
        max_rows=5  # 仅处理5行演示
    )
    
    print(f"\n处理结果:")
    print(f"  成功: {result['success']}")
    print(f"  处理行数: {result['statistics'].get('total_rows', 0)}")
    print(f"  术语标准化成功: {result['statistics'].get('terms_standardized', 0)}")
    print(f"  术语未找到: {result['statistics'].get('terms_not_found', 0)}")
    print(f"  量纲统一: {result['statistics'].get('values_normalized', 0)}")
    
    if result.get("data") and len(result["data"]) > 0:
        print(f"\n处理后的第一行数据示例:")
        first_row = result["data"][0]
        for key, value in first_row.items():
            if value:  # 只显示非空值
                print(f"  {key}: {value}")


async def demo_unified_agent():
    """演示统一处理Agent"""
    from agents.unified_processing_agent import process_medical_data
    
    csv_file = "/path/to/your/data.csv"
    
    print("\n" + "=" * 60)
    print("4. 统一处理Agent演示")
    print("=" * 60)
    
    result = await process_medical_data(
        input_path=csv_file,
        use_llm=True,
        verbose=True
    )
    
    print(f"\n处理完成!")
    print(f"数据类型: {result.get('data_type')}")
    print(f"成功: {result.get('success')}")


async def main():
    """主函数"""
    print("Medical Data Cleaner - CSV处理演示")
    print("=" * 60)
    
    # 1. 分析CSV文件
    await demo_csv_analysis()
    
    # 2. 处理CSV文件
    await demo_csv_processing()
    
    # 3. 统一Agent处理（可选，需要更多时间）
    # await demo_unified_agent()
    
    print("\n演示完成!")


if __name__ == "__main__":
    asyncio.run(main())
