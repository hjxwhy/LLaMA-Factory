# 数据处理脚本重构说明

本文档说明了重构后的数据处理脚本的使用方法。重构后的脚本整合了三个原始脚本的功能：
- `process_func_mix.py` (原始功能)
- `random_combine.py` (随机合并功能)  
- `qwen2_5_vl_mix.py` (Qwen VL混合功能)

## 主要改进

1. **统一的配置管理**: 所有路径和参数通过配置文件或配置类管理
2. **面向对象设计**: 使用 `DataMixer` 类封装所有功能
3. **模块化功能**: 支持三种处理模式的选择
4. **更好的错误处理**: 添加了日志记录和异常处理
5. **灵活的配置**: 支持通过命令行参数覆盖配置

## 文件结构

```
scripts/
├── process_func_mix_refactored.py    # 重构后的主脚本
├── common/
│   └── config.py                     # 配置管理模块
├── config_templates/
│   ├── data_mix_config.json          # 配置文件模板
│   └── custom_config.json            # 自定义配置示例
└── README_refactored.md              # 本文档
```

## 配置说明

### 1. 配置文件结构

配置文件使用JSON格式，包含以下主要部分：

```json
{
  "paths": {
    "llama_factory_root": "/path/to/LLaMA-Factory",
    "data_root": "/path/to/data",
    "output_path": "/path/to/output"
  },
  "dataset_paths": {
    "unitree_audio_user_path": "...",
    "voice_assistant_path": "...",
    "...": "..."
  },
  "processing_params": {
    "num_tulu_tools_samples": 50000,
    "num_llava_tools_samples": 20000,
    "...": "..."
  },
  "random_combine_params": {
    "num_unitree_samples": 500,
    "num_fight_qa_samples": 200,
    "...": "..."
  },
  "qwen_vl_params": {
    "tulu_subset_ratio": 0.2,
    "robospatial_subset_ratio": 0.5,
    "...": "..."
  }
}
```

### 2. 路径配置

配置中的所有路径都相对于 `llama_factory_root` 或 `data_root`：

- `llama_factory_root`: LLaMA-Factory项目根目录
- `data_root`: 数据存储根目录 (通常是 `{llama_factory_root}/data/data`)
- `output_path`: 输出目录

## 使用方法

### 1. 基本用法

```bash
# 运行完整的数据处理流程（原始功能）
python scripts/process_func_mix_refactored.py --mode full

# 运行随机合并功能
python scripts/process_func_mix_refactored.py --mode random_combine

# 运行Qwen VL混合功能  
python scripts/process_func_mix_refactored.py --mode qwen_vl_mix
```

### 2. 使用自定义配置文件

```bash
# 使用自定义配置文件
python scripts/process_func_mix_refactored.py \
    --mode full \
    --config-file scripts/config_templates/custom_config.json
```

### 3. 通过命令行覆盖配置

```bash
# 覆盖特定配置项
python scripts/process_func_mix_refactored.py \
    --mode full \
    --output-path /custom/output/path \
    --data-root /custom/data/root \
    --llama-factory-root /custom/llama/factory
```

### 4. 完整示例

```bash
# 使用自定义配置运行随机合并，并覆盖输出路径
python scripts/process_func_mix_refactored.py \
    --mode random_combine \
    --config-file scripts/config_templates/custom_config.json \
    --output-path /data/mixed_datasets/random_combined_v1
```

## 三种处理模式详解

### 1. Full Mode (完整模式)

这是原始 `process_func_mix.py` 的功能，处理多种数据集并合并：

**处理的数据集:**
- Unitree toolcall 数据集
- Tulu 数据集
- Glaive function calling 数据集
- XLAM 数据集
- LLaVA OneVision 数据集
- Voice assistant 数据集
- System prompt 数据集

**主要功能:**
- 随机添加工具到样本
- 添加系统提示
- 数据增强
- 合并所有数据集

**输出:** 保存为多个parquet分片文件

### 2. Random Combine Mode (随机合并模式)

这是原始 `random_combine.py` 的功能：

**处理流程:**
1. 加载Unitree数据集、Fight QA数据集和Voice Assistant数据集
2. 随机选择指定数量的对话
3. 将Voice Assistant对话随机插入到原始对话中
4. 生成合并后的数据集

**配置参数:**
- `num_unitree_samples`: Unitree样本数量
- `num_fight_qa_samples`: Fight QA样本数量  
- `min_va_conversations`/`max_va_conversations`: 插入的VA对话数量范围
- `duplicate_count`: 特殊样本复制次数

### 3. Qwen VL Mix Mode (Qwen VL混合模式)

这是原始 `qwen2_5_vl_mix.py` 的功能：

**处理的数据集:**
- ShareRobot 数据集
- LLaVA OneVision 数据集
- Pixmo-points 数据集
- RoboSpatial 数据集
- Tulu 文本数据集

**主要功能:**
- ShareRobot格式转换和步骤修正
- RoboSpatial图像路径处理
- 坐标标注处理
- 多模态数据混合

**配置参数:**
- `tulu_subset_ratio`: Tulu数据集子集比例
- `robospatial_subset_ratio`: RoboSpatial数据集子集比例

## 配置自定义

### 1. 创建自定义配置

复制配置模板并修改路径：

```bash
cp scripts/config_templates/data_mix_config.json scripts/config_templates/my_config.json
```

然后编辑 `my_config.json` 中的路径：

```json
{
  "paths": {
    "llama_factory_root": "/your/llama/factory/path",
    "data_root": "/your/data/path",
    "output_path": "/your/output/path"
  }
}
```

### 2. 常用配置项说明

**路径配置:**
- `llama_factory_root`: LLaMA-Factory根目录
- `data_root`: 数据根目录
- `output_path`: 输出目录

**数据集路径:** (相对于data_root)
- `unitree_audio_user_path`: Unitree音频用户数据
- `voice_assistant_path`: 语音助手数据
- `tulu_path`: Tulu数据集
- `glaive_func_path`: Glaive函数调用数据

**处理参数:**
- `num_*_samples`: 各种数据集的样本数量
- `*_subset_ratio`: 数据集子集比例

## 依赖要求

确保安装了所有必需的依赖：

```bash
pip install datasets torch transformers
pip install pillow  # 用于图像处理
pip install qwen-vl-utils  # Qwen VL功能（可选）
```

## 日志和调试

脚本包含详细的日志记录：

```bash
# 查看详细日志
python scripts/process_func_mix_refactored.py --mode full 2>&1 | tee processing.log
```

日志包含：
- 数据集加载进度
- 处理步骤信息
- 错误和警告信息
- 最终输出统计

## 故障排除

### 常见问题

1. **路径不存在错误:**
   - 检查配置文件中的路径是否正确
   - 确保数据文件确实存在

2. **内存不足:**
   - 减少样本数量配置
   - 使用更小的子集比例

3. **依赖缺失:**
   - 安装所需的Python包
   - 检查CUDA版本兼容性

### 调试技巧

1. **测试小规模运行:**
   ```bash
   # 修改配置文件中的样本数量为较小值进行测试
   ```

2. **检查单个数据集:**
   ```bash
   # 在配置中禁用某些数据集路径来隔离问题
   ```

3. **查看详细错误信息:**
   ```bash
   # 使用Python调试模式
   python -u scripts/process_func_mix_refactored.py --mode full
   ```

## 性能优化

1. **内存管理:**
   - 脚本包含垃圾回收调用
   - 大数据集会分批处理

2. **并行处理:**
   - Datasets库自动使用多核处理
   - 可通过环境变量调整worker数量

3. **存储格式:**
   - 输出使用Parquet格式，便于快速加载
   - 自动分片处理大数据集

## 总结

重构后的脚本提供了：
- 统一的配置管理
- 灵活的功能选择
- 更好的错误处理
- 清晰的代码结构

通过选择不同的处理模式，您可以根据需要处理不同类型的数据集，而无需运行多个独立的脚本。 