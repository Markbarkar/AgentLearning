# 预处理层构建文档

## 概述

预处理层负责将原始法律文档转换为结构化的文档块，包括：
1. **法律文档分块器 (LegalChunker)** - 按法律条款结构智能分块
2. **元数据提取器 (LegalMetadataExtractor)** - 提取法规名称、地区、类型等结构化信息

## 目录结构

```
agent_system/rag/
├── chunkers/
│   ├── __init__.py
│   ├── base.py              # 分块器抽象基类
│   └── legal_chunker.py     # 法律文档分块器
│
└── extractors/
    ├── __init__.py
    ├── base.py              # 提取器抽象基类
    └── legal_metadata.py    # 法律元数据提取器
```

## 组件详情

### 1. ChunkerBase (抽象基类)

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.documents import Document

class ChunkerBase(ABC):
    """分块器抽象基类"""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        将文本分块
        
        Args:
            text: 原始文本
            metadata: 文档元数据
            
        Returns:
            Document 列表
        """
        pass
```

### 2. LegalChunker (法律文档分块器)

**核心功能：**
- 识别法律文档结构（章、节、条、款、项）
- 按条款边界分块，保持语义完整性
- 处理过长条款的二次分割（最大 1500 字）
- 为每个 chunk 添加父级上下文信息

**识别模式：**
| 结构 | 正则模式 | 示例 |
|-----|---------|------|
| 章 | `^第[一二三四五六七八九十百]+章\s+.+` | 第一章 总则 |
| 节 | `^第[一二三四五六七八九十百]+节\s+.+` | 第一节 一般规定 |
| 条 | `^第[一二三四五六七八九十\d]+条\s*` | 第十五条 |
| 款 | `^[（\(][一二三四五六七八九十\d]+[）\)]` | （一）... |
| 项 | `^[1-9][\.\、]` | 1. ... |

**输出元数据：**
```python
{
    "chapter": "第一章 总则",      # 所属章
    "section": "第一节 一般规定",  # 所属节
    "article_num": "第十五条",     # 条款号
    "chunk_id": 0,                 # 块序号
    "total_chunks": 10,            # 总块数
    "parent_context": "...",       # 父级上下文摘要
}
```

### 3. ExtractorBase (抽象基类)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class ExtractorBase(ABC):
    """元数据提取器抽象基类"""
    
    @abstractmethod
    def extract(self, file_path: str, text: str) -> Dict[str, Any]:
        """
        从文档提取元数据
        
        Args:
            file_path: 文件路径
            text: 文档内容
            
        Returns:
            结构化元数据字典
        """
        pass
```

### 4. LegalMetadataExtractor (法律元数据提取器)

**提取字段：**
| 字段 | 类型 | 提取方式 | 示例 |
|-----|------|---------|------|
| law_name | str | 文件名/标题 | 深圳经济特区消防条例 |
| law_type | str | 规则匹配 | 条例/规定/办法/决定 |
| region | str | 文件名/关键词 | 深圳/海南/厦门 |
| region_level | str | 规则推断 | 经济特区/自治州/自治县 |
| topics | List[str] | 关键词分类 | ["消防", "安全"] |
| publish_date | str | 正则提取 | 2023-01-01 |
| issuing_authority | str | 正则提取 | XX人民代表大会常务委员会 |

**地区识别规则：**
- 经济特区：深圳、海南、厦门、珠海
- 自治州：伊犁哈萨克自治州、海北藏族自治州...
- 自治县：互助土族自治县、巴里坤哈萨克自治县...

## 配置文件

### legal_regions.json

```json
{
  "economic_zones": ["深圳", "海南", "厦门", "珠海"],
  "autonomous_prefectures": [
    "伊犁哈萨克自治州",
    "海北藏族自治州",
    "海南藏族自治州",
    "海西蒙古族藏族自治州",
    "玉树藏族自治州",
    "黄南藏族自治州",
    "果洛藏族自治州"
  ],
  "autonomous_counties": [
    "互助土族自治县",
    "化隆回族自治县",
    "大通回族土族自治县",
    "巴里坤哈萨克自治县",
    "木垒哈萨克自治县",
    "察布查尔锡伯自治县"
  ],
  "topic_keywords": {
    "environment": ["环境", "环保", "污染", "生态", "排放"],
    "water": ["水资源", "水利", "供水", "排水", "河道", "水条例"],
    "forestry": ["森林", "林地", "林木", "林业"],
    "fire": ["消防", "防火", "灭火"],
    "traffic": ["交通", "道路", "公路", "车辆"],
    "construction": ["建设", "建筑", "工程", "施工"],
    "education": ["教育", "学校", "义务教育"],
    "health": ["卫生", "医疗", "健康", "防疫"]
  },
  "law_types": {
    "条例": "regulation",
    "规定": "provision",
    "办法": "measure",
    "决定": "decision",
    "细则": "detailed_rules"
  }
}
```

## 使用示例

```python
from agent_system.rag.chunkers import LegalChunker
from agent_system.rag.extractors import LegalMetadataExtractor

# 初始化
chunker = LegalChunker(max_chunk_size=1500, overlap=100)
extractor = LegalMetadataExtractor()

# 读取文档
with open("深圳经济特区消防条例_.docx", "r") as f:
    text = f.read()

# 提取元数据
metadata = extractor.extract(
    file_path="深圳经济特区消防条例_.docx",
    text=text
)
# 结果: {
#     "law_name": "深圳经济特区消防条例",
#     "law_type": "条例",
#     "region": "深圳",
#     "region_level": "经济特区",
#     "topics": ["消防", "安全"],
#     ...
# }

# 智能分块
documents = chunker.chunk(text, metadata)
# 每个 document 包含:
# - page_content: 条款内容
# - metadata: 合并了提取的元数据 + 分块元数据
```

## 测试用例

```python
def test_legal_chunker():
    """测试法律文档分块"""
    text = """
    第一章 总则
    
    第一条 为了加强消防工作，预防火灾和减少火灾危害，保护人身、财产安全，
    维护公共安全，根据《中华人民共和国消防法》及有关法律、行政法规的基本原则，
    结合深圳经济特区实际，制定本条例。
    
    第二条 本条例适用于深圳经济特区。
    
    第二章 火灾预防
    
    第三条 任何单位和个人都有维护消防安全、保护消防设施、预防火灾、
    报告火警的义务。
    """
    
    chunker = LegalChunker()
    chunks = chunker.chunk(text, {"source": "test.docx"})
    
    assert len(chunks) >= 2
    assert chunks[0].metadata["chapter"] == "第一章 总则"
    assert "第一条" in chunks[0].page_content

def test_metadata_extractor():
    """测试元数据提取"""
    extractor = LegalMetadataExtractor()
    
    metadata = extractor.extract(
        file_path="深圳经济特区消防条例_.docx",
        text="..."
    )
    
    assert metadata["law_name"] == "深圳经济特区消防条例"
    assert metadata["law_type"] == "条例"
    assert metadata["region"] == "深圳"
    assert metadata["region_level"] == "经济特区"
    assert "消防" in metadata["topics"]
```

## 验收标准

- [ ] LegalChunker 能正确识别章、节、条结构
- [ ] 分块后每个 chunk 保持语义完整（不切断条款）
- [ ] 过长条款能正确二次分割
- [ ] LegalMetadataExtractor 能从文件名提取法规名称
- [ ] 能正确识别地区（经济特区/自治州/自治县）
- [ ] 能正确分类法规类型（条例/规定/办法/决定）
- [ ] 能提取主题标签
- [ ] 单元测试通过率 100%
