"""
查询解析模块

负责解析用户查询，提取关键词、实体、过滤条件等
"""

import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class QueryType(Enum):
    """查询类型"""
    KEYWORD = "keyword"         # 关键词查询
    QUESTION = "question"       # 问题型查询
    ENTITY = "entity"           # 实体查询
    SEMANTIC = "semantic"       # 语义查询
    HYBRID = "hybrid"           # 混合查询


@dataclass
class ParsedQuery:
    """解析后的查询"""
    original_query: str                           # 原始查询
    normalized_query: str                         # 规范化后的查询
    query_type: QueryType = QueryType.HYBRID      # 查询类型
    keywords: List[str] = field(default_factory=list)  # 提取的关键词
    entities: Dict[str, List[str]] = field(default_factory=dict)  # 提取的实体
    filters: Dict[str, Any] = field(default_factory=dict)  # 过滤条件
    expansion_terms: List[str] = field(default_factory=list)  # 扩展词
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加信息
    
    @property
    def search_query(self) -> str:
        """用于搜索的查询"""
        if self.expansion_terms:
            return f"{self.normalized_query} {' '.join(self.expansion_terms)}"
        return self.normalized_query


class QueryParser:
    """
    查询解析器
    
    功能：
    - 查询规范化（去除多余空格、特殊字符等）
    - 关键词提取
    - 法律实体识别（法规名称、地区、条款号等）
    - 过滤条件提取
    - 查询扩展
    """
    
    # 法律相关实体模式
    LAW_NAME_PATTERNS = [
        r'《([^》]+)》',                    # 《民法典》
        r'(?:中华人民共和国)?(\w+法)',      # 民法、刑法
        r'(\w+条例)',                       # XX条例
        r'(\w+规定)',                       # XX规定
        r'(\w+办法)',                       # XX办法
    ]
    
    REGION_PATTERNS = [
        r'(北京|上海|天津|重庆)',
        r'(广东|江苏|浙江|山东|河南|四川|湖北|湖南|福建|安徽)',
        r'(河北|陕西|辽宁|江西|黑龙江|吉林|云南|贵州|山西|甘肃)',
        r'(广西|海南|宁夏|青海|西藏|新疆|内蒙古)',
        r'(深圳|广州|杭州|南京|成都|武汉|西安|苏州|厦门|宁波)',
    ]
    
    ARTICLE_PATTERNS = [
        r'第([一二三四五六七八九十百千\d]+)条',
        r'第([一二三四五六七八九十百千\d]+)章',
        r'第([一二三四五六七八九十百千\d]+)节',
    ]
    
    # 法律类型关键词
    LAW_TYPE_KEYWORDS = {
        "法律": ["法律", "法", "民法典", "刑法", "宪法"],
        "行政法规": ["行政法规", "条例", "国务院"],
        "地方法规": ["地方法规", "地方性法规", "省", "市", "自治区", "经济特区"],
        "司法解释": ["司法解释", "最高人民法院", "最高人民检察院"],
        "规章": ["规章", "部门规章", "地方规章"],
    }
    
    # 问题词
    QUESTION_WORDS = {"什么", "怎么", "如何", "哪些", "哪个", "为什么", "是否", "能否", "可以"}
    
    def __init__(
        self,
        enable_expansion: bool = True,
        max_keywords: int = 10,
    ):
        """
        初始化查询解析器
        
        Args:
            enable_expansion: 是否启用查询扩展
            max_keywords: 最大关键词数量
        """
        self.enable_expansion = enable_expansion
        self.max_keywords = max_keywords
        
        # 编译正则表达式
        self._law_patterns = [re.compile(p) for p in self.LAW_NAME_PATTERNS]
        self._region_patterns = [re.compile(p) for p in self.REGION_PATTERNS]
        self._article_patterns = [re.compile(p) for p in self.ARTICLE_PATTERNS]
    
    def parse(self, query: str) -> ParsedQuery:
        """
        解析查询
        
        Args:
            query: 用户查询
            
        Returns:
            解析后的查询对象
        """
        # 规范化查询
        normalized = self._normalize_query(query)
        
        # 识别查询类型
        query_type = self._detect_query_type(normalized)
        
        # 提取实体
        entities = self._extract_entities(normalized)
        
        # 提取关键词
        keywords = self._extract_keywords(normalized, entities)
        
        # 生成过滤条件
        filters = self._generate_filters(entities)
        
        # 查询扩展
        expansion_terms = []
        if self.enable_expansion:
            expansion_terms = self._expand_query(normalized, entities)
        
        return ParsedQuery(
            original_query=query,
            normalized_query=normalized,
            query_type=query_type,
            keywords=keywords[:self.max_keywords],
            entities=entities,
            filters=filters,
            expansion_terms=expansion_terms,
        )
    
    def _normalize_query(self, query: str) -> str:
        """规范化查询"""
        # 去除多余空格
        normalized = re.sub(r'\s+', ' ', query.strip())
        
        # 统一中文标点
        replacements = {
            '，': ',',
            '。': '.',
            '？': '?',
            '！': '!',
            '：': ':',
            '；': ';',
        }
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def _detect_query_type(self, query: str) -> QueryType:
        """检测查询类型"""
        # 检查是否是问题型查询
        if any(word in query for word in self.QUESTION_WORDS) or query.endswith('?'):
            return QueryType.QUESTION
        
        # 检查是否包含书名号（通常是法规名称查询）
        if '《' in query and '》' in query:
            return QueryType.ENTITY
        
        # 检查是否是简短的关键词查询
        if len(query) <= 10 and ' ' not in query:
            return QueryType.KEYWORD
        
        return QueryType.HYBRID
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """提取实体"""
        entities = {
            "law_names": [],
            "regions": [],
            "articles": [],
            "law_types": [],
        }
        
        # 提取法规名称
        for pattern in self._law_patterns:
            matches = pattern.findall(query)
            entities["law_names"].extend(matches)
        
        # 提取地区
        for pattern in self._region_patterns:
            matches = pattern.findall(query)
            entities["regions"].extend(matches)
        
        # 提取条款号
        for pattern in self._article_patterns:
            matches = pattern.findall(query)
            entities["articles"].extend(matches)
        
        # 提取法律类型
        for law_type, keywords in self.LAW_TYPE_KEYWORDS.items():
            if any(kw in query for kw in keywords):
                entities["law_types"].append(law_type)
        
        # 去重
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _extract_keywords(
        self, 
        query: str, 
        entities: Dict[str, List[str]]
    ) -> List[str]:
        """提取关键词"""
        # 移除实体后的文本
        text = query
        for entity_list in entities.values():
            for entity in entity_list:
                text = text.replace(entity, ' ')
        
        # 移除书名号和标点
        text = re.sub(r'[《》、，。？！：；\s]+', ' ', text)
        
        # 分词（简单按空格分割）
        words = text.split()
        
        # 过滤太短或太长的词
        keywords = [w for w in words if 2 <= len(w) <= 20]
        
        # 加入实体作为关键词
        for entity_list in entities.values():
            keywords.extend(entity_list)
        
        return list(set(keywords))
    
    def _generate_filters(self, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """生成过滤条件"""
        filters = {}
        
        if entities.get("regions"):
            if len(entities["regions"]) == 1:
                filters["region"] = entities["regions"][0]
            else:
                filters["region"] = entities["regions"]
        
        if entities.get("law_types"):
            if len(entities["law_types"]) == 1:
                filters["law_type"] = entities["law_types"][0]
            else:
                filters["law_type"] = entities["law_types"]
        
        if entities.get("law_names"):
            if len(entities["law_names"]) == 1:
                filters["law_name"] = entities["law_names"][0]
            else:
                filters["law_name"] = entities["law_names"]
        
        return filters
    
    def _expand_query(
        self, 
        query: str, 
        entities: Dict[str, List[str]]
    ) -> List[str]:
        """查询扩展"""
        expansions = []
        
        # 法律术语扩展映射
        term_expansions = {
            "消防": ["火灾", "灭火", "防火", "消防安全"],
            "合同": ["契约", "协议", "约定"],
            "侵权": ["损害赔偿", "责任", "过错"],
            "继承": ["遗产", "遗嘱", "法定继承"],
            "婚姻": ["结婚", "离婚", "夫妻"],
            "劳动": ["用工", "劳动合同", "工资", "社保"],
            "知识产权": ["专利", "商标", "著作权", "版权"],
            "刑事": ["犯罪", "刑罚", "量刑"],
            "民事": ["诉讼", "起诉", "判决"],
        }
        
        # 根据查询中的词添加扩展
        for term, related in term_expansions.items():
            if term in query:
                expansions.extend(related[:2])  # 最多添加2个扩展词
        
        return list(set(expansions))[:5]  # 最多5个扩展词
