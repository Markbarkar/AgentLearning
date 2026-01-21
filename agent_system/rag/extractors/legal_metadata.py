"""
法律文档元数据提取器

从法律文档中提取结构化元数据：
- 法规名称
- 法规类型（条例/规定/办法/决定）
- 地区
- 地区级别（经济特区/自治州/自治县）
- 主题分类
- 颁布日期
- 颁布机关
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base import ExtractorBase


class LegalMetadataExtractor(ExtractorBase):
    """
    法律文档元数据提取器
    
    从法律文档的文件名和内容中提取结构化元数据
    """
    
    # 法规类型正则
    LAW_TYPE_PATTERN = re.compile(r"(条例|规定|办法|决定|细则|解释|法)_?$")
    
    # 日期提取正则（支持多种格式）
    DATE_PATTERNS = [
        # 2023年1月1日
        re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日"),
        # 2023-01-01
        re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})"),
        # 2023/01/01
        re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})"),
    ]
    
    # 颁布机关正则
    AUTHORITY_PATTERNS = [
        re.compile(r"([\u4e00-\u9fa5]+人民代表大会常务委员会)"),
        re.compile(r"([\u4e00-\u9fa5]+人民政府)"),
        re.compile(r"([\u4e00-\u9fa5]+人民代表大会)"),
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化元数据提取器
        
        Args:
            config_path: 地区配置文件路径，默认使用内置配置
        """
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            # 使用默认配置路径
            default_path = Path(__file__).parent.parent.parent / "config" / "legal_regions.json"
            config_path = str(default_path)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """返回默认配置"""
        return {
            "economic_zones": ["深圳", "海南", "厦门", "珠海", "汕头"],
            "autonomous_prefectures": [],
            "autonomous_counties": [],
            "provinces": [],
            "cities": [],
            "topic_keywords": {},
            "law_types": {
                "条例": "regulation",
                "规定": "provision",
                "办法": "measure",
                "决定": "decision",
            },
            "region_levels": {}
        }
    
    def extract(
        self,
        file_path: str,
        text: str
    ) -> Dict[str, Any]:
        """
        从法律文档提取元数据
        
        Args:
            file_path: 文件路径
            text: 文档内容
            
        Returns:
            结构化元数据字典
        """
        file_name = Path(file_path).stem  # 不含扩展名的文件名
        
        # 提取各项元数据
        law_name = self._extract_law_name(file_name, text)
        law_type = self._extract_law_type(law_name)
        region, region_level = self._extract_region(file_name, text)
        topics = self._extract_topics(law_name, text)
        publish_date = self._extract_date(text, "颁布|公布|通过")
        effective_date = self._extract_date(text, "施行|生效|实施")
        issuing_authority = self._extract_authority(text)
        
        return {
            "law_name": law_name,
            "law_type": law_type,
            "law_type_en": self.config.get("law_types", {}).get(law_type, ""),
            "region": region,
            "region_level": region_level,
            "region_level_en": self.config.get("region_levels", {}).get(region_level, ""),
            "topics": topics,
            "publish_date": publish_date,
            "effective_date": effective_date,
            "issuing_authority": issuing_authority,
            "source": file_path,
            "file_name": Path(file_path).name,
        }
    
    def _extract_law_name(self, file_name: str, text: str) -> str:
        """
        提取法规名称
        
        优先从文件名提取，其次从文档标题提取
        """
        # 清理文件名中的下划线后缀
        law_name = re.sub(r"_+$", "", file_name)
        
        # 如果文件名不包含法规类型关键词，尝试从文档内容提取
        if not self.LAW_TYPE_PATTERN.search(law_name):
            # 尝试从文档开头提取标题
            lines = text.strip().split("\n")
            for line in lines[:5]:  # 检查前5行
                line = line.strip()
                if self.LAW_TYPE_PATTERN.search(line) and len(line) < 100:
                    law_name = re.sub(r"_+$", "", line)
                    break
        
        return law_name
    
    def _extract_law_type(self, law_name: str) -> str:
        """提取法规类型"""
        match = self.LAW_TYPE_PATTERN.search(law_name)
        if match:
            return match.group(1)
        return ""
    
    def _extract_region(self, file_name: str, text: str) -> tuple:
        """
        提取地区和地区级别
        
        Returns:
            (region, region_level) 元组
        """
        combined_text = file_name + " " + text[:500]  # 只检查前500字符
        
        # 检查经济特区
        for zone in self.config.get("economic_zones", []):
            if zone in combined_text:
                return zone, "经济特区"
        
        # 检查自治州
        for prefecture in self.config.get("autonomous_prefectures", []):
            if prefecture in combined_text:
                # 提取简称
                short_name = prefecture.replace("自治州", "")
                return short_name, "自治州"
        
        # 检查自治县
        for county in self.config.get("autonomous_counties", []):
            if county in combined_text:
                short_name = county.replace("自治县", "")
                return short_name, "自治县"
        
        # 检查省份
        for province in self.config.get("provinces", []):
            if province in combined_text:
                return province, "省"
        
        # 检查城市
        for city in self.config.get("cities", []):
            if city in combined_text:
                return city, "市"
        
        # 尝试从文件名中提取（格式：地区名+法规名）
        # 例如：伊犁河流域土地开发管理条例 -> 伊犁
        region_match = re.match(r"^([\u4e00-\u9fa5]{2,6}?)(?:经济特区|市|省|县|州|地区)", file_name)
        if region_match:
            potential_region = region_match.group(1)
            if len(potential_region) >= 2:
                return potential_region, "地区"
        
        return "", ""
    
    def _extract_topics(self, law_name: str, text: str) -> List[str]:
        """提取主题分类"""
        topics = []
        combined_text = law_name + " " + text[:1000]
        
        topic_keywords = self.config.get("topic_keywords", {})
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in combined_text:
                    topics.append(topic)
                    break  # 每个主题只添加一次
        
        return topics
    
    def _extract_date(self, text: str, context_pattern: str) -> str:
        """
        提取日期
        
        Args:
            text: 文档内容
            context_pattern: 上下文正则（如 "颁布|公布"）
            
        Returns:
            ISO 格式日期字符串，如 "2023-01-01"
        """
        # 只检查文档开头和结尾（通常日期在这些位置）
        search_text = text[:1500] + text[-1500:] if len(text) > 3000 else text
        
        # 构建带上下文的搜索模式
        for date_pattern in self.DATE_PATTERNS:
            # 尝试找到上下文附近的日期（前后都搜索）
            context_matches = list(re.finditer(context_pattern, search_text))
            for ctx_match in context_matches:
                # 在上下文前后80个字符内搜索日期
                # 先搜索前面（如 "2023年6月28日...通过"）
                start_before = max(0, ctx_match.start() - 80)
                end_before = ctx_match.start()
                nearby_before = search_text[start_before:end_before]
                
                date_match = date_pattern.search(nearby_before)
                if date_match:
                    try:
                        year = int(date_match.group(1))
                        month = int(date_match.group(2))
                        day = int(date_match.group(3))
                        if 1980 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                            return f"{year:04d}-{month:02d}-{day:02d}"
                    except (ValueError, IndexError):
                        pass
                
                # 再搜索后面（如 "施行自2023年9月1日"）
                start_after = ctx_match.end()
                end_after = min(start_after + 50, len(search_text))
                nearby_after = search_text[start_after:end_after]
                
                date_match = date_pattern.search(nearby_after)
                if date_match:
                    try:
                        year = int(date_match.group(1))
                        month = int(date_match.group(2))
                        day = int(date_match.group(3))
                        if 1980 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                            return f"{year:04d}-{month:02d}-{day:02d}"
                    except (ValueError, IndexError):
                        continue
        
        # 如果上下文搜索失败，尝试直接搜索日期
        for date_pattern in self.DATE_PATTERNS:
            date_match = date_pattern.search(search_text)
            if date_match:
                try:
                    year = int(date_match.group(1))
                    month = int(date_match.group(2))
                    day = int(date_match.group(3))
                    # 验证日期合理性
                    if 1980 <= year <= datetime.now().year + 1 and 1 <= month <= 12 and 1 <= day <= 31:
                        return f"{year:04d}-{month:02d}-{day:02d}"
                except (ValueError, IndexError):
                    continue
        
        return ""
    
    def _extract_authority(self, text: str) -> str:
        """提取颁布机关"""
        # 只检查文档开头和结尾
        search_text = text[:1000] + text[-1000:] if len(text) > 2000 else text
        
        for pattern in self.AUTHORITY_PATTERNS:
            match = pattern.search(search_text)
            if match:
                authority = match.group(1)
                # 清理并验证
                if 5 <= len(authority) <= 50:
                    return authority
        
        return ""
