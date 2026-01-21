"""
法律文档智能分块器

根据法律文档的结构（章、节、条、款、项）进行智能分块
保持条款的语义完整性
"""

import re
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document

from .base import ChunkerBase


class LegalChunker(ChunkerBase):
    """
    法律文档智能分块器
    
    特点：
    1. 识别法律文档结构（章、节、条、款、项）
    2. 按条款边界分块，保持语义完整性
    3. 处理过长条款的二次分割
    4. 为每个 chunk 添加父级上下文信息
    """
    
    # 中文数字映射
    CN_NUMBERS = "零一二三四五六七八九十百千"
    
    # 法律文档结构正则模式
    PATTERNS = {
        # 标题（法规名称）
        "title": re.compile(r"^(.+(?:条例|规定|办法|决定|细则|解释|法))_?$", re.MULTILINE),
        
        # 章：第一章 总则
        "chapter": re.compile(r"^(第[一二三四五六七八九十百]+章)\s*(.+)$", re.MULTILINE),
        
        # 节：第一节 一般规定
        "section": re.compile(r"^(第[一二三四五六七八九十百]+节)\s*(.+)$", re.MULTILINE),
        
        # 条：第一条、第10条
        "article": re.compile(r"^(第[一二三四五六七八九十百\d]+条)\s*", re.MULTILINE),
        
        # 款：（一）、（1）、(一)
        "paragraph": re.compile(r"^[（\(]([一二三四五六七八九十\d]+)[）\)]", re.MULTILINE),
        
        # 项：1.、1、
        "item": re.compile(r"^([1-9]\d*)[\.\、]", re.MULTILINE),
    }
    
    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 100,
        overlap: int = 50,
        include_context: bool = True
    ):
        """
        初始化法律文档分块器
        
        Args:
            max_chunk_size: 最大分块大小（超过则二次分割）
            min_chunk_size: 最小分块大小（过小则合并）
            overlap: 分块重叠大小（用于二次分割）
            include_context: 是否在元数据中包含父级上下文
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        self.include_context = include_context
    
    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        对法律文档进行智能分块
        
        Args:
            text: 法律文档文本
            metadata: 文档元数据
            
        Returns:
            分块后的 Document 列表
        """
        if metadata is None:
            metadata = {}
        
        # 1. 解析文档结构
        structure = self._parse_structure(text)
        
        # 2. 按条款分块
        raw_chunks = self._split_by_articles(text, structure)
        
        # 3. 处理过长/过短的块
        processed_chunks = self._process_chunk_sizes(raw_chunks)
        
        # 4. 创建 Document 对象
        documents = []
        for i, chunk_info in enumerate(processed_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "total_chunks": len(processed_chunks),
                "chunker": "legal",
                "chapter": chunk_info.get("chapter", ""),
                "section": chunk_info.get("section", ""),
                "article_num": chunk_info.get("article_num", ""),
            })
            
            if self.include_context and chunk_info.get("context"):
                chunk_metadata["parent_context"] = chunk_info["context"]
            
            documents.append(Document(
                page_content=chunk_info["content"],
                metadata=chunk_metadata
            ))
        
        return documents
    
    def _parse_structure(self, text: str) -> Dict[str, Any]:
        """
        解析文档结构，提取章、节、条的位置信息
        
        Args:
            text: 文档文本
            
        Returns:
            结构信息字典
        """
        structure = {
            "chapters": [],
            "sections": [],
            "articles": [],
        }
        
        # 提取章
        for match in self.PATTERNS["chapter"].finditer(text):
            structure["chapters"].append({
                "num": match.group(1),
                "title": match.group(2).strip(),
                "start": match.start(),
                "end": match.end(),
                "full": f"{match.group(1)} {match.group(2).strip()}"
            })
        
        # 提取节
        for match in self.PATTERNS["section"].finditer(text):
            structure["sections"].append({
                "num": match.group(1),
                "title": match.group(2).strip(),
                "start": match.start(),
                "end": match.end(),
                "full": f"{match.group(1)} {match.group(2).strip()}"
            })
        
        # 提取条
        for match in self.PATTERNS["article"].finditer(text):
            structure["articles"].append({
                "num": match.group(1),
                "start": match.start(),
                "end": match.end(),
            })
        
        return structure
    
    def _split_by_articles(
        self,
        text: str,
        structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        按条款分割文档
        
        Args:
            text: 文档文本
            structure: 结构信息
            
        Returns:
            分块信息列表
        """
        articles = structure["articles"]
        chapters = structure["chapters"]
        sections = structure["sections"]
        
        if not articles:
            # 没有条款结构，返回整个文档作为一个块
            return [{
                "content": text.strip(),
                "chapter": "",
                "section": "",
                "article_num": "",
                "context": ""
            }]
        
        chunks = []
        
        # 处理第一条之前的内容（总则等）
        if articles[0]["start"] > 0:
            preamble = text[:articles[0]["start"]].strip()
            if preamble and len(preamble) >= self.min_chunk_size:
                chunks.append({
                    "content": preamble,
                    "chapter": self._find_chapter(0, chapters),
                    "section": self._find_section(0, sections),
                    "article_num": "",
                    "context": "序言/总则"
                })
        
        # 按条款分割
        for i, article in enumerate(articles):
            # 确定条款内容的结束位置
            if i + 1 < len(articles):
                end_pos = articles[i + 1]["start"]
            else:
                end_pos = len(text)
            
            content = text[article["start"]:end_pos].strip()
            
            if content:
                chunks.append({
                    "content": content,
                    "chapter": self._find_chapter(article["start"], chapters),
                    "section": self._find_section(article["start"], sections),
                    "article_num": article["num"],
                    "context": self._build_context(article["start"], chapters, sections)
                })
        
        return chunks
    
    def _find_chapter(self, pos: int, chapters: List[Dict]) -> str:
        """查找位置所属的章"""
        current_chapter = ""
        for chapter in chapters:
            if chapter["start"] <= pos:
                current_chapter = chapter["full"]
            else:
                break
        return current_chapter
    
    def _find_section(self, pos: int, sections: List[Dict]) -> str:
        """查找位置所属的节"""
        current_section = ""
        for section in sections:
            if section["start"] <= pos:
                current_section = section["full"]
            else:
                break
        return current_section
    
    def _build_context(
        self,
        pos: int,
        chapters: List[Dict],
        sections: List[Dict]
    ) -> str:
        """构建父级上下文"""
        chapter = self._find_chapter(pos, chapters)
        section = self._find_section(pos, sections)
        
        parts = []
        if chapter:
            parts.append(chapter)
        if section:
            parts.append(section)
        
        return " > ".join(parts) if parts else ""
    
    def _process_chunk_sizes(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        处理分块大小，分割过长的块，合并过短的块
        
        Args:
            chunks: 原始分块列表
            
        Returns:
            处理后的分块列表
        """
        processed = []
        buffer = None  # 用于合并过短的块
        
        for chunk in chunks:
            content = chunk["content"]
            
            if len(content) > self.max_chunk_size:
                # 先处理 buffer
                if buffer:
                    processed.append(buffer)
                    buffer = None
                
                # 分割过长的块
                sub_chunks = self._split_long_chunk(chunk)
                processed.extend(sub_chunks)
                
            elif len(content) < self.min_chunk_size:
                # 尝试与 buffer 合并
                if buffer:
                    buffer["content"] += "\n\n" + content
                    # 更新 article_num 为范围
                    if chunk["article_num"] and buffer["article_num"]:
                        if buffer["article_num"] != chunk["article_num"]:
                            buffer["article_num"] = f"{buffer['article_num']}-{chunk['article_num']}"
                else:
                    buffer = chunk.copy()
                
                # 如果合并后够大了，输出
                if buffer and len(buffer["content"]) >= self.min_chunk_size:
                    processed.append(buffer)
                    buffer = None
            else:
                # 正常大小
                if buffer:
                    # 合并 buffer
                    buffer["content"] += "\n\n" + content
                    if chunk["article_num"] and buffer["article_num"]:
                        if buffer["article_num"] != chunk["article_num"]:
                            buffer["article_num"] = f"{buffer['article_num']}-{chunk['article_num']}"
                    processed.append(buffer)
                    buffer = None
                else:
                    processed.append(chunk)
        
        # 处理剩余的 buffer
        if buffer:
            processed.append(buffer)
        
        return processed
    
    def _split_long_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        分割过长的块
        
        按款（paragraph）分割，如果还是太长则按句子分割
        """
        content = chunk["content"]
        
        # 尝试按款分割
        parts = self.PATTERNS["paragraph"].split(content)
        
        if len(parts) > 1:
            # 有款结构
            sub_chunks = []
            current = ""
            
            for part in parts:
                if len(current) + len(part) <= self.max_chunk_size:
                    current += part
                else:
                    if current:
                        sub_chunk = chunk.copy()
                        sub_chunk["content"] = current.strip()
                        sub_chunks.append(sub_chunk)
                    current = part
            
            if current:
                sub_chunk = chunk.copy()
                sub_chunk["content"] = current.strip()
                sub_chunks.append(sub_chunk)
            
            return sub_chunks if sub_chunks else [chunk]
        
        # 按句子分割
        sentences = re.split(r'([。！？；])', content)
        sub_chunks = []
        current = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]  # 加上标点
            
            if len(current) + len(sentence) <= self.max_chunk_size:
                current += sentence
            else:
                if current:
                    sub_chunk = chunk.copy()
                    sub_chunk["content"] = current.strip()
                    sub_chunks.append(sub_chunk)
                current = sentence
        
        if current:
            sub_chunk = chunk.copy()
            sub_chunk["content"] = current.strip()
            sub_chunks.append(sub_chunk)
        
        return sub_chunks if sub_chunks else [chunk]
