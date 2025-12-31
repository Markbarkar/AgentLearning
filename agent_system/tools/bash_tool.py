"""
安全受限的 Bash 工具

提供给 Agent 使用的受限 Bash 命令执行能力
包含多层安全检查：目录白名单、命令白名单、注入检测、危险模式过滤
"""

import json
import shlex
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from .base import register_tool, auto_register

# 配置日志
logger = logging.getLogger("bash_tool")


class BashSecurityPolicy:
    """
    Bash 安全策略类
    
    负责验证和执行受限的 Bash 命令
    """
    
    _instance = None
    _config: Dict[str, Any] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """加载安全策略配置"""
        config_path = Path(__file__).parent.parent / "config" / "bash_policy.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Bash 安全策略配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = json.load(f)
        
        # 设置日志
        if self._config.get("logging", {}).get("enabled"):
            log_file = Path(__file__).parent.parent.parent / self._config["logging"]["log_file"]
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            handler = logging.FileHandler(log_file, encoding='utf-8')
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s'
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    @property
    def config(self) -> Dict[str, Any]:
        return self._config
    
    @property
    def enabled(self) -> bool:
        return self._config.get("enabled", False)
    
    def _is_allowed_directory(self, path: str) -> bool:
        """检查目录是否在白名单中"""
        if not path:
            return False
        
        path = Path(path).resolve()
        allowed_dirs = self._config.get("allowed_directories", [])
        
        for allowed_dir in allowed_dirs:
            allowed_path = Path(allowed_dir).resolve()
            try:
                path.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        
        return False
    
    def _is_command_allowed(self, cmd_name: str) -> bool:
        """检查命令是否在白名单中"""
        whitelist = self._config.get("command_whitelist", {})
        return cmd_name in whitelist
    
    def _check_dangerous_patterns(self, command: str) -> Tuple[bool, str]:
        """检查命令是否包含危险模式"""
        blocked = self._config.get("blocked_patterns", [])
        
        for pattern in blocked:
            if pattern.lower() in command.lower():
                return False, f"检测到危险模式: '{pattern}'"
        
        return True, ""
    
    def _check_injection(self, command: str) -> Tuple[bool, str]:
        """检查命令注入"""
        dangerous_chars = self._config.get("dangerous_chars", [])
        
        for char in dangerous_chars:
            if char in command:
                return False, f"不允许使用字符: '{char}'"
        
        return True, ""
    
    def _decode_output(self, data: bytes) -> str:
        """
        解码命令输出，自动尝试多种编码
        
        Args:
            data: 原始字节数据
            
        Returns:
            解码后的字符串
        """
        if not data:
            return ""
        
        # 尝试的编码列表（按优先级）
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']
        
        for encoding in encodings:
            try:
                return data.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        
        # 所有编码都失败，使用 utf-8 并忽略错误
        return data.decode('utf-8', errors='replace')
    
    def _check_path_traversal(self, command: str, working_dir: str) -> Tuple[bool, str]:
        """检查路径遍历攻击"""
        # 检查 .. 是否会导致跳出允许的目录
        try:
            parts = shlex.split(command)
        except ValueError:
            return True, ""  # 解析失败时让后续检查处理
        
        for part in parts:
            if part.startswith('/') or part.startswith('~'):
                # 绝对路径，检查是否在允许目录内
                resolved = Path(part).expanduser().resolve()
                if not self._is_allowed_directory(str(resolved)):
                    return False, f"路径 '{part}' 不在允许的目录范围内"
            elif '..' in part:
                # 相对路径包含 ..，计算实际路径
                resolved = (Path(working_dir) / part).resolve()
                if not self._is_allowed_directory(str(resolved)):
                    return False, f"路径 '{part}' 会跳出允许的目录范围"
        
        return True, ""
    
    def validate_command(self, command: str, working_dir: str) -> Tuple[bool, str]:
        """
        验证命令是否安全
        
        Args:
            command: 要执行的命令
            working_dir: 工作目录
            
        Returns:
            (是否通过验证, 错误消息)
        """
        if not self.enabled:
            return False, "Bash 工具已禁用"
        
        # 1. 检查工作目录
        if not self._is_allowed_directory(working_dir):
            allowed = self._config.get("allowed_directories", [])
            return False, f"目录 '{working_dir}' 不在允许范围内。允许的目录: {allowed}"
        
        # 2. 解析命令
        try:
            parts = shlex.split(command)
        except ValueError as e:
            return False, f"命令解析失败: {e}"
        
        if not parts:
            return False, "空命令"
        
        cmd_name = parts[0]
        
        # 3. 检查命令白名单
        if not self._is_command_allowed(cmd_name):
            allowed_cmds = list(self._config.get("command_whitelist", {}).keys())
            return False, f"命令 '{cmd_name}' 不在白名单中。允许的命令: {allowed_cmds}"
        
        # 4. 检查危险模式
        is_safe, error = self._check_dangerous_patterns(command)
        if not is_safe:
            return False, error
        
        # 5. 检查命令注入
        is_safe, error = self._check_injection(command)
        if not is_safe:
            return False, error
        
        # 6. 检查路径遍历
        is_safe, error = self._check_path_traversal(command, working_dir)
        if not is_safe:
            return False, error
        
        return True, ""
    
    def execute(self, command: str, working_dir: str) -> str:
        """
        安全执行命令
        
        Args:
            command: 要执行的命令
            working_dir: 工作目录
            
        Returns:
            命令输出或错误信息
        """
        # 记录命令
        logger.info(f"尝试执行: {command} (目录: {working_dir})")
        
        # 验证命令
        is_valid, error = self.validate_command(command, working_dir)
        if not is_valid:
            logger.warning(f"命令被拒绝: {error}")
            return f"❌ 安全检查失败: {error}"
        
        # 确保工作目录存在
        working_path = Path(working_dir)
        if not working_path.exists():
            return f"❌ 工作目录不存在: {working_dir}"
        
        try:
            # 执行命令（获取原始字节，避免编码问题）
            result = subprocess.run(
                shlex.split(command),
                cwd=working_dir,
                capture_output=True,
                text=False,  # 返回 bytes，手动处理编码
                timeout=self._config.get("timeout_seconds", 30)
            )
            
            # 解码输出（尝试多种编码）
            output = self._decode_output(result.stdout)
            stderr = self._decode_output(result.stderr)
            
            if result.returncode != 0 and stderr:
                output = f"{output}\n[stderr]: {stderr}" if output else stderr
            
            # 限制输出大小
            max_bytes = self._config.get("max_output_bytes", 102400)
            if len(output) > max_bytes:
                output = output[:max_bytes] + "\n\n...(输出已截断，超过 100KB 限制)"
            
            logger.info(f"命令执行成功: {command}")
            return output if output else "(命令执行成功，无输出)"
            
        except subprocess.TimeoutExpired:
            timeout = self._config.get("timeout_seconds", 30)
            logger.error(f"命令超时: {command}")
            return f"❌ 命令执行超时（超过 {timeout} 秒）"
        except FileNotFoundError:
            return f"❌ 命令不存在或无法执行"
        except Exception as e:
            logger.error(f"命令执行失败: {e}")
            return f"❌ 执行失败: {str(e)}"


# 全局策略实例
_policy: Optional[BashSecurityPolicy] = None


def get_bash_policy() -> BashSecurityPolicy:
    """获取 Bash 安全策略实例"""
    global _policy
    if _policy is None:
        _policy = BashSecurityPolicy()
    return _policy


def _parse_bash_input(input_str: str) -> Dict[str, Any]:
    """
    解析 Bash 工具输入
    
    格式: command 或 command,working_dir
    """
    parts = input_str.rsplit(',', 1)
    
    if len(parts) == 2:
        command = parts[0].strip()
        working_dir = parts[1].strip()
    else:
        command = input_str.strip()
        # 默认工作目录
        working_dir = "/projects/AgentLearning/workspace"
    
    return {
        "command": command,
        "working_dir": working_dir
    }


@auto_register
class BashToolExecutor:
    """Bash 工具执行器"""
    
    @register_tool(
        name="execute_bash",
        description="""在安全受限环境中执行 Bash 命令。

【安全限制】
- 只能在指定目录下执行（/projects/AgentLearning/workspace, /projects/AgentLearning/data）
- 只支持白名单命令：ls, cat, head, tail, grep, find, wc, pwd, echo, du, df, file, stat, tree
- 禁止使用管道(|)、重定向(>)、命令组合(&&, ;)等

【输入格式】
command,working_dir
- command: 要执行的命令
- working_dir: 工作目录（可选，默认 /projects/AgentLearning/workspace）

【使用示例】
- 列出目录：ls -la,/projects/AgentLearning/workspace
- 查看文件：cat file.txt,/projects/AgentLearning/data
- 搜索文本：grep -rn 'def',/projects/AgentLearning/workspace
- 查找文件：find . -name '*.py',/projects/AgentLearning/workspace

【注意事项】
- 命令执行有 30 秒超时限制
- 输出最大 100KB，超出部分会被截断""",
        input_parser=_parse_bash_input
    )
    def execute_bash(self, command: str, working_dir: str = "/projects/AgentLearning/workspace") -> str:
        """
        执行受限的 Bash 命令
        
        Args:
            command: 要执行的命令
            working_dir: 工作目录
            
        Returns:
            命令输出或错误信息
        """
        policy = get_bash_policy()
        return policy.execute(command, working_dir)


# 创建实例（@auto_register 装饰器会自动注册到 ToolRegistry）
_bash_executor = BashToolExecutor()


def get_allowed_commands() -> Dict[str, Any]:
    """获取允许的命令列表"""
    policy = get_bash_policy()
    return policy.config.get("command_whitelist", {})


def get_allowed_directories() -> list:
    """获取允许的目录列表"""
    policy = get_bash_policy()
    return policy.config.get("allowed_directories", [])

