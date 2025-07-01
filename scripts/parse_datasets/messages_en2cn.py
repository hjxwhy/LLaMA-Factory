import json
import asyncio
import logging
import re
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import time
from pathlib import Path
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MessageTranslator:
    def __init__(self, api_key: str = "none", base_url: str = "http://0.0.0.0:8888/v1", 
                 model: str = "Qwen3-30B-A3B", max_retries: int = 3, concurrent_limit: int = 5,
                 conversation_batch_size: int = 10):
        """
        初始化翻译器
        
        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
            model: 使用的模型名称
            max_retries: 最大重试次数
            concurrent_limit: 消息级并发请求限制
            conversation_batch_size: 对话级批处理大小
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.max_retries = max_retries
        self.concurrent_limit = concurrent_limit
        self.conversation_batch_size = conversation_batch_size
        self.semaphore = asyncio.Semaphore(concurrent_limit)
        
        # 翻译提示词
        self.translation_prompt = """请将以下英文文本翻译成中文。要求：
1. 保持原文的语气和风格
2. 翻译要准确、自然、流畅
3. 只返回翻译结果，不要添加任何解释

英文文本：{text}

中文翻译："""

    def _extract_multimedia_tags(self, text: str) -> Tuple[str, List[Tuple[str, int]]]:
        """
        提取多媒体标签并返回清理后的文本和标签位置信息
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本和标签位置信息列表 [(tag, position), ...]
        """
        # 定义多媒体标签的正则表达式
        multimedia_pattern = r'<(image|audio|video)(?:\s[^>]*)?>(?:</\1>)?|<(?:image|audio|video)(?:\s[^>]*)?/?>'
        
        # 找到所有多媒体标签及其位置
        tags_info = []
        cleaned_text = text
        
        # 使用正则表达式查找所有匹配项
        matches = list(re.finditer(multimedia_pattern, text, re.IGNORECASE))
        
        # 从后往前处理，避免位置偏移
        for match in reversed(matches):
            tag = match.group(0)
            start_pos = match.start()
            end_pos = match.end()
            
            # 记录标签和其在清理后文本中的位置
            # 计算在已清理文本中的相对位置
            relative_pos = start_pos
            for stored_tag, stored_pos in tags_info:
                if stored_pos < start_pos:
                    relative_pos -= len(stored_tag)
            
            tags_info.append((tag, relative_pos))
            
            # 从文本中移除标签，同时处理可能的多余空格
            before_tag = cleaned_text[:start_pos]
            after_tag = cleaned_text[end_pos:]
            
            # 如果标签前后都有空格，移除一个空格以避免双空格
            if (before_tag.endswith(' ') and after_tag.startswith(' ')) or \
               (start_pos == 0 and after_tag.startswith(' ')):
                after_tag = after_tag.lstrip(' ')
            elif before_tag.endswith(' ') and start_pos == len(before_tag):
                # 标签在末尾的情况
                before_tag = before_tag.rstrip(' ')
            
            cleaned_text = before_tag + after_tag
        
        # 反转列表以保持原始顺序
        tags_info.reverse()
        
        # 清理多余的空格并标准化
        cleaned_text = ' '.join(cleaned_text.split())
        
        return cleaned_text, tags_info

    def _restore_multimedia_tags(self, translated_text: str, tags_info: List[Tuple[str, int]]) -> str:
        """
        将多媒体标签恢复到翻译后的文本中
        
        Args:
            translated_text: 翻译后的文本
            tags_info: 标签位置信息列表
            
        Returns:
            恢复标签后的文本
        """
        if not tags_info:
            return translated_text
        
        result = translated_text.strip()
        
        # 如果翻译后的文本为空，将所有标签放在开头
        if not result:
            tags = [tag for tag, _ in tags_info]
            return ' '.join(tags)
        
        # 按位置排序标签信息
        sorted_tags = sorted(tags_info, key=lambda x: x[1])
        
        # 智能恢复标签位置
        for i, (tag, original_pos) in enumerate(sorted_tags):
            if original_pos == 0:
                # 标签在开头
                result = tag + " " + result
            elif i == len(sorted_tags) - 1 and original_pos >= len(translated_text.split()):
                # 最后一个标签，放在末尾
                result = result + " " + tag
            else:
                # 标签在中间，尝试找到合适的插入位置
                words = result.split()
                if len(words) > 1:
                    # 根据原始位置比例来确定插入位置
                    insert_index = min(len(words) // 2, len(words) - 1)
                    words.insert(insert_index, tag)
                    result = ' '.join(words)
                else:
                    # 只有一个词，标签放在前面
                    result = tag + " " + result
        
        return result.strip()

    async def translate_single_message(self, content: str, role: str, message_index: int) -> Dict[str, Any]:
        """
        翻译单条消息
        
        Args:
            content: 消息内容
            role: 消息角色 (user/assistant)
            message_index: 消息索引，用于日志
            
        Returns:
            翻译后的消息字典
        """
        async with self.semaphore:  # 限制并发数量
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"正在翻译第 {message_index + 1} 条消息 ({role}), 尝试 {attempt + 1}/{self.max_retries}")
                    
                    # 如果内容已经是中文或包含中文，跳过翻译
                    if self._contains_chinese(content):
                        logger.info(f"第 {message_index + 1} 条消息已包含中文，跳过翻译")
                        return {"content": content, "role": role}
                    
                    # 提取多媒体标签
                    cleaned_content, tags_info = self._extract_multimedia_tags(content)
                    
                    # 如果清理后的内容为空或只有空白字符，直接返回原内容
                    if not cleaned_content.strip():
                        logger.info(f"第 {message_index + 1} 条消息清理后无文本内容，保持原样")
                        return {"content": content, "role": role}
                    
                    logger.info(f"第 {message_index + 1} 条消息提取了 {len(tags_info)} 个多媒体标签")
                    
                    # 调用OpenAI API进行翻译
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": self.translation_prompt.format(text=cleaned_content)
                            }
                        ],
                        temperature=0.3,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                    )
                    
                    translated_content = response.choices[0].message.content.strip()
                    
                    # 恢复多媒体标签
                    final_content = self._restore_multimedia_tags(translated_content, tags_info)
                    
                    logger.info(f"第 {message_index + 1} 条消息翻译完成，已恢复多媒体标签")
                    
                    return {
                        "content": final_content,
                        "role": role
                    }
                    
                except Exception as e:
                    logger.error(f"翻译第 {message_index + 1} 条消息时出错 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                    if attempt == self.max_retries - 1:
                        logger.error(f"第 {message_index + 1} 条消息翻译失败，使用原文")
                        return {"content": content, "role": role}
                    
                    # 指数退避策略
                    await asyncio.sleep(2 ** attempt)
            
            return {"content": content, "role": role}

    def _contains_chinese(self, text: str) -> bool:
        """检查文本是否包含中文字符"""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    async def translate_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        并发翻译所有消息
        
        Args:
            messages: 消息列表
            
        Returns:
            翻译后的消息列表
        """
        logger.info(f"开始翻译 {len(messages)} 条消息")
        
        # 创建翻译任务
        tasks = []
        for i, message in enumerate(messages):
            if "content" in message and "role" in message:
                task = self.translate_single_message(
                    message["content"], 
                    message["role"], 
                    i
                )
                tasks.append(task)
            else:
                logger.warning(f"第 {i + 1} 条消息格式不正确，跳过翻译")
                tasks.append(asyncio.coroutine(lambda m=message: m)())

        # 并发执行翻译任务
        translated_messages = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        result = []
        for i, translated in enumerate(translated_messages):
            if isinstance(translated, Exception):
                logger.error(f"第 {i + 1} 条消息翻译失败: {translated}")
                result.append(messages[i])  # 使用原消息
            else:
                result.append(translated)
        
        logger.info("所有消息翻译完成")
        return result

    def load_json_file(self, file_path: str) -> Dict[str, Any]:
        """
        加载JSON文件
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            JSON数据
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载文件: {file_path}")
            return data
        except Exception as e:
            logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            raise

    def save_json_file(self, data: Dict[str, Any], output_path: str) -> None:
        """
        保存JSON文件
        
        Args:
            data: 要保存的数据
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"翻译结果已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存文件 {output_path} 失败: {str(e)}")
            raise

    async def translate_conversation(self, conversation_item: Dict[str, Any], conversation_index: int) -> Dict[str, Any]:
        """
        翻译单个对话
        
        Args:
            conversation_item: 对话项字典
            conversation_index: 对话索引
            
        Returns:
            翻译后的对话项
        """
        if not isinstance(conversation_item, dict) or "messages" not in conversation_item:
            logger.warning(f"第 {conversation_index + 1} 个对话项格式不正确，跳过翻译")
            return conversation_item
        
        logger.info(f"正在翻译第 {conversation_index + 1} 个对话")
        
        # 翻译该对话的messages
        original_messages = conversation_item["messages"]
        translated_messages = await self.translate_messages(original_messages)
        
        # 创建翻译后的对话项
        translated_item = conversation_item.copy()
        translated_item["messages"] = translated_messages
        
        logger.info(f"第 {conversation_index + 1} 个对话翻译完成")
        return translated_item

    async def translate_conversations_batch(self, conversations: List[Dict[str, Any]], 
                                          start_index: int = 0) -> List[Dict[str, Any]]:
        """
        批量翻译多个对话，控制并发数量
        
        Args:
            conversations: 对话列表
            start_index: 起始索引（用于日志显示）
            
        Returns:
            翻译后的对话列表
        """
        logger.info(f"开始批量翻译 {len(conversations)} 个对话 (索引 {start_index} - {start_index + len(conversations) - 1})")
        
        # 创建翻译任务
        translation_tasks = []
        for i, conversation in enumerate(conversations):
            task = self.translate_conversation(conversation, start_index + i)
            translation_tasks.append(task)
        
        # 并发执行翻译任务
        translated_results = await asyncio.gather(*translation_tasks, return_exceptions=True)
        
        # 处理翻译结果和异常
        result = []
        for i, translated in enumerate(translated_results):
            if isinstance(translated, Exception):
                logger.error(f"第 {start_index + i + 1} 个对话翻译失败: {translated}")
                result.append(conversations[i])  # 使用原对话
            else:
                result.append(translated)
        
        logger.info(f"批量翻译完成，成功翻译 {len([r for r in translated_results if not isinstance(r, Exception)])} 个对话")
        return result

    async def translate_file(self, input_path: str, output_path: str = None) -> None:
        """
        翻译整个JSON文件
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径（可选，默认为input_path的_cn版本）
        """
        # 生成输出文件名
        if output_path is None:
            input_file = Path(input_path)
            output_path = str(input_file.parent / f"{input_file.stem}_cn{input_file.suffix}")
        
        # 加载原始数据
        data = self.load_json_file(input_path)
        
        # 检查数据格式
        if isinstance(data, list):
            # 新格式：list[dict(messages=...)]
            total_conversations = len(data)
            logger.info(f"检测到列表格式数据，包含 {total_conversations} 个对话")
            
            # 如果数据量较大，使用批处理
            if total_conversations > self.conversation_batch_size:
                logger.info(f"数据量较大，将使用批处理模式，每批处理 {self.conversation_batch_size} 个对话")
                
                translated_data = []
                processed_count = 0
                
                # 分批处理对话
                for i in range(0, total_conversations, self.conversation_batch_size):
                    batch_end = min(i + self.conversation_batch_size, total_conversations)
                    batch_conversations = data[i:batch_end]
                    
                    logger.info(f"正在处理第 {i//self.conversation_batch_size + 1} 批，对话 {i+1}-{batch_end}")
                    
                    # 翻译当前批次
                    batch_results = await self.translate_conversations_batch(batch_conversations, i)
                    translated_data.extend(batch_results)
                    
                    processed_count += len(batch_results)
                    logger.info(f"已完成 {processed_count}/{total_conversations} 个对话的翻译 ({processed_count/total_conversations*100:.1f}%)")
                    
                    # 可选：在批次之间添加短暂延迟，避免API限流
                    if i + self.conversation_batch_size < total_conversations:
                        await asyncio.sleep(0.1)
            
            else:
                # 数据量较小，直接并发处理
                logger.info(f"数据量较小，直接并发翻译所有对话")
                translated_data = await self.translate_conversations_batch(data)
            
        elif isinstance(data, dict) and "messages" in data:
            # 原格式：dict(messages=...)
            logger.info("检测到单个对话格式数据")
            
            # 翻译messages
            original_messages = data["messages"]
            translated_messages = await self.translate_messages(original_messages)
            
            # 更新数据
            translated_data = data.copy()
            translated_data["messages"] = translated_messages
            
        else:
            logger.error("JSON文件格式不正确，应该是包含 'messages' 字段的字典或字典列表")
            raise ValueError("JSON文件格式不正确，应该是包含 'messages' 字段的字典或字典列表")
        
        # 保存结果
        self.save_json_file(translated_data, output_path)
        
        logger.info(f"翻译完成! 原文件: {input_path}, 译文文件: {output_path}")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="将JSON文件中的英文messages翻译成中文")
    parser.add_argument("input_file", help="输入的JSON文件路径")
    parser.add_argument("-o", "--output", help="输出的JSON文件路径（可选）")
    parser.add_argument("--api-key", default="none", help="OpenAI API密钥")
    parser.add_argument("--base-url", default="http://10.0.8.131:8888/v1", help="API基础URL")
    parser.add_argument("--model", default="Qwen3-30B-A3B", help="使用的模型名称")
    parser.add_argument("--max-retries", type=int, default=3, help="最大重试次数")
    parser.add_argument("--concurrent-limit", type=int, default=5, help="消息级并发请求限制")
    parser.add_argument("--batch-size", type=int, default=10, help="对话级批处理大小（用于控制内存使用）")
    
    args = parser.parse_args()
    
    # 创建翻译器
    translator = MessageTranslator(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_retries=args.max_retries,
        concurrent_limit=args.concurrent_limit,
        conversation_batch_size=args.batch_size
    )
    
    try:
        # 执行翻译
        await translator.translate_file(args.input_file, args.output)
    except Exception as e:
        logger.error(f"翻译过程中发生错误: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    # 运行示例（如果直接执行脚本）
    import sys
    
    # 可以在这里设置默认的输入文件路径进行测试
    # 示例用法：python messages_en2cn.py input.json -o output.json
    
    if len(sys.argv) == 1:
        # 如果没有提供命令行参数，可以在这里设置默认路径进行测试
        print("使用方法:")
        print("python messages_en2cn.py input.json [-o output.json] [其他选项]")
        print("\n可选参数:")
        print("  -o, --output          输出文件路径")
        print("  --api-key            OpenAI API密钥 (默认: none)")
        print("  --base-url           API基础URL (默认: http://0.0.0.0:8888/v1)")
        print("  --model              模型名称 (默认: Qwen3-30B-A3B)")
        print("  --max-retries        最大重试次数 (默认: 3)")
        print("  --concurrent-limit   并发请求限制 (默认: 5)")
        print("  --batch-size         对话级批处理大小（用于控制内存使用）")
        sys.exit(0)
    
    # 运行异步主函数
    exit_code = asyncio.run(main())
    sys.exit(exit_code)