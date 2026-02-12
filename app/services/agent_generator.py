# app/services/agent_generator.py
import asyncio
import logging
import random
import re
import json
import io
import shutil
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import UploadFile
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain_core.tools import tool, StructuredTool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.core.config import get_settings
from app.services.llm_provider import LLMProviderFactory, QAList
from app.services.text_extractor import TextExtractor
from app.utils.exceptions import DataGenerationError

logger = logging.getLogger(__name__)

# --- Agent / Generator ---

class AgentGenerator:
    """
    Unified service for generating data using either a structured pipeline or an agentic approach.
    """

    def __init__(self, provider: str, model: Optional[str] = None, temperature: float = 0.7, api_key: Optional[str] = None):
        self.settings = get_settings()
        self.provider = provider

        # Initialize LLM Provider Factory (it handles model selection)
        self.llm_factory = LLMProviderFactory(provider, model_name=model, temperature=temperature, api_key=api_key)
        self.llm = self.llm_factory.llm
        self.parser = JsonOutputParser(pydantic_object=QAList)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
        )

        # Initialize Embeddings (for RAG)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    async def generate(
        self,
        prompt: str,
        files: List[UploadFile],
        demo_file: Optional[UploadFile],
        count: int,
        agentic: bool = False,
        mcp_servers: Optional[List[str]] = None,
        use_rag: bool = False,
        conserve_tokens: bool = False,
        rate_limit: int = 0
    ) -> List[Dict[str, str]]:
        """
        Main entry point for generation.
        """
        logger.info(f"Starting generation. Agentic: {agentic}, RAG: {use_rag}, Count: {count}, Conserve: {conserve_tokens}, Rate: {rate_limit}")

        # 1. Process Inputs
        context_text = ""
        documents = []

        if files:
            logger.info(f"Processing {len(files)} source files...")
            for file in files:
                try:
                    text = await TextExtractor.extract(file)
                    documents.append(Document(page_content=text, metadata={"source": file.filename}))
                    context_text += f"--- File: {file.filename} ---\n{text}\n"
                except Exception as e:
                    logger.warning(f"Skipping file {file.filename} due to error: {e}")

        few_shot_examples = []
        if demo_file:
            logger.info(f"Processing demo file {demo_file.filename}...")
            try:
                content = await demo_file.read()
                few_shot_examples = TextExtractor.parse_csv_to_dicts(content)

                demo_text = TextExtractor._extract_from_csv(io.BytesIO(content))
                if demo_text:
                     documents.append(Document(page_content=demo_text, metadata={"source": demo_file.filename, "type": "few-shot"}))
                     context_text += f"--- Demo File: {demo_file.filename} ---\n{demo_text}\n"

            except Exception as e:
                logger.warning(f"Failed to parse demo file: {e}")

        # Optimize Context if needed
        if conserve_tokens and context_text:
            logger.info("Conserving tokens: Truncating/Summarizing context.")
            # Simple optimization: Limit char count (approx 1 token ~= 4 chars)
            # Limit to ~2000 tokens context
            MAX_CONTEXT_CHARS = 8000
            if len(context_text) > MAX_CONTEXT_CHARS:
                context_text = context_text[:MAX_CONTEXT_CHARS] + "\n...[truncated for token conservation]..."

        # 2. Execute Generation Strategy
        if agentic:
            return await self._generate_agentic(prompt, context_text, documents, few_shot_examples, count, mcp_servers, use_rag)
        else:
            return await self._generate_pipeline(prompt, context_text, few_shot_examples, count, rate_limit)

    async def _generate_pipeline(
        self,
        prompt: str,
        context: str,
        examples: List[Dict],
        count: int,
        rate_limit: int = 0
    ) -> List[Dict[str, str]]:
        """
        Structured pipeline: Split context -> Parallel LLM calls.
        """
        base_instruction = (
            f"You are a data generation expert. {prompt}\n"
            f"Generate {count} question-answer pairs."
        )

        if examples:
            example_str = "\n".join([str(ex) for ex in examples[:5]]) # Limit examples
            base_instruction += f"\n\nUse these examples as a guide for style and format:\n{example_str}"

        all_qa_pairs = []

        if not context:
            logger.info("Generating from prompt only (no context).")
            result = await self._invoke_llm(base_instruction, "", count)
            if result:
                all_qa_pairs.extend(result)
        else:
            chunks = self.text_splitter.split_text(context)
            if not chunks:
                 chunks = [context]

            logger.info(f"Context split into {len(chunks)} chunks.")

            pairs_per_chunk = max(1, self.settings.QA_BATCH_SIZE)
            total_needed = count

            tasks = []
            generated_count = 0
            chunk_pool = list(chunks)
            random.shuffle(chunk_pool)

            while generated_count < total_needed:
                if not chunk_pool:
                    chunk_pool = list(chunks)
                    random.shuffle(chunk_pool)

                chunk = chunk_pool.pop()
                current_batch = min(pairs_per_chunk, total_needed - generated_count)
                chunk_instruction = base_instruction + f"\n\nFocus on the following content segment to generate {current_batch} pairs."

                # Apply Rate Limiting
                if rate_limit > 0:
                    delay = 60.0 / rate_limit
                    logger.info(f"Rate limiting enabled: Sleeping for {delay:.2f}s before request.")
                    await asyncio.sleep(delay)

                if rate_limit > 0:
                    result = await self._invoke_llm(chunk_instruction, chunk, current_batch)
                    if result:
                        all_qa_pairs.extend(result)
                else:
                    task = self._invoke_llm(chunk_instruction, chunk, current_batch)
                    tasks.append(task)

                generated_count += current_batch

            if rate_limit == 0 and tasks:
                results = await asyncio.gather(*tasks)
                for res in results:
                    if res:
                        all_qa_pairs.extend(res)

        return all_qa_pairs[:count]

    async def _invoke_llm(self, instruction: str, context: str, num_questions: int) -> List[Dict]:
        try:
            prompt_template = ChatPromptTemplate.from_template(
                """
                {instruction}

                Context:
                ---
                {context}
                ---

                Output Format:
                {format_instructions}
                """
            )
            chain = prompt_template | self.llm | self.parser

            response = await chain.ainvoke({
                "instruction": instruction,
                "context": context,
                "format_instructions": self.parser.get_format_instructions()
            })

            if response and "qa_pairs" in response:
                return [dict(pair) for pair in response["qa_pairs"]]
            return []

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return []

    async def _generate_agentic(
        self,
        prompt: str,
        context: str,
        documents: List[Document],
        examples: List[Dict],
        count: int,
        mcp_servers: Optional[List[str]],
        use_rag: bool
    ) -> List[Dict[str, str]]:
        """
        Agentic generation.
        """
        logger.info("Starting Agentic Generation Mode (using create_agent)")

        tools = []
        vectorstore = None

        if use_rag and documents:
            logger.info("Initializing RAG vector store...")
            try:
                splits = self.text_splitter.split_documents(documents)
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    collection_name=f"rag_{random.randint(0, 10000)}"
                )
                retriever = vectorstore.as_retriever()

                @tool
                def search_documents(query: str) -> str:
                    """Search the uploaded documents."""
                    docs = retriever.invoke(query)
                    return "\n\n".join([d.page_content for d in docs])

                tools.append(search_documents)
            except Exception as e:
                logger.error(f"Failed to initialize RAG: {e}")
                use_rag = False

        if not use_rag:
            @tool
            def read_context_files(query: str) -> str:
                """Read the content of the uploaded source files."""
                if not context:
                    return "No source files."
                return context[:10000]
            tools.append(read_context_files)

        @tool
        def get_few_shot_examples() -> str:
            """Get examples of the desired question-answer pairs format."""
            if not examples:
                return "No examples provided."
            return str(examples[:5])
        tools.append(get_few_shot_examples)

        if mcp_servers:
             @tool
             def mcp_search(query: str) -> str:
                 """Search using MCP tools."""
                 return f"Results from MCP for {query}"
             tools.append(mcp_search)

        try:
            system_prompt = (
                f"You are a helpful assistant. {prompt}\n"
                f"Generate exactly {count} question-answer pairs. "
                f"Format output as JSON with key 'qa_pairs'."
            )

            agent = create_agent(
                model=self.llm,
                tools=tools,
                system_prompt=system_prompt
            )

            user_trigger = "Begin data generation."
            result = await agent.ainvoke({"messages": [{"role": "user", "content": user_trigger}]})

            output_text = ""
            if isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
                if hasattr(last_message, "content"):
                    output_text = last_message.content
                elif isinstance(last_message, dict):
                    output_text = last_message.get("content", "")
                else:
                    output_text = str(last_message)
            elif isinstance(result, str):
                output_text = result
            else:
                output_text = result.get("output", "")

            if vectorstore:
                try:
                    vectorstore.delete_collection()
                except:
                    pass

            try:
                parsed = self.parser.parse(output_text)
                if parsed and "qa_pairs" in parsed:
                     return [dict(pair) for pair in parsed["qa_pairs"]]
                return []
            except Exception as parse_error:
                json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
                if json_match:
                     return json.loads(json_match.group(0)).get("qa_pairs", [])
                return []

        except Exception as e:
            logger.error(f"Agent execution failed: {e}. Falling back to pipeline.")
            return await self._generate_pipeline(prompt, context, examples, count)
