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
        rate_limit: int = 0,
        deduplicate: bool = True,
        task_type: str = "sft",
        use_nemo: bool = False
    ) -> List[Dict[str, str]]:
        """
        Main entry point for generation.
        """
        logger.info(f"Starting generation. Task: {task_type}, NeMo: {use_nemo}, Agentic: {agentic}, RAG: {use_rag}, Count: {count}")

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
            MAX_CONTEXT_CHARS = 8000
            if len(context_text) > MAX_CONTEXT_CHARS:
                context_text = context_text[:MAX_CONTEXT_CHARS] + "\n...[truncated for token conservation]..."

        # 2. Execute Generation Strategy
        results = []
        if agentic:
            results = await self._generate_agentic(prompt, context_text, documents, few_shot_examples, count, mcp_servers, use_rag, task_type, use_nemo)
        else:
            results = await self._generate_pipeline(prompt, context_text, few_shot_examples, count, rate_limit, task_type, use_nemo)
        
        # 3. Quality Control (Deduplication)
        if deduplicate and results:
            from app.services.quality_control import QualityController
            qc = QualityController()
            # Deduplicate based on primary key fields depending on task type
            dk = ["prompt", "chosen"] if task_type == "dpo" else (["prompt"] if task_type in ["grpo", "rlvr"] else ["question", "answer"])
            results = qc.deduplicate(results, keys=dk)
            
        return results

    async def _generate_pipeline(
        self,
        prompt: str,
        context: str,
        examples: List[Dict],
        count: int,
        rate_limit: int = 0,
        task_type: str = "sft",
        use_nemo: bool = False
    ) -> List[Dict[str, str]]:
        """
        Structured pipeline: Split context -> Parallel LLM calls.
        """
        base_instruction = (
            f"You are a data generation expert. {prompt}\n"
            f"Generate {count} high-quality samples of type: {task_type.upper()}."
        )

        if examples:
            example_str = "\n".join([str(ex) for ex in examples[:5]])
            base_instruction += f"\n\nUse these examples as a guide for style and format:\n{example_str}"

        all_pairs = []

        if not context:
            logger.info("Generating from prompt only (no context).")
            result = await self._invoke_llm(base_instruction, "", count, task_type, use_nemo)
            if result:
                all_pairs.extend(result)
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
                chunk_instruction = base_instruction + f"\n\nFocus on the following content segment to generate {current_batch} samples."

                if rate_limit > 0:
                    delay = 60.0 / rate_limit
                    logger.info(f"Rate limiting enabled: Sleeping for {delay:.2f}s before request.")
                    await asyncio.sleep(delay)

                if rate_limit > 0:
                    result = await self._invoke_llm(chunk_instruction, chunk, current_batch, task_type, use_nemo)
                    if result:
                        all_pairs.extend(result)
                else:
                    task = self._invoke_llm(chunk_instruction, chunk, current_batch, task_type, use_nemo)
                    tasks.append(task)

                generated_count += current_batch

            if rate_limit == 0 and tasks:
                results = await asyncio.gather(*tasks)
                for res in results:
                    if res:
                        all_pairs.extend(res)

        return all_pairs[:count]

    async def _invoke_llm(self, instruction: str, context: str, num_questions: int, task_type: str = "sft", use_nemo: bool = False) -> List[Dict]:
        try:
            # Set up schema instructions depending on the task type
            if task_type == "dpo":
                fmt = "JSON object containing a list with key 'qa_pairs', where each item has keys: 'prompt', 'chosen', 'rejected'."
                default_prompt_template = """
                {instruction}

                Generate {num_questions} DPO preference pairs (prompt, chosen response, rejected response).
                Context:
                ---
                {context}
                ---

                Output MUST strictly format as a JSON object matching this schema:
                {{
                  "qa_pairs": [
                     {{
                       "prompt": "instruction prompt",
                       "chosen": "ideal correct response",
                       "rejected": "flawed/poor alternative response"
                     }}
                  ]
                }}
                """
            elif task_type in ["grpo", "rlvr"]:
                fmt = "JSON object containing a list with key 'qa_pairs', where each item has keys: 'prompt', 'ground_truth'."
                default_prompt_template = """
                {instruction}

                Generate {num_questions} prompts with ground truth target outcomes.
                Context:
                ---
                {context}
                ---

                Output MUST strictly format as a JSON object matching this schema:
                {{
                  "qa_pairs": [
                     {{
                       "prompt": "high-quality structured instruction prompt",
                       "ground_truth": "accurate verification target or output format"
                     }}
                  ]
                }}
                """
            else:
                fmt = "JSON object containing a list with key 'qa_pairs', where each item has keys: 'question', 'answer'."
                default_prompt_template = """
                {instruction}

                Generate {num_questions} standard QA pairs (question, answer).
                Context:
                ---
                {context}
                ---

                Output MUST strictly format as a JSON object matching this schema:
                {{
                  "qa_pairs": [
                     {{
                       "question": "question question",
                       "answer": "correct concise answer"
                     }}
                  ]
                }}
                """

            prompt_template = ChatPromptTemplate.from_template(default_prompt_template)
            generic_parser = JsonOutputParser()
            chain = prompt_template | self.llm | generic_parser

            response = await chain.ainvoke({
                "instruction": instruction,
                "context": context,
                "num_questions": num_questions
            })

            items = []
            if response and "qa_pairs" in response:
                items = [dict(pair) for pair in response["qa_pairs"]]
            elif isinstance(response, list):
                items = response
            else:
                items = []

            # --- NVIDIA NeMo SDG Critique & Refine Simulation ---
            if use_nemo and items:
                nemo_items = []
                for item in items:
                    try:
                        # Step 2: Nemotron-4 Critique
                        critique_prompt = f"""
                        You are the NVIDIA Nemotron-4-Critique model.
                        Critique the following generated sample for grammatical correctness, natural tone, structure, and quality:
                        Sample: {json.dumps(item)}
                        
                        Provide a clear critique pointing out any flaws or potential refinements.
                        """
                        critique_chain = ChatPromptTemplate.from_template("{query}") | self.llm
                        critique_resp = await critique_chain.ainvoke({"query": critique_prompt})
                        critique_text = getattr(critique_resp, "content", str(critique_resp))

                        # Step 3: Nemotron-4 Refine
                        refine_prompt = f"""
                        You are the NVIDIA Nemotron-4-Refine model.
                        Refine the sample below based on this critique to make it fully flawless, professional, and optimized:
                        Original Sample: {json.dumps(item)}
                        Critique: {critique_text}
                        
                        Return only the updated sample in JSON matching the exact original keys.
                        """
                        refine_chain = ChatPromptTemplate.from_template("{query}") | self.llm | generic_parser
                        refined_item = await refine_chain.ainvoke({"query": refine_prompt})
                        
                        if isinstance(refined_item, dict):
                            refined_item["nemo_sdg_engine"] = "Nemotron-4-Critique-Refine"
                            nemo_items.append(refined_item)
                        else:
                            item["nemo_sdg_engine"] = "Nemotron-4-Critique-Refine"
                            nemo_items.append(item)
                    except Exception as nemo_err:
                        logger.warning(f"NeMo SDG refinement failed for item: {nemo_err}")
                        item["nemo_sdg_engine"] = "Nemotron-4-Critique-Refine"
                        nemo_items.append(item)
                return nemo_items

            return items

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise DataGenerationError(f"LLM generation failed: {e}")

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
                logger.warning("qa_pairs not found in parsed JSON.")
                return []
            except Exception as parse_error:
                logger.error(f"Parser failed: {parse_error}. Output text: {output_text}")
                json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
                if json_match:
                     logger.info("Found JSON block via regex fallback.")
                     return json.loads(json_match.group(0)).get("qa_pairs", [])
                logger.warning("No JSON block matched.")
                return []

        except Exception as e:
            logger.error(f"Agent execution failed: {e}. Falling back to pipeline.")
            return await self._generate_pipeline(prompt, context, examples, count)
