# app/services/agent_generator.py
import asyncio
import logging
import re
import json
import random
from typing import List, Dict, Any, Optional

from fastapi import UploadFile
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_openai_functions_agent, AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage

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

    def __init__(self, provider: str, model: Optional[str] = None, temperature: float = 0.7):
        self.settings = get_settings()
        self.provider = provider

        # Initialize LLM Provider Factory (it handles model selection)
        self.llm_factory = LLMProviderFactory(provider, model_name=model, temperature=temperature)
        self.llm = self.llm_factory.llm
        self.parser = JsonOutputParser(pydantic_object=QAList)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
        )

    async def generate(
        self,
        prompt: str,
        files: List[UploadFile],
        demo_file: Optional[UploadFile],
        count: int,
        agentic: bool = False,
        mcp_servers: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Main entry point for generation.
        """
        logger.info(f"Starting generation. Agentic: {agentic}, Count: {count}")

        # 1. Process Inputs
        context_text = ""
        if files:
            logger.info(f"Processing {len(files)} source files...")
            extracted_texts = []
            for file in files:
                try:
                    text = await TextExtractor.extract(file)
                    extracted_texts.append(f"--- File: {file.filename} ---\n{text}\n")
                except Exception as e:
                    logger.warning(f"Skipping file {file.filename} due to error: {e}")
            context_text = "\n".join(extracted_texts)

        few_shot_examples = []
        if demo_file:
            logger.info(f"Processing demo file {demo_file.filename}...")
            try:
                content = await demo_file.read()
                few_shot_examples = TextExtractor.parse_csv_to_dicts(content)
            except Exception as e:
                logger.warning(f"Failed to parse demo file: {e}")

        # 2. Execute Generation Strategy
        if agentic:
            return await self._generate_agentic(prompt, context_text, few_shot_examples, count, mcp_servers)
        else:
            return await self._generate_pipeline(prompt, context_text, few_shot_examples, count)

    async def _generate_pipeline(
        self,
        prompt: str,
        context: str,
        examples: List[Dict],
        count: int
    ) -> List[Dict[str, str]]:
        """
        Structured pipeline: Split context -> Parallel LLM calls.
        """
        # Prepare the base prompt template
        base_instruction = (
            f"You are a data generation expert. {prompt}\n"
            f"Generate {count} question-answer pairs."
        )

        if examples:
            example_str = "\n".join([str(ex) for ex in examples[:5]]) # Limit examples
            base_instruction += f"\n\nUse these examples as a guide for style and format:\n{example_str}"

        all_qa_pairs = []

        if not context:
            # Generate from prompt only
            logger.info("Generating from prompt only (no context).")
            result = await self._invoke_llm(base_instruction, "", count)
            if result:
                all_qa_pairs.extend(result)
        else:
            # Generate from context
            chunks = self.text_splitter.split_text(context)
            if not chunks:
                 chunks = [context]

            logger.info(f"Context split into {len(chunks)} chunks.")

            pairs_per_chunk = max(1, self.settings.QA_BATCH_SIZE)
            total_needed = count

            tasks = []
            generated_count = 0

            # Let's iterate chunks and spawn tasks until we have enough potential pairs
            chunk_pool = list(chunks)
            random.shuffle(chunk_pool)

            while generated_count < total_needed:
                if not chunk_pool:
                    chunk_pool = list(chunks) # Refill if exhausted
                    random.shuffle(chunk_pool)

                chunk = chunk_pool.pop()
                current_batch = min(pairs_per_chunk, total_needed - generated_count)

                # Construct a prompt for this chunk
                chunk_instruction = base_instruction + f"\n\nFocus on the following content segment to generate {current_batch} pairs."

                task = self._invoke_llm(chunk_instruction, chunk, current_batch)
                tasks.append(task)
                generated_count += current_batch

            results = await asyncio.gather(*tasks)
            for res in results:
                if res:
                    all_qa_pairs.extend(res)

        # Trim to exact count
        return all_qa_pairs[:count]

    async def _invoke_llm(self, instruction: str, context: str, num_questions: int) -> List[Dict]:
        """Helper to call LLM chain."""
        try:
            # Create a specific chain for this call to inject prompt
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
        examples: List[Dict],
        count: int,
        mcp_servers: Optional[List[str]]
    ) -> List[Dict[str, str]]:
        """
        Agentic generation using LangChain agents.
        """
        logger.info("Starting Agentic Generation Mode")

        # Define Tools

        @tool
        def read_context_files(query: str) -> str:
            """Read the content of the uploaded source files. Use this to understand the domain."""
            if not context:
                return "No source files were uploaded."
            # In a real agent, we might search specifically, but here we return full context (truncated if needed)
            return context[:10000] # Return first 10k chars to avoid token limits in tool output

        @tool
        def get_few_shot_examples() -> str:
            """Get examples of the desired question-answer pairs format."""
            if not examples:
                return "No examples provided."
            return str(examples[:5])

        # Placeholder for MCP tools
        mcp_tools = []
        if mcp_servers:
             logger.info(f"Connecting to MCP servers: {mcp_servers}")
             # In a real implementation, we would inspect the MCP servers and add their tools dynamically.
             # Since we don't have real MCP infrastructure here, we log it.
             # We can add a mock tool to simulate MCP capability.
             @tool
             def mcp_search(query: str) -> str:
                 """Mock MCP search tool."""
                 return f"Results from MCP for {query}"
             mcp_tools.append(mcp_search)

        tools = [read_context_files, get_few_shot_examples] + mcp_tools

        # Choose Agent Type
        # OpenAI Functions Agent is robust, but only works with OpenAI-compatible models.
        # Tool Calling Agent is more generic for newer LangChain versions.
        # For other providers like Groq/Google, we might need ReAct or similar if they don't support function calling properly via LangChain.
        # Assuming the LLM supports tool calling (most modern ones do).

        try:
            # Create Prompt
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant task with generating synthetic data."),
                    ("user", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            # Note: create_tool_calling_agent requires model that supports bind_tools
            # If the provider doesn't support it, we fall back to ReAct or just standard pipeline.
            # Groq, OpenAI, Google generally support tool calling in recent versions.

            agent = create_tool_calling_agent(self.llm, tools, prompt_template)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            agent_instruction = (
                f"{prompt}\n\n"
                f"You have access to context files and examples via tools. "
                f"Use them to understand the content and style. "
                f"Then, generate exactly {count} question-answer pairs. "
                f"Format the final output strictly as a JSON object with a single key 'qa_pairs' containing a list of objects with 'question' and 'answer' keys."
            )

            result = await agent_executor.ainvoke({"input": agent_instruction})
            output_text = result["output"]

            # Parse the output
            # The agent might output text + JSON or just JSON. We try to parse it.
            # Use the parser we already have
            try:
                # We reuse the parser but we need to pass a prompt to it usually?
                # No, parser.parse(text) works.
                parsed = self.parser.parse(output_text)
                if parsed and "qa_pairs" in parsed:
                     return [dict(pair) for pair in parsed["qa_pairs"]]
                return []
            except Exception as parse_error:
                logger.warning(f"Failed to parse agent output directly: {parse_error}. Trying to extract JSON.")
                # Fallback: Try to find JSON in text
                import re
                json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
                if json_match:
                     import json
                     return json.loads(json_match.group(0)).get("qa_pairs", [])
                return []

        except Exception as e:
            logger.error(f"Agent execution failed: {e}. Falling back to pipeline.")
            return await self._generate_pipeline(prompt, context, examples, count)
