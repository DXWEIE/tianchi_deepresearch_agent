import pandas as pd
import json
from langchain_community.document_loaders import WikipediaLoader
import re
import math
from typing import List, Dict, Optional
import wikipedia
from urllib.parse import urlparse
import time
import numpy as np
from http import HTTPStatus
import dashscope
import ast

import os
from openai import OpenAI
import json
import dashscope
import numpy as np
from http import HTTPStatus
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

import math

import re
from bs4 import BeautifulSoup
from typing import Any, Dict, Optional

from ddgs import DDGS

# 这里需要注意，可能很长很长，需要做chunk切分内容，先从内容相关性上来做，选择最相近的topN个chunk来回答问题
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import serpapi

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Any

# 待修改为可并发的
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

QWEN_KEY = 'sk-xxx'
DEBUG_MODE = True
IQS_KEY ="AZxxx"
search_scan_key = "vcans_xxx"

serpapi_key = "xxx"

EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024
MAX_RETRIES = 3
MAX_TEXT_LENGTH = 40960


TOTAL_INPUT_TOKENS = 0
TOTAL_OUTPUT_TOKENS = 0
TOTAL_COST_ESTIMATE = 0
SEARCH_TIMES = 0
SINGLE_MAX_TIME = 450 # 450秒


language = 'english'

def print_and_log(*args, file='output_zh_11.txt', **kwargs):
    need_print = kwargs.pop('need_print', False)
    # 1. 屏幕打印通常不需要锁，因为 stdout 本身有缓冲机制，且就算乱了也能看
    if True:
        try:
            print(*args, **kwargs)
        except Exception:
            pass # 屏幕打印失败也忽略

    try:
        # encoding='utf-8' 防止中文乱码
        with open(file, 'a', encoding='utf-8') as f:
            print(*args, file=f, **kwargs)
            
    except Exception:
        pass

print_and_log("This appears in both!",need_print=True)


MODEL_PRICING = {
    "qwen-plus": {"input": 0.8, "output": 2.0},  # 元 / 百万 tokens
    "qwen-flash": {"input": 0.25, "output": 1.5},
    "qwen-turbo": {"input": 0.3, "output": 0.6},
    "qwen3-max": {"input": 2.5, "output": 10},
    "qwen3.5-plus": {"input": 0.8, "output": 4.8}
}

plan_model = "qwen3-max"
base_model = "qwen-plus"
plus_model = "qwen-plus"
max_model = "qwen3-max"
flash_model = "qwen-flash"
MAX_WORKER = 4
web_cache = {} # question_idx -> {url:content}
wiki_cache = {}
wiki_query_cache = {}
iqs_cache = {}
ddgs_cache = {}
google_cache = {}

evaluate_search_result_template_en="""You are a precise information extraction assistant. Your task is to produce a concise, self-contained factual statement that captures the exact claim from the content relevant to the sub-question.

Instructions:
1. ⚠️ NEVER output only a single word or bare noun phrase (e.g., "Tesla", "2023", "CEO", "北京").  
   Always include enough context to form a minimal factual statement (e.g., include a verb or linking phrase like "is", "was named", "founded", "担任").
2. Extract verbatim or near-verbatim text from the content. Do not paraphrase unless necessary for grammatical coherence.
3. Preserve official names exactly as they appear, especially in encyclopedic sources like Wikipedia or Baidu Baike, which often use the format: `中文名 (English name)`.
4. The extracted snippet must be self-contained: a reader should understand what claim is being made without seeing the original page.
5. Use the same language as the search result content. If the content is in Chinese, output in Chinese; if in English, output in English.
6. If no such explicit statement exists, output exactly: "NOT RELEVANT".

Example (English):
- Sub-question: "Who is the CEO of OpenAI?"
- Content: "Sam Altman has been the CEO of OpenAI since 2019, with a brief departure in November 2023."
- ✅ Good output: "Sam Altman has been the CEO of OpenAI since 2019"
- ❌ Bad output: "Sam Altman"

Example (Chinese, Baidu Baike style):
- Sub-question: "萨姆·奥特曼是谁？"
- Content: "萨姆·奥特曼（Sam Altman）是OpenAI的首席执行官，也是前Y Combinator的总裁。"
- ✅ Good output: "萨姆·奥特曼（Sam Altman）是OpenAI的首席执行官"
- ❌ Bad output: "萨姆·奥特曼"
- ❌ Bad output: "Sam Altman"

Another example (English):
- Sub-question: "When was SpaceX founded?"
- Content: "Elon Musk founded SpaceX in March 2002 in El Segundo, California."
- ✅ Good output: "Elon Musk founded SpaceX in March 2002"
- ❌ Bad output: "March 2002"

Now process the following:
- User Question: {main_question}
- Sub-question: {sub_question}
- Search Query: {search_query}
- URL: {URL}
- Search Result Content: {content}

Your output is:"""

evaluate_search_result_template = evaluate_search_result_template_en


# visit相关
# 决定visit哪个，是否visit
# 决定visit哪个，是否visit
select_top_search_results_template_en = """You are a precise and analytical AI assistant. Your task is to select up to **three** search results that are ** most likely to contain the answer** to a given sub-question.

Given:
- The **main user question** (for context)
- The **sub-question** (the specific part you need to answer)
- The **search query** used to retrieve the results
- A list of **search results**, each with an index (`idx`), title, and content

Instructions:
1. Carefully evaluate **all** provided search results.
2. Select **at most 3** results that:
   - Directly or indirectly provide factual information helpful for answering the **sub-question**
   - Are more informative than others (e.g., contain key entities, numbers, dates, explanations, or direct answers)
3. **Do NOT select** results that are generic, promotional, navigational, or lack concrete information.
4. Return **only the `idx` values** (as strings) of at most 3 results, in order of relevance (most relevant first).
5. If **none** of them contain the answer, return an empty list (i.e, `[]`).

Rules:
- Do NOT use external knowledge.
- Do NOT assume missing context.
- Partial relevance (e.g., mentions a related concept, organization, or timeframe) is acceptable if it helps narrow down the answer.
- Ignore boilerplate text, ads, or repeated phrases.

Output format: a list of index seperated by commas (e.g., `[0, 2]`). Never add extra text.

Main User Question:
{main_question}

Sub-Question:
{sub_question}

Search Query:
{search_query}

Search Results:
{formatted_results}

Your output is:"""



# visit后提取信息
EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""


evaluate_search_result_template_en = """You are a precise information extraction assistant. Your task is to convert a search result into a single, self-contained factual sentence that directly addresses the sub-question.

Instructions:
1. If the content provides any information that helps answer the sub-question, synthesize it into **one clear, grammatical sentence**.
2. The sentence must:
   - Explicitly mention the **key entity from the sub-question** (e.g., number, name, title, date etc.)
   - State the **relevant fact** about it (e.g., its name, title, age, etc.)
   - Use only information from the content + the explicit terms in the search query (you may use the query to resolve ambiguity)
3. Generally, you need to use English in your main answer. For Chinese user question and Chinese sub-question and search query (e.g., Chinese historical figures, actors, schoolars, films, cities, organizations, etc.), you can choose to use Chinese.
4. If no useful information is present, output exactly: "NOT RELEVANT".

**CRITICAL RULE: Formal Entity Name Preservation (Especially for Baidu Baike & Wikipedia)**
- When extracting names from **Baidu Baike** or **Wikipedia**, you MUST preserve the full entity string including any parenthetical information if it is relevant to the question or sub-question.
- **Pattern Recognition**: In these sources, entity titles often follow the format `Name (Disambiguation/English Name/Official Name)`.
  - Example: "·奥尔特萨姆曼（Sam Altman）", "埃隆·里夫·马斯克（Elon Reeve Musk）"


Now process the following:
- User Question: {main_question}
- Sub-question: {sub_question}
- Search Query: {search_query}
- URL: {URL}
- Search Result Content: {content}


Your output is:"""

evaluate_search_result_template = evaluate_search_result_template_en


answer_sub_question_template_en = """You are a rigorous and transparent AI assistant. Your task is to answer a **sub-question** using all available contextual information.

You are given:
1. A **history of previous reasoning steps** (what has already been established)
2. The current **sub-question** you must address
3. **Search result summaries**, which may include:
   - High-level summaries from initial search
   - Detailed page content summaries (from crawled pages)

Your job:
- Analyze all provided information **strictly without external knowledge**.
- Determine whether the sub-question can be:
  a) **Fully and uniquely answered** → use `"can_answer": "yes"`
  b) **Has multiple plausible answers** (e.g., one ID maps to several entities, conflicting records, or unresolved ambiguity) → use `"can_answer": "ambiguous"`
     - List all reasonable candidates in `"answer_candidates"`
     - Explain what **discriminating information** is needed to resolve it
  c) **Only partially answerable** (e.g., you know the year but not the event, or the name but not the role) → use `"can_answer": "partial"`
  d) **Not answerable at all** (no relevant info found) → use `"can_answer": "no"`
  e) Infer from reliable information that the question is contradictory (e.g., a sub-question asks which Asian company an entity belongs to, but the provided solid information indicates that the entity belongs to a European company) → `"can_answer": "contradictory"`

**Handling Conflicting Information & Source Credibility:**
- When multiple sources provide conflicting facts, **resolve conflicts by prioritizing higher-credibility sources**—do not require high-credibility confirmation as an absolute prerequisite for every claim.
- Use the following credibility hierarchy to weigh evidence:
**High-Credibility Sources (Preferred for verification):**
    - *Chinese context*: Baidu Baike (baike.baidu.com) for widely accepted entity facts (unless contested), official government (.gov.cn) or institutional (.org.cn) websites, accredited universities (.edu.cn).
    - *English/global context*: Wikipedia (for well-established topics), official organization websites (.org, .gov), accredited academic institutions (.edu).
    - *Technical/academic claims*: Peer-reviewed publications, official technical documentation, university repository papers (e.g., arXiv with institutional affiliation).

  **Low-Credibility Sources (Use only if no higher-quality source exists and information is consistent):**
    - Baidu Zhidao, Baijiahao, Tieba, Weibo, Douban, personal blogs, unverified news aggregators.
- **Key Principle**:  
  → If a high-credibility source contradicts a low-credibility one, **trust the high-credibility source**.  
  → If only low-credibility sources exist but they consistently report the same fact (and it’s plausible), you may provisionally accept it—but remain cautious and avoid overconfident assertions.  
  → Never treat low-credibility sources as definitive when high-credibility alternatives are available.

**CRITICAL RULE: Formal Entity Name Preservation (Especially for Baidu Baike & Wikipedia)**
- When extracting names from **Baidu Baike** or **Wikipedia**, you MUST preserve the full entity string including any parenthetical information.
- **Pattern Recognition**: In these sources, entity titles often follow the format `Name (Disambiguation/English Name/Official Name)`.
  - Example: "萨姆·奥尔特曼（Sam Altman）", "埃隆·里夫·马斯克（Elon Reeve Musk）".
- **Action**: Do NOT strip the content inside the parentheses `(...)`. The text inside often contains the English name, specific identity marker, or formal name required to distinguish the entity from others with similar names.
- **Output Requirement**: If the source provides "Name (Extra Info)", your `"answer"` field must reflect this full form if it is crucial for identification, or explicitly state both versions (e.g., "Zhang San (also known as John Zhang)"). Never reduce "Name (Info)" to just "Name" if the "(Info)" part defines the specific entity being discussed.

Rules:
- Never hallucinate. If uncertain, downgrade confidence or mark as ambiguous/partial.
- In the "reason" field, briefly explain how the final answer was derived, with explicit reference to the provided Search Result Summaries. Specifically:
    - If Baidu Baike (Baidu Baike is used for extracted standard name) is present, directly quote or closely paraphrase its opening definition sentence — including the full name with parenthetical foreign original name if it appears (e.g., “萨尔布吕肯乒乓球俱乐部（德语：1. FC Saarbrücken-Tischtennis）是位于德国萨尔州萨尔布吕肯的职业乒乓球俱乐部...”).
    - Explicitly mention the source, e.g., “According to Baidu Baike: ‘...’”.
    - Do not omit or alter the parentheses in the quoted snippet.
    - Keep the entire reason to 1–3 sentences.
- In `"answer"`, provide a clear natural-language statement summarizing what is known. 
- **Language & Naming**: Your `"answer"` must be written in the **SAME language** as the primary evidence found in `Search Result Summaries`. 
  - **DO NOT** transliterate or translate proper nouns (names, places, organizations) unless the evidence itself provides the translation. 
  - **If the evidence provides both Chinese and English (e.g., in parentheses), YOU MUST INCLUDE BOTH in your answer.**
- In `"answer_candidates"`, only include short, distinct candidate answers (e.g., ["tiger", "dragon"]), and **only when `"can_answer"` is `"ambiguous"`**.

- `"confidence"` reflects how reliable and consistent the supporting evidence is.

Output format: a **valid JSON object** with exactly these fields:
- "reason": a short string that explains the rationale behind the answer supported by the evidence (Search Result Summaries)
- "can_answer": one of "yes", "ambiguous", "partial", "no" or "contradictory"
- "answer": a string (can be empty only if "can_answer" is "no"; if contradictory, answer based on reliable information even if it's the contradictory result)
- "answer_candidates": a list of strings (e.g., ["candidate1", "candidate2"]); must be empty list [] if not "ambiguous"
- "confidence": one of "high", "medium", or "low"

Do NOT output anything else—no explanations, no markdown, no extra fields. Only the JSON.

Previous Reasoning History:
{history_info}

Current sub-question:
{sub_question}

Search Result Summaries:
{search_summary}

Your JSON output is:"""



final_answer_template_en = """You are a precise and transparent AI assistant. Your task is to answer the original question using **only** the verified global key information provided.

### Output Format
Output a valid JSON object with exactly two fields:
- "think": a brief explanation of how you derived the answer from the key information, including why the chosen format matches the question's requirement.
- "answer": the final answer, **in the exact format implied or required by the original question**, based solely on the provided key information.
    • If the question asks for a name, use the form (e.g., full name, last name only, etc.) that best matches both the question’s phrasing and the available evidence.
    • If the question asks for a date/time/number/code/etc., output it exactly as it appears or is logically implied in the key info.
    • If the key information does not contain enough to determine the answer **in the required form**, output "Unknown".
    • If ambiguity exists among the key information, briefly analyze the possibilities in the "think" section, but provide the most probable answer in the "answer" section.

### Answer Rules
- NEVER use external knowledge or assumptions.
- NEVER add units, formatting, or structure not justified by the question or key info.
- NEVER output anything except the JSON object.

### Language Rules
The answer must be a **single, concise phrase or name**, and its language should follow these priorities:
   - **If the question explicitly requests a specific language or giving those examples**, obey it exactly
   - Otherwise, you must use English answer for English question and Chinese answer for Chinese question.
      * For English question, if no specific formatting examples or requirements are given in the question, the final answer must be in English. If the original question provides specific name examples (e.g., "John Fitzgerald Kennedy") or explicitly asks for a full name/middle name (or express as "全名"), you MUST match that format exactly, including middle names or initials. If NO specific formatting examples or requirements are given in the question, default to the concise **"First Name Last Name"** format by **omitting middle names**, unless the middle name is critical for disambiguation (e.g., distinguishing two famous people with the same first/last name) or is universally recognized as part of the common name (e.g., "Martin Luther King, Jr").
      * For Chinese question, if no specific formatting examples or requirements are given in the question, you must to answer in Chinese with its standard full Chinese name (e.g., "华罗庚", "李清照","尼格买提·热合曼").
   - Output ONLY the core name. Strictly REMOVE all parenthetical suffixes containing codes, dates, IDs, or aliases (e.g., strip "(A/1998 74)", "(1900-1990)", "(aka ...)"), unless the code IS the only existing name.
   - **Never output a name in the format "中文名（外文名）" or "外文名（中文名）" or "中文名 (别名)" or "xxx (xxx)"; choose the standard canonical form based on the language rules above (e.g., "萨尔布吕肯乒乓球俱乐部（德语：1. FC Saarbrücken-Tischtennis）是位于德国萨尔州萨尔布吕肯的职业乒乓球俱乐部" should be "萨尔布吕肯乒乓球俱乐部" if asked for Chinese answer).** 

### Source Credibility Guidelines
**High Credibility Sources** (Crucial for Confidence Scoring):
    - Chinese Context: *Baidu Baike** (baike.com for general entity facts, unless disputed), Official websites (gov, org), educational institutions (edu).
    - English/Global Context: **Wikipedia** (generally reliable for established facts), Official organization websites(org), educational institutions(edu)
    - **Academic/Technical**: Peer-reviewed papers, official websites (arxiv, journals, university repository publications)

---

### Input Data
Original question:
{question}

Global key information summary (verified facts only):
{keyinfo_summary_global}

Your JSON output is:"""



strong_fallback_template_en = """You are a capable and responsible AI assistant. All attempts to answer the question using external evidence have failed. No verified facts are available in the provided context.

Now, you **must** provide a final response using your internal knowledge and reasoning.

Output a valid JSON object with exactly two fields:
- "think": Explain clearly that this answer is based on your internal knowledge (not external evidence) if external evidence can not lead to any potential answer.
- "answer": Provide a **concrete, properly formatted answer that strictly matches the expected format of the original question**. 
    • Never output "Unknown", "N/A", or refuse to answer.
    • If the question asks for a name, give a plausible full name in an appropriate format.
    • If it asks for a date, number, code, list, etc., provide a realistic example or best-known value in the correct form.
    • The answer must be usable as-is by the user.
    • Note: If the question includes **examples in a specific language or explicitly asks for English**, follow that. Otherwise, use the same language in "answer" field as the question.

Rules:
- This is the FINAL fallback. You MUST produce an answer.
- Keep "answer" concise and directly responsive — no disclaimers inside it.
- Only the JSON object is allowed — no extra text.

Original question:
{question}

Global key information summary (verified facts only):
{keyinfo_summary_global}

Your JSON output is:"""

final_answer_template = final_answer_template_en
strong_fallback_template = strong_fallback_template_en


answer_sub_question_template = answer_sub_question_template_en


final_structure_template = """You now have enough verified information to give a final answer.

Output ONLY the exact answer to the original question with the format as requested — nothing else.

Rules:
The answer must be a **single, concise phrase or name**, and its language should follow these priorities:
   - **If the question explicitly requests a specific language or giving those examples**, obey it exactly
   - Otherwise, you must use English answer for English question and Chinese answer for Chinese question.
      * For English question, if no specific formatting examples or requirements are given in the question, the final answer must be in English. If the original question provides specific name examples (e.g., "John Fitzgerald Kennedy") or explicitly asks for a full name/middle name (or express as "全名"), you MUST match that format exactly, including middle names or initials. If NO specific formatting examples or requirements are given in the question, default to the concise **"First Name Last Name"** format by **omitting middle names**, unless the middle name is critical for disambiguation (e.g., distinguishing two famous people with the same first/last name) or is universally recognized as part of the common name (e.g., "Martin Luther King, Jr").
      * For Chinese question, if no specific formatting examples or requirements are given in the question, you must to answer in Chinese with its standard full Chinese name (e.g., "华罗庚", "李清照","尼格买提·热合曼").
   - Output ONLY the core name. Strictly REMOVE all parenthetical suffixes containing codes, dates, IDs, or aliases (e.g., strip "(A/1998 74)", "(1900-1990)", "(aka ...)"), unless the code IS the only existing name.
   - **Never output a name in the format "中文名（外文名）" or "外文名（中文名）" or "中文名 (别名)" or "xxx (xxx)"; choose the standard canonical form based on the language rules above (e.g., "萨尔布吕肯乒乓球俱乐部（德语：1. FC Saarbrücken-Tischtennis）是位于德国萨尔州萨尔布吕肯的职业乒乓球俱乐部" should be "萨尔布吕肯乒乓球俱乐部" if asked for Chinese answer).** 

2. **Special rule for NUMBERS:**
   - If the original question provides a **numeric example** (e.g., "such as 7.6", "format like 123.45", or shows a number in the expected answer style),  
     → your answer **MUST match that numeric format exactly** in terms of:
       • Number of decimal places (e.g., if example is "7.6" → one decimal place; "1.80" is INVALID, should be "1.8")
       • No unnecessary leading/trailing zeros (e.g., "01.8" or "1.800" are INVALID if example is "1.8")
       • No commas, spaces, or units (e.g., "1,800" or "1.8 kg" are INVALID unless explicitly shown in example)
   - If no numeric example is given, output the number in its **simplest standard form** (e.g., "1.8", not "1.80"; "5", not "5.0")

3. Do NOT include:
   - Full sentences or explanations
   - Quotes, periods (unless part of a version like "Qwen3"), commas, parentheses, or extra punctuation
   - Articles ("the", "a"), filler words ("I think", "probably", "maybe")
   - Units (%, $, kg, etc.) unless the question example includes them

4. If the answer is a character, role, or person, give **only the canonical name as it would appear in official or widely recognized sources** — matching the format implied by the question.

5. Wrap your answer in <answer> tags and output nothing else.

Original Question: {question}
Original Answer: {answer}
Summary Info: {summary}
"""



react_prompt_template = """You are an expert AI assistant solving complex factual questions (GAIA-style benchmark).

**CRITICAL OUTPUT FORMAT RULES (VIOLATION = FAILURE):**
1. ✅ Output MUST be valid JSON object with exact keys: ["think", "goal", "query", "wiki", "answer"]
2. ✅ "think": max 3 sentences, ≤800 words, brief reasoning
3. ✅ "goal": single sentence, ≤100 words, state search purpose
4. ✅ "query": plain text search query in appropriate language used for web search
5. ✅ "wiki": OPTIONAL, only for verifying specific entities (person/org/product/term) which may appear in the final answer
6. ✅ "answer": OPTIONAL, ONLY output after fully verified
7. ❌ NO additional keys beyond the 5 specified
8. ❌ NO markdown code blocks (no ```json ... ```)
9. ❌ NO nested JSON or complex structures

**JSON Output Structure (STRICT ORDER):**
{{
  "think": "Brief reasoning, max 3 sentences, less than 800 words",
  "goal": "Single sentence, less than 100 words",
  "query": "Search query in appropriate language",
  "wiki": "Optional: exact entity name",
  "answer": "Optional: ONLY when fully verified"
}}

**Field Rules**
- "think": 
    • First, remind ALL hard constraints from the original question (e.g., time, location, event).
    • Then, if proposing a candidate, explicitly state WHICH key claims remain unverified.
    • If prior searches confirm a candidate can not meet a critical constraint, ABANDON that candidate and pivot to a NEW distinguishing clue.
    • If a candidate appears plausible early in the think process, explicitly verify its key claims rather than discarding it prematurely.
- "goal": ONE sentence purpose (e.g., "Confirm the official Chinese name of WeChat's founder")
- "query": Use MOST EFFECTIVE language (Chinese for China topics, English otherwise)
- "wiki": Only for verifying specific entities that may appear in final answer
- "answer": ONLY when factually correct, all constraints satisfied, official names have been checked

**Search Guidelines**
- Default in English
- If Chinese question and Chinese topic (e.g., Chinese historical figures, actors, schoolars, films, cities, organizations, etc.) → can try Chinese query
- If last 2 searches failed:
  - switch query to Chinese if the Original question is in Chinese, otherwise try to rethink assumption and find the most distinguishable clues for searching

**Answer Rules**
- ONLY include "answer" key when:
  ✓ Factually correct
  ✓ All constraints satisfied (time, location, format, language)
  ✓ Official names has been cross-checked via Wikipedia/baike (which means you can not return answer in the first step, at least one search must be conducted)
  ✓ Language/format matches requirement
- Otherwise: set "answer" to empty "" or omit the key
- Note: If the question includes **examples in a specific language or explicitly asks for English**, follow that. Otherwise, use the same language in "answer" field as the question.

**Examples**

✅ CORRECT (no final answer):
{{
  "think": "I have two candidate names but need to verify which is official. Last search returned irrelevant results, so I'll try a Chinese query.",
  "goal": "Verify the official Chinese name of Tencent's COO",
  "query": "腾讯 首席运营官 任宇昕",
  "wiki": "任宇昕",
  "answer": ""
}}

✅ CORRECT (with final answer):
{{
  "think": "All constraints verified. Official name confirmed via Wikipedia cross-check.",
  "answer": "任宇昕"
}}

Now solve this task:
Original Question: {question}
Current Context and core findings log:
{context}
Remember: Output ONLY raw JSON, NO markdown, NO extra keys!
"""



think_compress_template = """You are compressing the internal monologue (thinking process) of an AI agent. 
Your goal is to reduce the token count by **70%~90%** while preserving the **logical skeleton** required to understand how the final answer was derived.

**INPUT CONTEXT:**
The input is a raw "Chain of Thought" containing searches, reasoning, self-corrections, and dead ends.

**PRESERVATION PRIORITIES (Must Keep):**
1. **Search Intent & Query**: What was looked for and the exact query string.
2. **Key Findings**: Specific entities, numbers, dates, or facts extracted from search results.
3. **Hypothesis Management**: 
   - All candidate answers generated (e.g., "Cand: A", "Cand: B").
   - All rejected hypotheses with the specific reason for rejection (e.g., "Drop: A (conflict with source X)").
4. **Logic Jumps**: Critical deductions that connect evidence to conclusions.

**AGGRESSIVE COMPRESSION RULES (Must Remove):**
- **NO Conversational Fillers**: Remove "Let me think", "I need to", "Wait", "Okay", "So", "Hmm".
- **NO Self-Doubt Narratives**: Convert "I am not sure if X is true, maybe I should check Y" -> "Check Y to verify X".
- **NO Repetition**: If a fact is established once, do not restate it in later paragraphs.
- **NO Verbose Snippets**: Do not copy-paste search results. Extract only the relevant value.
- **NO Polite Formatting**: Use dense bullet points or symbolic notation.


**RAW THINKING PROCESS TO COMPRESS:**
{full_text}

**COMPRESSED THINKING SKELETON:**
"""

compress_template = """You are compressing the interaction history of a ReAct agent solving a complex factual question.
Your task is to produce a **concise, dense summary (less than input).

**CRITICAL: Preserve ALL of the following:**
1. Every UNIQUE search step with:
   - Search goal (what we wanted to find)
   - Exact query used
   - Key findings (ONLY if they led to new conclusions or ruled out hypotheses) and clues
2. ALL intermediate conclusions and insights (even if later revised)
3. ALL answer candidates mentioned
4. Ruled-out hypotheses (and WHY they were ruled out)
5. Critical evidence (e.g., entities or factual claims collected from authoritative sources)

**SAFE to remove:**
- Repetitive long reasoning steps
- Self-doubt expressions ("I'm not sure...", "Let me think...")
- Obvious statements restating known facts
- Duplicate queries or near-duplicate searches
- Verbose formatting and transition phrases

**Output format:**
- Use bullet points for clarity
- Keep factual, dense information
- Preserve numbers, dates, names exactly in original language

**DO NOT:**
- Merge distinct conclusions into vague statements
- Remove answer candidates even if they seem wrong

Full history to compress:
{full_text}

Compressed history (preserve all key information):
"""

short_compress_template = """Compress the following text into a shorter version while strictly preserving its original meaning and key information. Output only the compressed plain text as a single continuous paragraph. Do not use bullet points, numbered lists, bold formatting, or any markdown structures. Do not include any introductory phrases, explanations, or thinking processes. Just provide the final compressed text directly.
Text to compress:
{full_text}"""


name_consistency_verification_template = """You are an expert AI assistant performing a final **language validation** for a GAIA-style factual question.

You are given:
- The **original question** (analyze its language, explicit instructions, and cultural context)
- A **candidate answer**
- A **core findings log**: verified facts from prior reasoning

Your task is to output the answer **in the exact language demanded by the question**, with **authoritative precision**.
Your task is not to find the right language format of the given answer.
So, in most cases, you need to launch a search for the official name on Wikipedia (for engilish entities), Baidu Baike (for Chinese entities), or some official sources.

**CRITICAL OUTPUT FORMAT RULES (VIOLATION = FAILURE):**
1. ✅ Output MUST be valid JSON only - NO markdown, NO code blocks, NO extra text
2. ✅ EXACTLY ONE `think` field at the beginning (max 2 sentences)
3. ✅ EXACTLY ONE `action` field: either "final_answer" or "need_verification"
4. ✅ If action = "final_answer": include `answer` field ONLY
5. ✅ If action = "need_verification": include `goal` and `query` fields ONLY
6. ❌ NEVER mix answer with goal/query in the same output
7. ❌ NO nested objects - flat JSON structure only

**Output Structure (STRICT):**

Option A - Answer Ready:
{{
    "think": "...",
    "action": "final_answer",
    "answer": "your final answer"
}}

Option B - Need Verification:
{{
    "think": "...",
    "action": "need_verification",
    "goal": "the search goal",
    "query": "the search query"
}}

**Validation Rules (for `think` field):**
The answer must be a **single, concise phrase or name**, and its language should follow these priorities:
   - **If the question explicitly requests a specific language or giving those examples**, obey it exactly
   - Otherwise, you must use English answer for English question and Chinese answer for Chinese question.
      * For English question, if no specific formatting examples or requirements are given in the question, the final answer must be in English. If the original question provides specific name examples (e.g., "John Fitzgerald Kennedy") or explicitly asks for a full name/middle name (or express as "全名"), you MUST match that format exactly, including middle names or initials. If NO specific formatting examples or requirements are given in the question, default to the concise **"First Name Last Name"** format by **omitting middle names**, unless the middle name is critical for disambiguation (e.g., distinguishing two famous people with the same first/last name) or is universally recognized as part of the common name (e.g., "Martin Luther King, Jr").
      * For Chinese question, if no specific formatting examples or requirements are given in the question, you must to answer in Chinese with its standard full Chinese name (e.g., "华罗庚", "李清照","尼格买提·热合曼").
   - Output ONLY the core name. Strictly REMOVE all parenthetical suffixes containing codes, dates, IDs, or aliases (e.g., strip "(A/1998 74)", "(1900-1990)", "(aka ...)"), unless the code IS the only existing name.
   - **Never output a name in the format "中文名（外文名）" or "外文名（中文名）" or "中文名 (别名)" or "xxx (xxx)"; choose the standard canonical form based on the language rules above (e.g., "萨尔布吕肯乒乓球俱乐部（德语：1. FC Saarbrücken-Tischtennis）是位于德国萨尔州萨尔布吕肯的职业乒乓球俱乐部" should be "萨尔布吕肯乒乓球俱乐部" if asked for Chinese answer).** 

**Action Decision Rules:**
→ Use `action: "final_answer"` when:
  - generally you need to verify (e.g., make sure the formal name for Alibaba is "Alibaba Group Limited" in English and "阿里巴巴集团控股有限公司" in Chinese, rather than "阿里巴巴" or "阿里巴巴公司". "Bloomberg L.P." in English and "彭博社" in Chinese)

→ Use `action: "need_verification"` when:
  - The answer is an organization, person, company, etc, you need to confirm the standard name, official spelling, or full legal name.
→ When the question is in English and the given answer is in Chinese:
   - Your verification goal should be: "Confirm the official English name of [entity]"
   - Your search query should be in English (or bilingual if needed)
→ When the question is in Chinese and the given answer is in English:   
   - Your verification goal should be: "确认[entity]的标准中文名" 
   - Your search query should be in Chinese (or bilingual if needed)
  
**Query Guidelines (when action = "need_verification"):**
For Original Question in Chinese:
    - `goal`: Clearly state what standard Chinese name you need to confirm (max 1 sentence)
    - `query`: A plain-language search query in **Chinese** to find the official name on Chinese Wikipedia, Baidu Baike, or official Chinese websites (e.g., "埃隆·马斯克")
For Other's language:
    - `goal`:  Confirm the standard English name, official spelling, or full legal name, especially in these case the candidate answer is in Chinese.
    - `query`: English search keywords optimized for once wikipedia search  (e.g., "Elon Reeve Musk").

**NEVER Output:**
- English names in Chinese questions without explicit permission
- Romanizations like "qianlong", "beijing", or "Elon Musk" when a standard Chinese form exists (e.g., "埃隆·马斯克")
- Approximate or invented translations

**Examples:**

Answer Ready:
{{
    "think": "Question is in Chinese with no English request. Entity 乾隆 has standard Chinese form already verified.",
    "action": "final_answer",
    "answer": "乾隆"
}}

Need Verification:
{{
    "think": "Question is in Chinese and candidate answer is in English but need to verify official Chinese name via Chinese sources.",
    "action": "need_verification",
    "goal": "Confirm the official Chinese name of Sam Altman",
    "query": "Sam Altman 中文名 百度百科"
}}


---

**Now perform the check:**

Original Question: {question}

Candidate Answer: {candidate_answer}

**Remember: Valid JSON only, NO markdown, EXACTLY one action type!**
"""

multi_react_prompt_template = """You are an expert AI assistant solving complex factual questions (GAIA-style benchmark).

**CRITICAL OUTPUT FORMAT RULES (VIOLATION = FAILURE):**
1. ✅ Output MUST be valid JSON object with exact keys: ["think", "goal", "query", "wiki", "answer"]
2. ✅ "think": max 3 sentences, ≤800 words, brief reasoning
3. ✅ "goal": single sentence, ≤100 words, state search purpose
4. ✅ "query": a list of up to 2 plain text search query in appropriate language used for web search
5. ✅ "wiki": OPTIONAL, only for verifying specific entities (person/org/product/term) which may appear in the final answer
6. ✅ "answer": OPTIONAL, ONLY output after fully verified
7. ❌ NO additional keys beyond the 5 specified
8. ❌ NO markdown code blocks (no ```json ... ```)
9. ❌ NO nested JSON or complex structures

**JSON Output Structure (STRICT ORDER):**
{{
  "think": "Brief reasoning, max 3 sentences, less than 800 words",
  "goal": "Single sentence, less than 100 words",
  "query": ["Search query1 in appropriate language","Optional search query2 in appropriate language if necessary"]
  "wiki": "Optional: exact entity name",
  "answer": "Optional: ONLY when fully verified"
}}

**Field Rules**
- "think": 
    • First, remind ALL hard constraints from the original question (e.g., time, location, event).
    • Then, if proposing a candidate, explicitly state WHICH key claims remain unverified.
    • If prior searches confirm a candidate can not meet a critical constraint, ABANDON that candidate and pivot to a NEW distinguishing clue.
    • If a candidate appears plausible early in the think process, explicitly verify its key claims rather than discarding it prematurely.
- "goal": ONE sentence purpose (e.g., "Confirm the official Chinese name of WeChat's founder")
- "query":
     * Use MOST EFFECTIVE language (Chinese for China topics, English otherwise)
     * If the goal requires finding multiple independent facts (e.g., Person A AND Person B), output a LIST of at most 2 queries like ["query for A", "query for B"].
- "wiki": Only for verifying specific entities that may appear in final answer
- "answer": ONLY when factually correct, all constraints satisfied, official names have been checked

**Search Guidelines**
- Default in English
- If Chinese question and Chinese topic (e.g., Chinese historical figures, actors, schoolars, films, cities, organizations, etc.) → can try Chinese query
- If last 2 searches failed:
  - switch query to Chinese if the Original question is in Chinese, otherwise try to rethink assumption and find the most distinguishable clues for searching

**Answer Rules**
- ONLY include "answer" key when:
  ✓ Factually correct
  ✓ All constraints satisfied (time, location, format, language)
  ✓ Official names cross-checked via Wikipedia/baike (which means you can not return answer in the first step, at least one search must be conducted)
  ✓ Language/format matches requirement
- Otherwise: set "answer" to empty "" or omit the key
- Note: If the question includes **examples in a specific language or explicitly asks for English**, follow that. Otherwise, use the same language in "answer" field as the question.

**Examples**

✅ CORRECT (no final answer):
{{
  "think": "I have two candidate names but need to verify which is official. Last search returned irrelevant results, so I'll try a Chinese query.",
  "goal": "Verify the official Chinese name of Tencent's COO",
  "query": ["腾讯 首席运营官 任宇昕"],
  "wiki": "任宇昕",
  "answer": ""
}}

✅ CORRECT (Multiple Queries):
{{
  "think": "The problem requires finding a European writer born in 1920 and an Asian writer who won a prize in 1968. These are independent facts. I will search for both simultaneously to save time.",
  "goal": "Identify the European writer born in 1920 and the Asian writer winning the 1968 prize",
  "query": ["European writer born 1920 famous novels", "Asian writer winner literature prize 1968"],
  "wiki": "",
  "answer": ""
}}

✅ CORRECT (with final answer):
{{
  "think": "All constraints verified. Official name confirmed via Wikipedia cross-check.",
  "answer": "任宇昕"
}}

Now solve this task:
Original Question: {question}
Current Context and core findings log:
{context}
Remember: Output ONLY raw JSON, NO markdown, NO extra keys!
"""

def detect_query_language(query: str, chinese_threshold: float = 0.5) -> str:

    if not query.strip():
        return 'en'  # default fallback
    
    # Remove spaces, digits, and common punctuation
    cleaned = re.sub(r'[^\w\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', '', query)
    if not cleaned:
        return 'en'
    
    # Count Chinese characters (including CJK extensions)
    chinese_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', cleaned)
    chinese_ratio = len(chinese_chars) / len(cleaned)
    
    return 'zh' if chinese_ratio >= chinese_threshold else 'en'


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
        
    chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')
    english_pattern = re.compile(r'[a-zA-Z0-9]+')
    
    zh_count = len(chinese_pattern.findall(text))
    en_count = len(english_pattern.findall(text))
    
    # 粗略估算
    return int(zh_count * 1.5 + en_count * 1.3)

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    """
    截取字符串的前 max_tokens 个 token。
    使用二分查找法快速定位截断点，避免逐字遍历的性能问题。
    
    Args:
        text: 原始字符串
        max_tokens: 目标最大 Token 数
    
    Returns:
        截取后的字符串 (长度 <= max_tokens 的估算值)
    """
    if not text:
        return ""
    
    # 1. 快速检查：如果全文都小于限制，直接返回
    if estimate_token_count(text) <= max_tokens:
        return text
    
    # 2. 二分查找截断点
    # 左边界 0, 右边界为文本长度 (字符数)
    left, right = 0, len(text)
    best_cut = 0
    
    while left <= right:
        mid = (left + right) // 2
        # 截取测试片段
        chunk = text[:mid]
        tokens = estimate_token_count(chunk)
        
        if tokens <= max_tokens:
            best_cut = mid  # 记录当前合法的最大位置
            left = mid + 1  # 尝试更长的片段
        else:
            right = mid - 1 # 片段太长，缩短
            
    # 3. 返回截取结果
    # 额外优化：避免截断在英文单词中间或中文词中间（可选，这里简单按字符截断）
    # 如果需要更完美的截断（不切断单词），可以在 best_cut 附近微调，但通常没必要
    return text[:best_cut]

def tail_by_tokens(text: str, max_tokens: int) -> str:
    """
    截取字符串的**最后** max_tokens 个 token。
    相当于获取 s[-N:] 的 Token 版本。
    
    Args:
        text: 原始字符串
        max_tokens: 目标最大 Token 数
    
    Returns:
        截取后的尾部字符串
    """
    if not text:
        return ""
    
    total_tokens = estimate_token_count(text)
    
    # 1. 如果全文都小于限制，直接返回
    if total_tokens <= max_tokens:
        return text
    
    # 2. 二分查找截断点 (start_index)
    # 我们要找一个 start_index，使得 text[start_index:] 的 token 数 <= max_tokens
    # 且尽可能接近 max_tokens (即 start_index 尽可能小)
    
    left, right = 0, len(text)
    best_start = len(text) # 默认最坏情况取空串（实际上循环会更新）
    
    while left <= right:
        mid = (left + right) // 2
        chunk = text[mid:] # 截取从 mid 到结尾
        tokens = estimate_token_count(chunk)
        
        if tokens <= max_tokens:
            # 当前长度符合要求，尝试让 start_index 更小（保留更多内容）
            best_start = mid
            right = mid - 1 
        else:
            # 当前片段太长（说明 mid 太靠前了），需要向后移（丢弃更多头部）
            left = mid + 1
            
    return text[best_start:]


def get_model_output(text_in,temperature=0,choosed_model=base_model,timeout=45, max_retries=2):
    global TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS, TOTAL_COST_ESTIMATE
    attemp_times = 0
    client = OpenAI(
        api_key=QWEN_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    while attemp_times < max_retries:
        try:
            completion = client.chat.completions.create(
                model=choosed_model,
                messages=[
                    {"role": "user", "content": text_in},
                ],
                timeout=timeout+(attemp_times)*5,
                temperature=temperature,
                extra_body={"enable_thinking": False},
            )
            # 提取 token 信息
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            
            # 更新全局计数
            TOTAL_INPUT_TOKENS += prompt_tokens
            TOTAL_OUTPUT_TOKENS += completion_tokens
            
            # 估算费用（可选）
            price = MODEL_PRICING.get(choosed_model, {"input": 0, "output": 0})
            cost = (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000
            TOTAL_COST_ESTIMATE += cost
            
            # 打印本次 + 累计统计
            print_and_log(
                f"[Token Usage] Model: {choosed_model} | "
                f"Input: {prompt_tokens}, Output: {completion_tokens} | "
                f"Total Input: {TOTAL_INPUT_TOKENS}, Total Output: {TOTAL_OUTPUT_TOKENS} | "
                f"Est. Cost: ¥{TOTAL_COST_ESTIMATE:.4f}"
            )
            return completion.choices[0].message.content
        except Exception as e:
            print_and_log(f"Error: {e}")
        attemp_times += 1

    return None


def get_max_model_output(text_in,temperature=0,choosed_model=max_model,timeout=45):
    global TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS, TOTAL_COST_ESTIMATE
    attemp_times = 0
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=QWEN_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    while attemp_times < 3:
        try:
            completion = client.chat.completions.create(
                # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                model=choosed_model,
                messages=[
                    {"role": "user", "content": text_in},
                ],
                temperature=temperature,
                timeout=timeout+(attemp_times*5),
                extra_body={"enable_thinking": False},
            )
            # 提取 token 信息
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            
            # 更新全局计数
            TOTAL_INPUT_TOKENS += prompt_tokens
            TOTAL_OUTPUT_TOKENS += completion_tokens
            
            # 估算费用（可选）
            price = MODEL_PRICING.get(choosed_model, {"input": 0, "output": 0})
            cost = (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000
            TOTAL_COST_ESTIMATE += cost
            
            # 打印本次 + 累计统计
            print_and_log(
                f"[Token Usage] Model: {choosed_model} | "
                f"Input: {prompt_tokens}, Output: {completion_tokens} | "
                f"Total Input: {TOTAL_INPUT_TOKENS}, Total Output: {TOTAL_OUTPUT_TOKENS} | "
                f"Est. Cost: ¥{TOTAL_COST_ESTIMATE:.4f}"
            )
            return completion.choices[0].message.content
        except Exception as e:
            print_and_log(f"Error: {e}")
        attemp_times += 1

    return None


def get_quick_output(text_in, temperature=0,choosed_model=flash_model, timeout=30, max_retries=3):
    global TOTAL_INPUT_TOKENS, TOTAL_OUTPUT_TOKENS, TOTAL_COST_ESTIMATE
    attemp_times = 0
    client = OpenAI(
        api_key=QWEN_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    while attemp_times < max_retries:
        try:
            completion = client.chat.completions.create(
                model=choosed_model,
                messages=[
                    {"role": "user", "content": text_in},
                ],
                timeout=timeout+attemp_times*1,
                temperature=temperature,
                extra_body={"enable_thinking": False},
            )
            # 提取 token 信息
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            
            # 更新全局计数
            TOTAL_INPUT_TOKENS += prompt_tokens
            TOTAL_OUTPUT_TOKENS += completion_tokens
            
            # 估算费用（可选）
            price = MODEL_PRICING.get(choosed_model, {"input": 0, "output": 0})
            cost = (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000
            TOTAL_COST_ESTIMATE += cost
            
            # 打印本次 + 累计统计
            print_and_log(
                f"[Token Usage] Model: {choosed_model}| "
                f"Input: {prompt_tokens}, Output: {completion_tokens} | "
                f"Total Input: {TOTAL_INPUT_TOKENS}, Total Output: {TOTAL_OUTPUT_TOKENS} | "
                f"Est. Cost: ¥{TOTAL_COST_ESTIMATE:.4f}"
            )
            return completion.choices[0].message.content
        except Exception as e:
            print_and_log(f"Error: {e}")
        attemp_times += 1

    return None



def condense_context(context,max_len=6000):
    # 压缩长度
    if estimate_token_count(context)>3000:
        history_dense = get_quick_output(compress_template.format(full_text=context),timeout=35,choosed_model=flash_model,max_retries=1)
    else:
        history_dense = get_model_output(compress_template.format(full_text=context),timeout=30,choosed_model=plus_model,max_retries=1)
        if history_dense is None:
            history_dense = get_quick_output(compress_template.format(full_text=context),timeout=30,choosed_model=flash_model,max_retries=1)
        
    if history_dense is None and len(context)>max_len:
        context = truncate_by_tokens(history_dense, 3000) + tail_by_tokens(history_dense,-3000)
    return context

def condense_think_context(think_context, max_len=6000, think_compress_threshold=800):
    """
    think_compress_threshold: 思考内容超过多少字符才触发压缩 (约 300-500 中文词)
    """
    
    # 1. 正则匹配 <think>...</think> 部分 (re.DOTALL 让 . 能匹配换行符)
    pattern = r'(<think>)(.*?)(</think>)'
    match = re.search(pattern, think_context, re.DOTALL | re.IGNORECASE)
    
    if match:
        start_tag = match.group(1)   # <think>
        thinking_content = match.group(2) # 中间的内容
        end_tag = match.group(3)     #</think>
        
        # 2. 判断是否需要压缩
        if estimate_token_count(thinking_content) <= think_compress_threshold:
            # 如果很短，直接保留原样，不做任何处理
            return think_context
        
        # 3. 尝试调用模型压缩思考部分
        try:
            compressed_thinking = get_quick_output(think_compress_template.format(full_text=thinking_content),timeout=20,choosed_model=flash_model,max_retries=1)
            
            # 如果模型压缩成功，替换原内容
            if compressed_thinking and estimate_token_count(compressed_thinking) > 0 and estimate_token_count(compressed_thinking) < estimate_token_count(thinking_content):
                # 重新组装：前缀 + 标签头 + 压缩后的内容 + 标签尾 + 后缀
                new_think_context = think_context[:match.start()] + \
                                    start_tag + compressed_thinking + end_tag + \
                                    think_context[match.end():]
                return new_think_context
            else:
                # 模型返回为空， fallback 到机械截断思考部分
                compressed_thinking = truncate_by_tokens(thinking_content,think_compress_threshold//2) + "\n...[omitted]...\n" + tail_by_tokens(thinking_content,think_compress_threshold//2)
                
        except Exception as e:
            print_and_log(f"[condense_think_context] Error compressing thought: {e}")
            # 出错时机械截断思考部分
            compressed_thinking = truncate_by_tokens(thinking_content,think_compress_threshold//2) + "\n...[error omitted]...\n" + tail_by_tokens(thinking_content,think_compress_threshold//2)

        # 应用机械截断的结果
        new_think_context = think_context[:match.start()] + \
                            start_tag + compressed_thinking + end_tag + \
                            think_context[match.end():]
        return new_think_context

    else:
        # 4. 如果没有找到 <think> 标签，但总长度依然超标，则对整个文本进行机械截断
        if estimate_token_count(think_context) > max_len:
            return truncate_by_tokens(think_context,max_len//2) + "\n...[truncated]...\n" + tail_by_tokens(think_context,max_len//2)
        return think_context
    

def clean_text(text: str) -> str:
    """清洗文本：去除控制字符、多余空白，防止 embedding 失败"""
    if not isinstance(text, str):
        text = str(text)
    # 移除 ASCII 控制字符（保留换行/空格等可读空白）
    cleaned = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\n\t ')
    # 合并连续空白（可选）
    # cleaned = ' '.join(cleaned.split())
    return cleaned[:MAX_TEXT_LENGTH]


def is_valid_embedding(vec, expected_dim=EMBEDDING_DIM):
    """校验 embedding 是否有效"""
    if vec is None:
        return False
    if not isinstance(vec, np.ndarray):
        vec = np.array(vec)
    if vec.shape != (expected_dim,):
        return False
    if np.any(np.isnan(vec)):
        return False
    if np.allclose(vec, 0.0, atol=1e-6):  # 全零向量视为无效
        return False
    return True


def get_text_embedding(text: str, max_retries=MAX_RETRIES):
    """
    获取文本 embedding，带重试和校验。
    成功返回 shape=(1024,) 的 np.ndarray；失败返回 None。
    """
    if not text or not text.strip():
        return None

    text = clean_text(text)

    for attempt in range(max_retries):
        try:
            client = OpenAI(
                api_key=QWEN_KEY,  
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            completion = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            embedding = json.loads(completion.model_dump_json())['data'][0]["embedding"]
            vec = np.array(embedding, dtype=np.float32)
            if is_valid_embedding(vec):
                return vec
        except Exception as e:
            time.sleep(2 * attempt)
    return None



def iqs_search(query,top_k=10, type="LiteAdvanced"): # Generic LiteAdvanced
    url = "https://cloud-iqs.aliyuncs.com/search/unified"

    headers = {
        "Authorization": f"Bearer {IQS_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "query": query,
        "engineType": type,
        "contents": {
            "mainText": True,
            "markdownText": True,
            "richMainBody": True,
            "summary": False,
            "rerankScore": True
        }
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        result_json = response.json()
        return result_json['pageItems']
    except Exception as e:
        print_and_log(f"Error occurred while searching: {e}")
        return []


def jina_read_page(target_url,timeout=10000,enbable_browser=True):
    url = "https://www.searchcans.com/api/url"
    headers = {
        "Authorization": f"Bearer {search_scan_key}",
        "Content-Type": "application/json"
    }
    data = {
        "s": target_url,
        "t": "url",
        "w": 3000,   # Wait 3s for JavaScript rendering
        "d": timeout,  # 20s timeout for complex pages
        "b": enbable_browser    # Enable browser mode (recommended)
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        result = response.json()
        if result.get('code') == 0:
            return result.get('data', {}).get('markdown', '')
        else:
            return None
    except Exception as e:
        print_and_log(f"Error jina fetching {url}: {e}")
        return None


def embed_chunk(chunk):
    return get_text_embedding(chunk)


def filter_chunks_with_head_tail(chunks, sub_question, query, min_keyword_len=2, max_keep=8):
    """
    过滤 Chunks，强制保留第一个和最后一个，并基于关键词筛选中间部分。
    
    Args:
        chunks: 切分好的文本列表
        sub_question: 子问题
        query: 原始查询
        min_keyword_len: 关键词最小长度 (默认 2，以支持"成龙"等双字词)
        max_keep: 最终保留的最大数量
    
    Returns:
        filtered_chunks: 过滤后的列表
    """
    if not chunks:
        return []

    # 1. 提取关键词 (转小写，去引号)
    # 注意：中文转小写无影响，主要是为了匹配英文
    text_to_process = (sub_question.replace('"', '') + " " + query.replace('"', '')).lower()
    # 使用正则提取 alphanumeric 和中文字符，避免纯标点干扰
    # \u4e00-\u9fff 匹配常用汉字
    keywords = set(re.findall(r'[a-z0-9\u4e00-\u9fff]+', text_to_process))
    
    # 过滤掉太短的无意义词 (如 "a", "i", "的" 如果单独出现)
    # 这里将阈值改为你要求的逻辑：>= 2 (支持双字中文)
    valid_keywords = {kw for kw in keywords if len(kw) >= min_keyword_len}

    if not valid_keywords:
        # 如果没有有效关键词，保守策略：直接返回首尾或全部（如果很少）
        if len(chunks) <= max_keep:
            return chunks
        else:
            # 没关键词时，只取首尾各一半？或者直接取前几个？
            # 这里选择：只取首尾，防止中间噪音
            return [chunks[0]] + ([chunks[-1]] if len(chunks) > 1 else [])

    filtered_chunks = []
    
    # 2. 强制保留第一个 Chunk
    filtered_chunks.append(chunks[0])
    
    # 3. 强制保留最后一个 Chunk (如果不止一个)
    if len(chunks) > 1:
        filtered_chunks.append(chunks[-1])

    # 4. 筛选中间的 Chunks (索引 1 到 -2)
    # 只有当总 chunk 数 > 2 时才需要遍历中间
    if len(chunks) > 2:
        for i in range(1, len(chunks) - 1):
            chunk = chunks[i]
            chunk_lower = chunk.lower()
            
            # 命中任意一个有效关键词即保留
            if any(kw in chunk_lower for kw in valid_keywords if len(kw)>=2):
                filtered_chunks.append(chunk)

    # 5. 数量控制与去重
    # 如果原始总数很少，直接返回所有原始 chunks (优先保证信息量)
    if len(chunks) <= max_keep:
        return chunks
    
    # 如果过滤后太多，截断 (保持首尾优先)
    if len(filtered_chunks) > max_keep:
        # 取前一半和后一半
        half = max_keep // 2
        head = filtered_chunks[:half]
        tail = filtered_chunks[-half:]
        
        # 合并并去重 (防止 head 和 tail 重叠)
        seen_ids = set()
        result = []
        for item in head + tail:
            # 使用 id 或内容哈希去重，这里用 id 最快 (假设 chunk 对象引用不变)
            if id(item) not in seen_ids:
                result.append(item)
                seen_ids.add(id(item))
        return result

    return filtered_chunks


def get_webpage_content(question_idx,url, if_text_only=True, timeout=10):
    global web_cache
    """
    静态爬虫获取网页内容（不依赖浏览器渲染）
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    }
    if web_cache.get(question_idx):
        if web_cache[question_idx].get(url):
            if len(web_cache[question_idx].get(url))>100:
                print_and_log(f"Cache hit for URL: {url}")
                return None, web_cache[question_idx].get(url)
    try:
        

        response = requests.get(url, headers=headers, timeout=timeout, verify=False)
        response.encoding = response.apparent_encoding
        html = response.text
    except Exception as e:
        print_and_log(f"Error fetching {url}: {e}")
        return None, None
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # 移除不需要的标签
        excluded_tags = ["nav", "footer", "aside", "header", "script", "style", "iframe", "meta"]
        for tag in excluded_tags:
            for element in soup.find_all(tag):
                element.decompose()
        
        # 提取标题
        title = soup.title.string if soup.title else ""
        
        # 提取正文内容
        if if_text_only:
            # 只提取文本
            text_parts = []
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th']):
                text = element.get_text(strip=True)
                if len(text) > 5:  # 过滤太短的文本
                    text_parts.append(text)
            webpage_text = '\n\n'.join(text_parts)
        else:
            # 保留链接和结构
            text_parts = []
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'a']):
                text = element.get_text(strip=True)
                if len(text) > 5:
                    text_parts.append(text)
            webpage_text = '\n\n'.join(text_parts)
        
        # 清理文本
        cleaned_text = webpage_text.replace("undefined", "") if webpage_text else ""
        cleaned_text = re.sub(r'(\n\s*){3,}', '\n\n', cleaned_text)
        cleaned_text = re.sub(r'[\r\t]', '', cleaned_text)
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        cleaned_text = re.sub(r'^\s+|\s+$', '', cleaned_text, flags=re.MULTILINE)
        
        # 模拟返回结构（兼容原有代码）
        result = {
            'markdown': type('obj', (object,), {'fit_markdown': cleaned_text})(),
            'url': url,
            'title': title,
            'status_code': response.status_code
        }
        # 设置cache
        if web_cache.get(question_idx):
            if len(cleaned_text.strip())>80:
                web_cache[question_idx][url] = cleaned_text.strip()
        else:
            web_cache[question_idx] = {}
            if len(cleaned_text.strip())>80:
                web_cache[question_idx][url] = cleaned_text.strip()
        return result, cleaned_text.strip()
    
    except Exception as e:
        print_and_log(f"Error parsing {url}: {e}")
        return None, None


def simple_chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    if not text:
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks


def web_visit(question_idx, main_question, sub_question, query, url, raw_content):
    web_summary = ""
    try:
        if '.pdf' in url or 'zhidao.baidu.com' in url:
            web_page_text = None
        elif 'wiki' in url or len(raw_content)<500: # 如果是维基百科或者太短了
            web_page_markdown, web_page_text = get_webpage_content(question_idx, url, if_text_only=True,timeout=10)
        else:
            web_page_text = None
    except Exception as e:
        print_and_log(f"[web_visit]Error fetching webpage content: {e}")
    
    if DEBUG_MODE:
        print_and_log("[web_visit]:", url)

    if  web_page_text is None or len(web_page_text.strip())<len(raw_content.strip()):
        web_page_text = raw_content
        if DEBUG_MODE:
            print_and_log("[Failed in][web_visit]:", url)

    # 1.chunk切分，可能需要有重叠的
    chunks = simple_chunk_text(web_page_text, chunk_size=2000, overlap=100) # 2000 100
    if not chunks or len(chunks) == 0:
        print_and_log("[chunk error] NO CHUNK!")
        return truncate_by_tokens(raw_content, 750), truncate_by_tokens(raw_content, 750)

    chunks = filter_chunks_with_head_tail(chunks,sub_question, query, min_keyword_len=2, max_keep=8)
    if not chunks or len(chunks) == 0:
        print_and_log("[chunk error] NO CHUNK!")
        return truncate_by_tokens(raw_content, 750), truncate_by_tokens(raw_content, 750)
    else:
        # 2.向量化 设定阈值（可调）+ top-k 约束
        threshold = 0.7
        top_k = 4
        min_keep = 2
        try:
            query_vec = get_text_embedding(sub_question + query)  # shape: (1024,)
            with ThreadPoolExecutor(max_workers=10) as executor:
                chunk_vecs = list(executor.map(embed_chunk, chunks))  # 保持 chunks 的原始顺序！
            
            # 转为 numpy array 便于计算
            query_vec = np.array(query_vec).reshape(1, -1)  # (1, 1024)
            chunk_vecs = np.array(chunk_vecs)               # (n_chunks, 1024)

            similarities = cosine_similarity(query_vec, chunk_vecs).flatten()  # (n_chunks,)
            
            # 获取高于阈值的索引，或 fallback 到 top-k
            high_sim_indices = np.where(similarities >= threshold)[0]
        
            if len(high_sim_indices) == 0:
                # 若无达标，取 top-k
                selected_indices = np.argsort(-similarities)[:min_keep]
            else:
                # 限制最多 top_k 个
                selected_indices = high_sim_indices[:top_k]

            # 至少保留 min_keep 个（即使低于阈值）
            if len(selected_indices) < min_keep:
                top_all = np.argsort(-similarities)[:min_keep]
                selected_indices = np.unique(np.concatenate([selected_indices, top_all]))

            n_chunks = len(chunks)
            if n_chunks > 0:
                first_idx = 0
                last_idx = n_chunks - 1
                
                # 将首尾索引加入列表
                selected_indices = np.unique(np.concatenate([selected_indices, [first_idx, last_idx]]))

            # 去重并按原文顺序排序
            selected_indices = sorted(set(selected_indices.tolist()))

            # 提取选中的 chunks（按原文顺序）
            chosen_chunks = [chunks[i] for i in selected_indices]
            relevant_docs = "\n\n---\n\n".join(chosen_chunks)
            
            evaluate_search_result_prompt = evaluate_search_result_template.format(
                main_question=main_question,
                sub_question=sub_question,
                search_query=query,
                content=relevant_docs,
                URL=url
            )

            web_summary = get_model_output(evaluate_search_result_prompt,choosed_model=plus_model, timeout=15, max_retries=1)
            if DEBUG_MODE:
                print_and_log("Web visit search result response is:", web_summary)
            if web_summary is None or len(web_summary.strip()) == 0:
                web_summary = truncate_by_tokens(raw_content, 750)
        except Exception as e:
            print_and_log(f"[download_and_read_html] Error during embedding: {e}")
            web_summary = truncate_by_tokens(raw_content, 750)
            web_page_text = truncate_by_tokens(raw_content,3500)

    return web_summary,web_page_text


def search_scan(query, engine='google'):
    attempt = 0
    MAX_RETRIES = 2
    result = {}
    while attempt < MAX_RETRIES:
        url = "https://www.searchcans.com/api/search"
        headers = {
            "Authorization": f"Bearer {search_scan_key}",
            "Content-Type": "application/json"
        }
        data = {
            "s": query,
            "t": engine,  # or "bing"
            "p": 1,
            "d": 20000,  # 30s timeout (production)
            "w": 5000    # 5s wait time
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=20)
            result = response.json()
        except Exception as e:
            print_and_log(f"Error occurred while searching: {e}")
            result = {"code": -1}

        if result.get('code',-1) == 0:
            break
        else:
            attempt += 1
            print_and_log(f"search_scan retry")
            time.sleep(2 * attempt)
    return result.get('data', [])


def get_search_scan_result(original_question, question_idx, current_subq, query):
    # 为title link snippet mainText rerankScore的格式
    try:
        api_result = search_scan(query)
        if len(api_result)==0:
            print_and_log("search scan No valid search results found or API error.")
            return []
        if len(api_result)>10:
            api_result = api_result[:10]  # 只保留前10条结果
        google_response = {"results": api_result}
        
    except Exception as e:
        print_and_log(f"Error occurred while parsing search response: {e}")
        return []

    search_result_list = []
    
    def process_single_result(index, result_item):
        url = result_item.get('url', '')
        content_snippet = result_item.get('content', '')
        
        try:
            web_summary, raw_content = web_visit(question_idx,original_question, current_subq, query, url, content_snippet)
            
            return {
                'question_idx': question_idx,
                'sub_question': current_subq,
                'query': query,
                'url': url,
                'title': result_item.get('title', ''),
                'content': web_summary,
                'raw_content': raw_content,
                'score': 1
            }
        except Exception as e:
            print_and_log(f"Error visiting URL {url}: {e}")
            return None
        
    try:
        with ThreadPoolExecutor(max_workers=10) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(process_single_result, i, item): i 
                for i, item in enumerate(google_response['results'])
            }

            for future in as_completed(future_to_index):
                result = future.result()
                if result is not None:
                    search_result_list.append(result)
                    
    except Exception as e:
        print_and_log(f"Thread pool error in search cans: {e}")
    
    print_and_log(f"search scan api google获取到的搜索结果数量: {len(search_result_list)}")
    return search_result_list


def serpapi_search(query,lang='en'):
    s = []
    try:
        s = serpapi.search(q=query, engine="google", location="Austin, Texas", hl=lang, gl="us", safe="off", api_key=serpapi_key)
    except Exception as e:
        return []
    return s['organic_results']


def get_serpapi_result(question_idx,current_subq, query,lang='en'):
    global google_cache
    try:
        if question_idx in google_cache:
            if google_cache[question_idx].get(query):
                print_and_log(f"Google Cache hit for query: {query}")
                return google_cache[question_idx][query]
        google_response = {"results": serpapi_search(query,lang)}
    except Exception as e:
        google_response = {"results": []}
        print_and_log(f"Error occurred while searching: {e}")
    if len(google_response['results']) == 0:
        print_and_log("serpapi No search results found.Error!")
    
    search_result_list = []
    for i in range(len(google_response['results'])):
        item = {}
        item['question_idx'] = question_idx # 问题的id
        item['sub_question'] = current_subq # 当前的子问题
        item['query'] = query
        item['url'] = google_response['results'][i]['link']
        item['title'] = google_response['results'][i]['title']
        item['content'] = google_response['results'][i]['snippet']
        item['raw_content'] = google_response['results'][i]['snippet']
        item['score'] = 1
        search_result_list.append(item.copy())
    print_and_log(f"serpapi google获取到的搜索结果数量: {len(search_result_list)}")
    try:
        if len(search_result_list)>0:
            if question_idx in google_cache:
                google_cache[question_idx][query] = search_result_list
            else:
                google_cache[question_idx] = {query: search_result_list}
    except Exception as e:
        pass
    return search_result_list


def get_iqs_search_result(question_idx,current_subq, query,top_k=10):
    global iqs_cache
    
    try:
        if question_idx in iqs_cache:
            if iqs_cache[question_idx].get(query):
                print_and_log(f"IQS Cache hit for query: {query}")
                return iqs_cache[question_idx][query]
            
        iqs_response = {"results": iqs_search(query, top_k)}
    except Exception as e:
        iqs_response = {"results": []}
        print_and_log(f"Error occurred while searching: {e}")
    if len(iqs_response['results']) == 0:
        print_and_log("IQS No search results found.Error!")

    search_result_list = []
    for i in range(min(top_k, len(iqs_response['results']))):
        item = {}
        item['question_idx'] = question_idx # 问题的id
        item['sub_question'] = current_subq # 当前的子问题
        item['query'] = query
        item['url'] = iqs_response['results'][i]['link']
        item['title'] = iqs_response['results'][i]['title']
        item['content'] = iqs_response['results'][i]['snippet']
        item['raw_content'] = iqs_response['results'][i]['mainText']
        item['score'] = iqs_response['results'][i]['rerankScore']
        search_result_list.append(item.copy())
    print_and_log(f"获取到的iqs搜索结果数量: {len(search_result_list)}")
    try:
        if len(search_result_list)>0:
            if question_idx in iqs_cache:
                iqs_cache[question_idx][query] = search_result_list
            else:
                iqs_cache[question_idx] = {query: search_result_list}
    except Exception as e:
        pass
    return search_result_list



# 全局变量：存储本次调用结束的时间
_last_call_end_time = time.time()

def search_wiki(query, lang='en', load_max_docs=5, doc_content_chars_max=10000, timeout=20):
    """
    Search Wikipedia with a timeout.
    
    Args:
        query: The search term.
        lang: Language code (e.g., 'en', 'zh').
        load_max_docs: Max number of documents to load.
        doc_content_chars_max: Max characters per doc.
        timeout: Timeout in seconds (default: 20).
    
    Returns:
        List of Document objects if successful and >5s; empty list if <5s or error.
    """
    global _last_call_end_time
    
    def _do_search():
        try:
            return WikipediaLoader(
                query=query,
                lang=lang,
                load_max_docs=load_max_docs,
                doc_content_chars_max=doc_content_chars_max
            ).load()
        except Exception as e:
            return []

    with ThreadPoolExecutor(max_workers=1) as executor:
        # if (time.time() - _last_call_end_time) < 5: # 如果上次调用结束时间距离现在不到5秒，直接返回
        #     print_and_log("Wikipedia search skipped due to recent call (within 5 seconds).")
        #     return []
        future = executor.submit(_do_search)
        try:
            docs = future.result(timeout=timeout)
            # end_time = time.time()
            # _last_call_end_time = end_time  # 更新全局变量
            return docs if docs is not None else []
        except Exception as e:
            print_and_log(f"Error occurred during Wikipedia search: {e}")
            return []



# 如果需要获取上次调用结束时间，可以添加这个辅助函数
def get_last_call_end_time():
    """获取上次调用结束的时间戳"""
    return _last_call_end_time



def my_get_wiki(question_idx, query: str, lang: str = 'en', load_max_docs: int = 5) -> List[Any]:
    global wiki_query_cache, web_cache
    results = []
    
    # 1. 设置语言
    wikipedia.set_lang(lang)
    
    search_titles = []
    
    # --- 第一步：执行搜索 (带 7 秒超时) ---
    try:
        def do_search():
            return wikipedia.search(query, results=load_max_docs)
            
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(do_search)
            # 设置 7 秒超时
            search_titles = future.result(timeout=7)
            
    except FuturesTimeoutError:
        print_and_log(f"Timeout: Wikipedia search for '{query}' exceeded 7 seconds. Returning empty.")
        return []
    except Exception as e:
        print_and_log(f"Error during Wikipedia search for query '{query}': {e}")
        return []

    if not search_titles or len(search_titles) == 0:
        return []

    # --- 第二步：循环获取详情 (每个条目 7 秒超时) ---
    
    # 定义内部函数用于执行单个条目的抓取逻辑
    def fetch_single_doc(title):
        doc = type('Document', (object,), {})()
        
        # --- 缓存检查逻辑 (保持原样) ---
        if wiki_query_cache.get(question_idx) is not None:
            if wiki_query_cache[question_idx].get(title) is not None:
                page_dict = wiki_query_cache[question_idx][title]
                doc.metadata = {
                    'source': page_dict.get('url', ""),
                    'title': page_dict.get('title', ""),
                    'summary': page_dict.get('summary', ""),
                    'content': page_dict.get('content', "")
                }
                return doc # 返回 doc
        
        # --- 核心抓取逻辑 ---
        try:
            #page = wikipedia.page(title)
            def do_page_fetch(t):
                return wikipedia.page(t)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(do_page_fetch, title)
                page = future.result(timeout=7)
            
            doc.metadata = {
                'source': page.url if hasattr(page, 'url') and page.url else "",
                'title': page.title if hasattr(page, 'title') and page.title else "",
                'summary': page.summary if hasattr(page, 'summary') and page.summary else "",
                'content': page.content if hasattr(page, 'content') and page.content else ""
            }
            
            # 缓存写入逻辑
            if web_cache.get(question_idx):
                if len(doc.metadata['content']) > 100:
                    web_cache[question_idx][doc.metadata['source']] = doc.metadata['content']
            else:
                web_cache[question_idx] = {}
                if len(doc.metadata['content']) > 100:
                    web_cache[question_idx][doc.metadata['source']] = doc.metadata['content']
            
            return doc

        except wikipedia.exceptions.DisambiguationError as e:
            # 消歧义处理逻辑
            if e.options:
                try:
                    alt_title = e.options[0]
                    # 再次检查缓存 (针对消歧义后的标题)
                    if wiki_query_cache.get(question_idx) is not None:
                        if wiki_query_cache[question_idx].get(alt_title) is not None:
                            page_dict = wiki_query_cache[question_idx][alt_title]
                            doc.metadata = {
                                'source': page_dict.get('url', ""),
                                'title': page_dict.get('title', ""),
                                'summary': page_dict.get('summary', ""),
                                'content': page_dict.get('content', "")
                            }
                            return doc
                    
                    page = wikipedia.page(alt_title, auto_suggest=False)
                    
                    doc.metadata = {
                        'source': page.url if hasattr(page, 'url') and page.url else "",
                        'title': page.title if hasattr(page, 'title') and page.title else "",
                        'summary': page.summary if hasattr(page, 'summary') and page.summary else "",
                        'content': page.content if hasattr(page, 'content') and page.content else ""
                    }
                    
                    return doc
                except Exception:
                    return None
            return None
            
        except Exception as e:
            # 网络错误或其他内部错误
            print_and_log(f"Warning: Could not fetch details for '{title}': {e}")
            return None

    # --- 主循环：应用 7 秒超时 ---
    for title in search_titles:
        try:
            # 使用线程池实现单次任务的超时控制
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fetch_single_doc, title)
                doc = future.result(timeout=7)
                
                if doc is not None:
                    results.append(doc)
                    
        except FuturesTimeoutError:
            # 捕获超时异常
            print_and_log(f"WIKI Timeout: Fetching '{title}' exceeded 7 seconds. Skipping...")
            continue
        except Exception as e:
            # 捕获其他未预料的异常
            print_and_log(f"Error processing '{title}': {e}")
            continue

    return results



def get_wiki_search_result(question_idx,current_subq, query,top_k=3,lang='en'):
    global wiki_cache
    # 为title link snippet mainText rerankScore的格式
    if wiki_cache.get(question_idx) is not None and wiki_cache[question_idx].get(query) is not None:
        print_and_log(f"Wiki search result for question_idx {question_idx} and query '{query}' found in cache.")
        return wiki_cache[question_idx][query]
    try:
        #wiki_response = {"results": search_wiki(query, lang=lang, load_max_docs=top_k)}
        wiki_response = {"results": my_get_wiki(question_idx,query, lang=lang, load_max_docs=top_k)}
    except Exception as e:
        wiki_response = {"results": []}
        print_and_log(f"Error occurred while searching: {e}")
    if len(wiki_response['results']) == 0:
        print_and_log("WIKI No search results found.Error!")
    
    search_result_list = []
    for i in range(len(wiki_response['results'])):
        item = {}
        item['question_idx'] = question_idx # 问题的id
        item['sub_question'] = current_subq # 当前的子问题
        item['query'] = query
        item['url'] = (wiki_response['results'][i].metadata)['source'] if 'source' in (wiki_response['results'][i].metadata) else ''
        item['title'] = (wiki_response['results'][i].metadata)['title'] if 'title' in (wiki_response['results'][i].metadata) else ''
        item['content'] = truncate_by_tokens((wiki_response['results'][i].metadata)['summary'],750) if 'summary' in (wiki_response['results'][i].metadata) else ''
        item['raw_content'] = (wiki_response['results'][i].metadata)['summary'] if 'summary' in (wiki_response['results'][i].metadata) else ''
        item['score'] = 1
        search_result_list.append(item.copy())
    print_and_log(f"wiki获取到的搜索结果数量: {len(search_result_list)}")
    if len(search_result_list) > 0:
        if wiki_cache.get(question_idx) is None:
            wiki_cache[question_idx] = {}
            wiki_cache[question_idx][query] = search_result_list
        else:
            wiki_cache[question_idx][query] = search_result_list
    return search_result_list


def ddgs_search(query, top_k=10,timeout=15):
    try:
        results = DDGS(timeout=timeout,verify=False).text(query, max_results=top_k, safesearch="off",region="us-en")
    except Exception as e:
        return []
    return results

# 这个需要手动调用qwen-flash来总结一下
def get_ddgs_result(question_idx,current_subq, query,top_k=5,timeout=15):
    global ddgs_cache
    try:
        if ddgs_cache.get(question_idx) is not None and ddgs_cache[question_idx].get(query) is not None:
            print_and_log(f"DDGS Cache hit for query: {query}")
            return ddgs_cache[question_idx][query]
    except Exception as e:
        pass
    # 为title link snippet mainText rerankScore的格式
    try:
        ddgs_response = {"results": ddgs_search(query,top_k,timeout)}
    except Exception as e:
        ddgs_response = {"results": []}
        print_and_log(f"Error occurred while searching: {e}")
    if len(ddgs_response['results']) == 0:
        print_and_log("DDGS No search results found.Error!")
    
    search_result_list = []
    for i in range(len(ddgs_response['results'])):
        item = {}
        item['question_idx'] = question_idx # 问题的id
        item['sub_question'] = current_subq # 当前的子问题
        item['query'] = query
        item['url'] = ddgs_response['results'][i]['href'] if 'href' in ddgs_response['results'][i] else ''
        item['title'] = ddgs_response['results'][i]['title'] if 'title' in ddgs_response['results'][i] else ''
        item['content'] = ddgs_response['results'][i]['body'] if 'body' in ddgs_response['results'][i] else ''
        item['raw_content'] = ddgs_response['results'][i]['body'] if 'body' in ddgs_response['results'][i] else ''
        item['score'] = 1
        search_result_list.append(item.copy())
    print_and_log(f"ddgs获取到的搜索结果数量: {len(search_result_list)}")
    try:
        if len(search_result_list) > 0:
            if question_idx in ddgs_cache:
                ddgs_cache[question_idx][query] = search_result_list
            else:
                ddgs_cache[question_idx] = {query: search_result_list}
    except Exception as e:
        pass
    return search_result_list



# 筛选出最有可能需要看raw_content之类的这部分
# 需要做一些上下文管理，比如目前还不能确定唯一答案，是一个集合，需要后续继续搜索
select_top_search_results_template_en = """You are a precise and analytical AI assistant. Your task is to select up to **two** search results that are **most relevant and most likely to contain the answer** to a given sub-question.

Given:
- The **main user question** (for context)
- The **sub-question** (the specific part you need to answer)
- The **search query** used to retrieve the results
- A list of **search results**, each with an index (`idx`), title, and content

Instructions:
1. Carefully evaluate **all** provided search results.
2. Select **at most two** results that:
   - Directly or indirectly provide factual information helpful for answering the **sub-question**
   - Are more informative than others (e.g., contain key entities, numbers, dates, explanations, or direct answers)
3. **Do NOT select** results that are generic, promotional, navigational, or lack concrete information.
4. Return **only the `idx` values** (as strings) of the top 1 or 2 results, in order of relevance (most relevant first).
5. If **none** are relevant, return an empty list (i.e, `[]`).

Rules:
- Do NOT use external knowledge.
- Do NOT assume missing context.
- Partial relevance (e.g., mentions a related concept, organization, or timeframe) is acceptable if it helps narrow down the answer.
- Ignore boilerplate text, ads, or repeated phrases.

Output format: a list of index seperated by commas (e.g., `[0, 2]`). Never add extra text.

Main User Question:
{main_question}

Sub-Question:
{sub_question}

Search Query:
{search_query}

Search Results:
{formatted_results}

Your output is:"""

select_top_search_results_template = select_top_search_results_template_en


multi_query_select_top_search_results_template = """You are a precise and analytical AI assistant. Your task is to select up to **two** search results that are **most relevant and most likely to contain the answer** to a given sub-question.

Given:
- The **main user question** (for context)
- The **sub-question** (the specific part you need to answer)
- A list of **search results**, each with an index (`idx`), title, and content

Instructions:
1. Carefully evaluate **all** provided search results.
2. Select **at most two** results that:
   - Directly or indirectly provide factual information helpful for answering the **sub-question**
   - Are more informative than others (e.g., contain key entities, numbers, dates, explanations, or direct answers)
3. **Do NOT select** results that are generic, promotional, navigational, or lack concrete information.
4. Return **only the `idx` values** (as strings) of the top 1 or 2 results, in order of relevance (most relevant first).
5. If **none** are relevant, return an empty list (i.e, `[]`).

Rules:
- Do NOT use external knowledge.
- Do NOT assume missing context.
- Partial relevance (e.g., mentions a related concept, organization, or timeframe) is acceptable if it helps narrow down the answer.
- Ignore boilerplate text, ads, or repeated phrases.

Output format: a list of index seperated by commas (e.g., `[0, 2]`). Never add extra text.

Main User Question:
{main_question}

Sub-Question:
{sub_question}

Search Results:
{formatted_results}

Your output is:"""

# 待修改为可并发的
# 返回值写入url_detail
def find_most_potential_url_for_visit(main_question, question_idx, sub_questions, query, search_result_list):
    formatted_results = ""
    summaries = []
    for i in range(len(search_result_list)):
        url_netloc = urlparse(search_result_list[i].get('url', '')).netloc
        title_clean = search_result_list[i].get('title', '').replace('\n', ' ')
        content_clean = search_result_list[i].get('content', '').replace('\n', ' ')
        formatted_results += f"【{i}】: {url_netloc} {title_clean} {content_clean}"
        formatted_results += "\n"
    if query=="":
        select_top_search_results_prompt = multi_query_select_top_search_results_template.format(
            main_question=main_question,
            sub_question=sub_questions,
            formatted_results=formatted_results
        )
    else:
        select_top_search_results_prompt = select_top_search_results_template.format(
            main_question=main_question,
            sub_question=sub_questions,
            search_query=query,
            formatted_results=formatted_results
        )
    select_top_search_results_response = get_model_output(select_top_search_results_prompt,choosed_model=plus_model,timeout=40)
    if DEBUG_MODE:
        print_and_log("Select top search results response is:", select_top_search_results_response)
    selected_idx = select_top_search_results_response.replace('[','').replace(']','').split(',') if select_top_search_results_response else []
    try:
        selected_idx = [int(x) for x in selected_idx]
    except Exception as e:
        selected_idx = []

    for i in range(len(search_result_list)):
        if 'baidu.baike' in search_result_list[i].get('url','').lower() or 'wikipedia' in search_result_list[i].get('url','').lower():
            selected_idx.append(i)
    selected_idx = list(set(selected_idx))
    return selected_idx


def summarize_main_search_results(
    question_idx, 
    main_question, 
    sub_question, 
    query, 
    search_result_list: List[Dict],
    max_workers=MAX_WORKER,  # 最大并发数
    timeout: int = 40      # 单个请求超时时间
):
    """并发总结搜索结果"""
    summaries = [None] * len(search_result_list)  # 预分配结果列表，保持顺序
    
    def process_single_summary_result(idx: int, search_result: Dict) -> Dict:
        """处理单个搜索结果"""
        try:
            evaluate_search_result_prompt = evaluate_search_result_template.format(
                main_question=main_question,
                sub_question=sub_question,
                search_query=query,
                content=search_result.get("title","") + "\n" + 
                        search_result.get("content","") + "\n" + search_result.get("url_detail", ""),
                URL=search_result.get("url","")
            )
            
            evaluate_search_result_response = get_quick_output(evaluate_search_result_prompt,timeout=timeout)
            
            if "NOT RELEVANT" in evaluate_search_result_response:
                evaluate_search_result_response = ""
            
            return {
                "idx": idx,
                "data": {
                    "question_idx": question_idx,
                    "sub_question": sub_question,
                    "query": query,
                    "url": search_result['url'],
                    "title": search_result['title'],
                    "content": search_result['content'],
                    "raw_content": search_result.get('raw_content', ''),
                    'url_detail': search_result.get('url_detail', ''),
                    "summary": evaluate_search_result_response
                },
                "error": None
            }
            
        except Exception as e:
            print_and_log(f"Error processing result {idx}: {e}")
            return {
                "idx": idx,
                "data": {
                    "question_idx": question_idx,
                    "sub_question": sub_question,
                    "query": query,
                    "url": search_result.get('url', ''),
                    "title": search_result.get('title', ''),
                    "content": search_result.get('content', ''),
                    "raw_content": search_result.get('raw_content', ''),
                    'url_detail': search_result.get('url_detail', ''),
                    "summary": ""
                },
                "error": str(e)
            }
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_summary_result, i, search_result): i 
            for i, search_result in enumerate(search_result_list)
        }
        
        # 收集结果（带进度日志）
        completed = 0
        total = len(search_result_list)
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result(timeout=timeout)
                summaries[idx] = result["data"]
                
                if result["error"]:
                    print_and_log(f" Result {idx}/{total} failed: {result['error']}")
                else:
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        print_and_log(f"📊 Progress: {completed}/{total} completed")
                        
            except Exception as e:
                print_and_log(f"Future exception for idx {idx}: {e}")
                # 填充空结果
                summaries[idx] = {
                    "question_idx": question_idx,
                    "sub_question": sub_question,
                    "query": query,
                    "url": search_result_list[idx].get('url', ''),
                    "title": search_result_list[idx].get('title', ''),
                    "content": search_result_list[idx].get('content', ''),
                    "raw_content": search_result_list[idx].get('raw_content', ''),
                    'url_detail': search_result_list[idx].get('url_detail', ''),
                    "summary": ""
                }
    
    # 过滤掉 None（理论上不会有）
    summaries = [s for s in summaries if s is not None]
    
    print_and_log(f"Completed: {len(summaries)}/{total} search results summarized")
    
    return summaries


def download_and_read_html_for_subquestion(main_question, question_idx, sub_question, query, search_result_list, search_idx,if_visit=True):
    evaluate_search_result_response = " "
    url = search_result_list[search_idx]['url']
    try:
        if if_visit:
            if '.pdf' in url:
                web_page_text = None
            elif 'wiki' in url or len(search_result_list[search_idx]['raw_content'])<1000: # 如果是维基百科或者太短了
                _, web_page_text = get_webpage_content(question_idx, url, if_text_only=True,timeout=10)
                if web_page_text is None or len(web_page_text.strip())<100:
                    web_page_text = jina_read_page(url)
            else:
                web_page_text = None
        else:
            web_page_text = search_result_list[search_idx]['raw_content']
    except Exception as e:
        print_and_log(f"[download_and_read_html]Error fetching webpage content: {e}")
    
    if DEBUG_MODE:
        print_and_log("[visit webpage]:", url)

    if  web_page_text is None or len(web_page_text.strip())<100:
        web_page_text = search_result_list[search_idx]['raw_content']
        if DEBUG_MODE:
            print_and_log("[Failed in][visit webpage]:", url)

    try:
        chunks = simple_chunk_text(web_page_text, chunk_size=1500, overlap=200)

        chunks = filter_chunks_with_head_tail(chunks,sub_question, query, min_keyword_len=2, max_keep=30)

        if not chunks:
            print_and_log("[chunk error] NO CHUNK!")
            return "NOT RELEVANT"
        else:
            # 2.向量化
            # 设定阈值（可调）+ top-k 约束
            threshold = 0.7
            top_k = 4
            min_keep = 2
            # 调用get_text_embedding(textin)，返回（1024,0）维度向量 可以get_text_embedding(sub_question+query)作为查询向量，然后再算相似度
            try:
                query_vec = get_text_embedding(sub_question + query)  # shape: (1024,)
                with ThreadPoolExecutor(max_workers=10) as executor:
                    chunk_vecs = list(executor.map(embed_chunk, chunks))

                #chunk_vecs = [get_text_embedding(chunk) for chunk in chunks]  # list of (1024,)
                # 转为 numpy array 便于计算
                query_vec = np.array(query_vec).reshape(1, -1)  # (1, 1024)
                chunk_vecs = np.array(chunk_vecs)               # (n_chunks, 1024)

                similarities = cosine_similarity(query_vec, chunk_vecs).flatten()  # (n_chunks,)
                

                # 获取高于阈值的索引，或 fallback 到 top-k
                high_sim_indices = np.where(similarities >= threshold)[0]
            
                if len(high_sim_indices) == 0:
                    # 若无达标，取 top-k
                    selected_indices = np.argsort(-similarities)[:min_keep]
                else:
                    # 限制最多 top_k 个
                    selected_indices = high_sim_indices[:top_k]

                # 至少保留 min_keep 个（即使低于阈值）
                if len(selected_indices) < min_keep:
                    top_all = np.argsort(-similarities)[:min_keep]
                    selected_indices = np.unique(np.concatenate([selected_indices, top_all]))

                n_chunks = len(chunks)
                if n_chunks > 0:
                    first_idx = 0
                    last_idx = n_chunks - 1
                    
                    # 将首尾索引加入列表
                    selected_indices = np.unique(np.concatenate([selected_indices, [first_idx, last_idx]]))

                # 去重并按原文顺序排序
                selected_indices = sorted(set(selected_indices.tolist()))

                # 提取选中的 chunks（按原文顺序）
                chosen_chunks = [chunks[i] for i in selected_indices]
                relevant_docs = "\n\n---\n\n".join(chosen_chunks)

                evaluate_search_result_prompt = evaluate_search_result_template.format(
                    main_question=main_question,
                    sub_question=sub_question,
                    search_query=query,
                    content=relevant_docs,
                    URL=url
                )
            except Exception as e:
                print_and_log(f"[download_and_read_html] Error during embedding: {e}")
                return "NOT RELEVANT"
            
            evaluate_search_result_response = get_model_output(evaluate_search_result_prompt,choosed_model=base_model,timeout=20)
            if DEBUG_MODE:
                print_and_log("Evaluate URL search result response is:", evaluate_search_result_response)
    except Exception as e:
        print_and_log(f"[download_and_read_html] Error during evaluation: {e}")
        return "NOT RELEVANT"

    return evaluate_search_result_response


def gen_sub_question_answer(sub_question, search_result_list,history_info,timeout=35):
    # 处理输入的search_result_list
    search_summary = ""
    for item in search_result_list:
        if 'url_detail' in item:
            if 'summary' in item:
                search_summary += ("-url {url}  {title}  Summary: {summary}   Web page detail summary: {url_detail}".format(url=item['url'], title=item['title'].replace('\n','\t') if 'title' in item and item['title'] else '', summary=item['summary'].replace('\n','\t') if 'summary' in item and item['summary'] else '', url_detail=item['url_detail'].replace('\n','\t') if 'url_detail' in item and item['url_detail'] else ''))
            else:
                search_summary += ("-url {url}  {title}  Summary: {summary}   Web page detail summary: {url_detail}".format(url=item['url'], title=item['title'].replace('\n','\t') if 'title' in item and item['title'] else '', summary=truncate_by_tokens(item['content'],750).replace('\n','\t') if 'content' in item and item['content'] else '', url_detail=item['url_detail'].replace('\n','\t') if 'url_detail' in item and item['url_detail'] else ''))
        elif 'summary' in item:
            search_summary += ("-url {url} {title}   Summary: {summary}".format(url=item['url'] if 'url' in item and item['url'] else '', title=item['title'].replace('\n','\t') if 'title' in item and item['title'] else '', summary=item['summary'].replace('\n','\t') if 'summary' in item and item['summary'] else ''))
        elif 'content' in item:
            search_summary += ("-url {url} {title}   Summary: {summary}".format(url=item['url'] if 'url' in item and item['url'] else '', title=item['title'].replace('\n','\t') if 'title' in item and item['title'] else '', summary=truncate_by_tokens(item['content'],750).replace('\n','\t') if 'content' in item and item['content'] else ''))

    answer_sub_question_prompt = answer_sub_question_template.format(
        history_info=history_info,
        sub_question=sub_question,
        search_summary=search_summary
    )
    answer_sub_question_response = get_model_output(answer_sub_question_prompt,timeout=35,choosed_model=base_model)
    return answer_sub_question_response


def gen_sub_question_answer_with_prob(sub_question, search_result_list,history_info,timeout=35):
    # 处理输入的search_result_list
    search_summary = ""
    for item in search_result_list:
        if 'url_detail' in item:
            if 'summary' in item:
                search_summary += ("-url {url}  {title}  Summary: {summary}   Web page detail summary: {url_detail}".format(url=item['url'], title=item['title'].replace('\n','\t') if 'title' in item and item['title'] else '', summary=item['summary'].replace('\n','\t') if 'summary' in item and item['summary'] else '', url_detail=item['url_detail'].replace('\n','\t') if 'url_detail' in item and item['url_detail'] else ''))
            else:
                search_summary += ("-url {url}  {title}  Summary: {summary}   Web page detail summary: {url_detail}".format(url=item['url'], title=item['title'].replace('\n','\t') if 'title' in item and item['title'] else '', summary=truncate_by_tokens(item['content'],750).replace('\n','\t') if 'content' in item and item['content'] else '', url_detail=item['url_detail'].replace('\n','\t') if 'url_detail' in item and item['url_detail'] else ''))
        elif 'summary' in item:
            search_summary += ("-url {url} {title}   Summary: {summary}".format(url=item['url'] if 'url' in item and item['url'] else '', title=item['title'].replace('\n','\t') if 'title' in item and item['title'] else '', summary=item['summary'].replace('\n','\t') if 'summary' in item and item['summary'] else ''))
        elif 'content' in item:
            search_summary += ("-url {url} {title}   Summary: {summary}".format(url=item['url'] if 'url' in item and item['url'] else '', title=item['title'].replace('\n','\t') if 'title' in item and item['title'] else '', summary=truncate_by_tokens(item['content'],750).replace('\n','\t') if 'content' in item and item['content'] else ''))

    answer_sub_question_prompt = answer_sub_question_template.format(
        history_info=history_info,
        sub_question=sub_question,
        search_summary=search_summary
    )    
    answer_sub_question_response = get_model_output(answer_sub_question_prompt,timeout=timeout, choosed_model=base_model)
    avg_prob = 0.99
    return answer_sub_question_response,avg_prob


def extract_final_answer(output: str) -> str:
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', output, re.IGNORECASE)
    if not match:
        raise ValueError("Final answer not found in <answer> tags.")
    return match.group(1).strip()


def validate_search_for_answer(question_index,original_question, search_goal,search_query,history_dense,timeout=15):
    # 下载并读取HTML内容
    search_result_list = []
    try:
        language_for_wiki = detect_query_language(search_query)
        if language_for_wiki=='en':
            search_result_list = get_wiki_search_result(question_index, search_goal, search_query,lang=language_for_wiki)
            if detect_query_language(original_question)=='zh':
                search_result_list += get_iqs_search_result(question_index, search_goal, search_query + " 百度百科",top_k=5)
        else:
            search_result_list = get_iqs_search_result(question_index, search_goal, search_query,top_k=5)
        if DEBUG_MODE:
            print_and_log(f"Current Goal: {search_goal}, Current Query: {search_query}")
        selected_idx = find_most_potential_url_for_visit(original_question, question_index, search_goal, search_query, search_result_list)
        if DEBUG_MODE:
            print_and_log(f"Selected indices for Current Goal {search_goal}, Query {search_query}: 【{selected_idx}】")
        if len(selected_idx)>0:
            with ThreadPoolExecutor() as executor:
                # 提交所有任务
                future_to_idx = {
                    executor.submit(
                        download_and_read_html_for_subquestion,
                        original_question,
                        question_index,
                        search_goal,
                        search_query,
                        search_result_list,
                        idx,
                        if_visit=True
                    ): idx
                    for idx in selected_idx
                }

            # 收集结果并更新原列表（线程安全：因为每个任务只写自己的 idx）
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    visit_html_response = future.result()
                    search_result_list[idx]['url_detail'] = visit_html_response
                except Exception as e:
                    # 可选：记录错误但不中断其他任务
                    print(f"Error processing search index {idx}: {e}")
        # for search_idx in selected_idx:
        #     visit_html_response = download_and_read_html_for_subquestion(original_question, question_index, search_goal, search_query, search_result_list, search_idx,if_visit=True)
        #     search_result_list[search_idx]['url_detail'] = visit_html_response
        #     if DEBUG_MODE:
        #         print_and_log(f"Visit HTML Response is : {visit_html_response}")
    except Exception as e:
        print_and_log(f"[download_and_read_html] Error during validate_search_for_answer: {e}")

    # 总结单个搜索结果
    search_result_list = summarize_main_search_results(
            question_idx=question_index,
            main_question=original_question,
            sub_question=search_goal,
            query=search_query,
            search_result_list=search_result_list,
            timeout=timeout
    )

    # 获取了之后，如果信息超长，压缩一下得到history_dense
    answer_sub_question_response = gen_sub_question_answer(search_goal, search_result_list, history_info=history_dense,timeout=35)
    return answer_sub_question_response


def multi_parse_react_output(llm_output: str) -> Dict[str, Any]:
    """
    Robust parser for JSON-formatted ReAct output.
    Handles both valid JSON and fallback to regex extraction.
    """
    result = {
        "thought": "",
        "action": "error",
        "goal": "",
        "query": [],
        "wiki": "",
        "answer": ""
    }

    # Try 1: Direct JSON parsing
    try:
        data = json.loads(llm_output.strip())
        
        # Map JSON fields to result
        result["thought"] = data.get("think", "").strip()
        result["goal"] = data.get("goal", "").strip()
        raw_query = data.get("query", [])
        result["wiki"] = data.get("wiki", "").strip() or ""
        result["answer"] = data.get("answer", "").strip() or ""
        

        raw_query = data.get("query")
        
        if isinstance(raw_query, list):
            # 情况 1: 本来就是列表 ["A", "B"]
            result["query"] = [str(q).strip() for q in raw_query if str(q).strip()]
        elif isinstance(raw_query, str):
            stripped_q = raw_query.strip()
            if not stripped_q:
                result["query"] = []
            else:
                # 情况 2, 3, 4: 是字符串，尝试还原为列表
                parsed_list = None
                
                # 尝试 A: 标准 JSON 解析 (处理 "[\"A\", \"B\"]")
                try:
                    parsed_list = json.loads(stripped_q)
                except (json.JSONDecodeError, ValueError):
                    pass
                
                # 尝试 B: Python 字面量解析 (处理 "['A', 'B']")
                if parsed_list is None:
                    try:
                        # ast.literal_eval 安全地解析 Python 字符串、列表、字典等
                        parsed_list = ast.literal_eval(stripped_q)
                    except (ValueError, SyntaxError):
                        pass
                
                # 验证解析结果是否为列表
                if isinstance(parsed_list, list):
                    result["query"] = [str(q).strip() for q in parsed_list if str(q).strip()]
                else:
                    # 情况 4: 解析失败或不是列表，视为单个查询
                    result["query"] = [stripped_q]
        else:
            # 其他奇怪类型，转为字符串放入列表
            result["query"] = [str(raw_query)] if raw_query is not None else []

        # Determine action
        if result["answer"]:
            result["action"] = "final_answer"
        elif result["query"]:
            result["action"] = "search"
        else:
            result["action"] = "error"
        
        return result
        
    except json.JSONDecodeError:
        pass

    # Try 2: Extract JSON from code block (```json ... ```)
    json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_output, re.DOTALL)
    if json_block_match:
        try:
            data = json.loads(json_block_match.group(1))
            result["thought"] = data.get("think", "").strip()
            result["goal"] = data.get("goal", "").strip()
            result["query"] = data.get("query", "").strip()
            result["wiki"] = data.get("wiki", "").strip() or ""
            result["answer"] = data.get("answer", "").strip() or ""
            
            if result["answer"]:
                result["action"] = "final_answer"
            elif result["query"]:
                result["action"] = "search"
            else:
                result["action"] = "error"
            
            return result
        except json.JSONDecodeError:
            pass

    # Try 3: Fallback to legacy XML tag parsing (backward compatibility)
    result = _parse_xml_fallback(llm_output)
    return result

def parse_react_output(llm_output: str) -> Dict[str, Any]:
    """
    Robust parser for JSON-formatted ReAct output.
    Handles both valid JSON and fallback to regex extraction.
    """
    result = {
        "thought": "",
        "action": "error",
        "goal": "",
        "query": "",
        "wiki": "",
        "answer": ""
    }

    # Try 1: Direct JSON parsing
    try:
        data = json.loads(llm_output.strip())
        
        # Map JSON fields to result
        result["thought"] = data.get("think", "").strip()
        result["goal"] = data.get("goal", "").strip()
        result["query"] = data.get("query", "").strip()
        result["wiki"] = data.get("wiki", "").strip() or ""
        result["answer"] = data.get("answer", "").strip() or ""
        
        # Determine action
        if result["answer"]:
            result["action"] = "final_answer"
        elif result["query"]:
            result["action"] = "search"
        else:
            result["action"] = "error"
        
        return result
        
    except json.JSONDecodeError:
        pass

    # Try 2: Extract JSON from code block (```json ... ```)
    json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_output, re.DOTALL)
    if json_block_match:
        try:
            data = json.loads(json_block_match.group(1))
            result["thought"] = data.get("think", "").strip()
            result["goal"] = data.get("goal", "").strip()
            result["query"] = data.get("query", "").strip()
            result["wiki"] = data.get("wiki", "").strip() or ""
            result["answer"] = data.get("answer", "").strip() or ""
            
            if result["answer"]:
                result["action"] = "final_answer"
            elif result["query"]:
                result["action"] = "search"
            else:
                result["action"] = "error"
            
            return result
        except json.JSONDecodeError:
            pass

    # Try 3: Fallback to legacy XML tag parsing (backward compatibility)
    result = _parse_xml_fallback(llm_output)
    return result


def _parse_xml_fallback(llm_output: str) -> Dict[str, Any]:
    """
    Fallback parser for legacy XML tag format.
    """
    result = {
        "thought": llm_output,
        "action": "error",
        "goal": "",
        "query": "",
        "wiki": "",
        "answer": ""
    }

    # Extract <think> (must be closed)
    think_match = re.search(r'<think>(.*?)</think>', llm_output, re.DOTALL | re.IGNORECASE)
    if think_match:
        result["thought"] = think_match.group(1).strip()

    # Check for <answer>
    answer_match = re.search(r'<answer>(.*?)</answer>', llm_output, re.DOTALL | re.IGNORECASE)
    if answer_match:
        result["action"] = "final_answer"
        result["answer"] = answer_match.group(1).strip()
        return result

    # Extract <goal> (must be closed)
    goal_match = re.search(r'<goal>(.*?)</goal>', llm_output, re.DOTALL | re.IGNORECASE)
    if goal_match:
        result["goal"] = goal_match.group(1).strip()
    else:
        return result

    # Extract <query> - take everything after <query> to end or next tag
    query_start = re.search(r'<query>', llm_output, re.IGNORECASE)
    if not query_start:
        return result
    
    query_start_pos = query_start.end()
    query_content = llm_output[query_start_pos:].strip()
    query_content = query_content.split('<')[0].strip()

    # Extract <wiki> if present
    wiki_match = re.search(r'<wiki>(.*?)</wiki>', llm_output, re.DOTALL | re.IGNORECASE)
    if wiki_match:
        result["wiki"] = wiki_match.group(1).strip()

    if query_content:
        result["query"] = query_content
        result["action"] = "search"
    
    return result


def normalize_title(title: str) -> str:
    """
    标准化标题：
    1. 转小写 (针对英文)
    2. 去除所有标点符号、特殊字符、空格、换行
    3. 只保留字母、数字、汉字
    这样 "成龙电影_2024!" 和 "成龙电影 2024" 都会变成 "成龙电影2024"
    """
    if not title:
        return ""
    
    # 转小写
    title = title.lower()
    
    # 正则替换：保留 字母(a-z)、数字(0-9)、汉字(\u4e00-\u9fff)，其他全部删掉
    # \u4e00-\u9fff 覆盖常用汉字
    cleaned = re.sub(r'[^a-z0-9\u4e00-\u9fff]', '', title)
    
    return cleaned

def deduplicate_by_url(results: list) -> list:
    """
    根据 url 字段对搜索结果列表去重，保留第一次出现的记录。
    """
    seen_urls = set()
    unique_results = []
    
    for item in results:
        url = item.get('url', '')
        if "zhidao.baidu.com" in url:
            continue
        if not url:
            unique_results.append(item)
            continue
            
        if url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(item)
        else:
            pass
            
    return unique_results

def deduplicate_by_url_and_title(results: list) -> list:
    """
    根据 URL 和 标准化后的 Title 对搜索结果列表去重。
    策略：
    1. 跳过百度知道 (zhidao.baidu.com)
    2. URL 完全相同 -> 去重
    3. Title 标准化后完全相同 -> 去重 (解决同文不同链问题)
    """
    seen_urls = set()
    seen_titles = set() # 存储标准化后的 title
    unique_results = []
    
    for item in results:
        url = item.get('url', '')
        title = item.get('title', '')
        
        # 1. 过滤百度知道 (根据你的需求)
        if "zhidao.baidu.com" in url:
            continue
            
        # 2. 处理缺失 URL 的情况 (保留，防止误删)
        if not url:
            # 如果没有 URL，尝试用 Title 去重，如果 Title 也没法判断，则直接保留
            norm_title = normalize_title(title)
            if norm_title and norm_title in seen_titles:
                continue # Title 重复也跳过
            if norm_title:
                seen_titles.add(norm_title)
            unique_results.append(item)
            continue
            
        # 3. URL 去重检查
        if url in seen_urls:
            continue
        
        # 4. Title 去重检查 (核心新增逻辑)
        norm_title = normalize_title(title)
        if norm_title in seen_titles:
            # 标题一样，认为是重复内容，跳过
            continue
            
        # 5. 通过检查，加入结果集
        seen_urls.add(url)
        if norm_title:
            seen_titles.add(norm_title)
        unique_results.append(item)
            
    return unique_results


def fetch_search(question_index, original_question, all_history_context, search_goal, search_query, wiki_query):
    # 初始化
    search_result_list_a = []
    search_result_list_b = []
    search_result_list_c = []
    search_result_list_d = []
    search_list_baike = []  # 保留占位（虽未使用）
    
    time_start_search = time.time()
    
    # 判断主查询是否为英文（用于决定是否跑 C）
    is_english_query = (detect_query_language(search_query) == 'en')
    
    # 所有任务结果容器
    results = {}

    # ===== 定义带日志计时的包装任务 =====
    def timed_task(name, func):
        t0 = time.time()
        res = func()
        duration = time.time() - t0
        print_and_log(f"[TIMER]{name} duration: {duration:.2f} seconds")
        return res

    # ===== 提交并发任务 =====
    with ThreadPoolExecutor() as executor:
        futures = {}

        # --- 任务 A: IQS 主搜（总是执行）---
        futures['a'] = executor.submit(
            timed_task,
            "IQS search",
            lambda: get_iqs_search_result(question_index, search_goal, search_query)
        )

        # --- 任务 D: DDGS（总是执行，因免费）---
        futures['d'] = executor.submit(
            timed_task,
            "DDGS search",
            lambda: get_ddgs_result(question_index, search_goal, search_query)
        )

        # --- 任务 B: Wiki + 百度百科（仅当 wiki_query 非空）---
        if wiki_query != "":
            def run_task_b():
                language_for_wiki = detect_query_language(wiki_query)
                if language_for_wiki == 'zh':
                    return get_iqs_search_result(question_index, search_goal, wiki_query + " 百度百科", top_k=5)
                else:
                    part1 = get_wiki_search_result(question_index, search_goal, wiki_query, lang=language_for_wiki)
                    part2 = get_iqs_search_result(question_index, search_goal, wiki_query + " 百度百科", top_k=5)
                    return part1 + part2
            futures['b'] = executor.submit(
                timed_task,
                "Wiki search",
                run_task_b
            )

        # --- 任务 C: Search Scan（仅英文时执行）---
        if is_english_query:
            futures['c'] = executor.submit(
                timed_task,
                "Search scan",
                lambda: get_search_scan_result(original_question, question_index, search_goal, search_query)
            )

        # ===== 按原始顺序收集结果 =====
        search_result_list_a = futures['a'].result()
        search_result_list_d = futures['d'].result()

        if 'b' in futures:
            search_result_list_b = futures['b'].result()
        else:
            search_result_list_b = []

        if is_english_query:
            search_result_list_c = futures['c'].result()
        else:
            search_result_list_c = []

        # 调试日志（保持原位置语义）
        if is_english_query and DEBUG_MODE:
            print_and_log(f"search scan length is: {len(search_result_list_c)}")

    # ===== 拼接（严格保持 a + b + c + d 顺序）=====
    search_result_list = search_result_list_a + search_result_list_b + search_result_list_c + search_result_list_d
    search_result_list = deduplicate_by_url_and_title(search_result_list)

    print_and_log(f"Total search results obtained: {len(search_result_list)}")
    print_and_log(f"[TIMER]Search duration: {time.time() - time_start_search:.2f} seconds")

    return search_result_list


# def fetch_search(question_index,original_question,all_history_context,search_goal,search_query,wiki_query):
#     search_result_list = []
#     search_result_list_a = []
#     search_result_list_b = []
#     search_result_list_c = []
#     search_result_list_d = []
#     search_list_baike = []
#     time_start_search = time.time()
#     if wiki_query != "":
#         language_for_wiki = detect_query_language(wiki_query)
#         wiki_begin = time.time()
#         if language_for_wiki == 'zh':
#             search_result_list_b = get_iqs_search_result(question_index, search_goal, wiki_query + " 百度百科",top_k=5)
#         else:
#             search_result_list_b = get_wiki_search_result(question_index, search_goal, wiki_query,lang=language_for_wiki)
#             search_result_list_b += get_iqs_search_result(question_index, search_goal, wiki_query + " 百度百科",top_k=5)
#         print_and_log(f"[TIMER]Wiki search duration: {time.time() - wiki_begin:.2f} seconds")
 
#     iqs_begin = time.time()
#     search_result_list_a = get_iqs_search_result(question_index, search_goal, search_query)
#     print_and_log(f"[TIMER]IQS search duration: {time.time() - iqs_begin:.2f} seconds")
#     ddgs_begin = time.time()
#     search_result_list_d = get_ddgs_result(question_index, search_goal, search_query)
#     print_and_log(f"[TIMER]DDGS search duration: {time.time() - ddgs_begin:.2f} seconds")
#     if detect_query_language(search_query) == 'en': # 英语
#         search_scan_begin = time.time()
#         search_result_list_c = get_search_scan_result(original_question,question_index, search_goal, search_query)
#         print_and_log(f"[TIMER]Search scan duration: {time.time() - search_scan_begin:.2f} seconds")
#         if DEBUG_MODE:
#             print_and_log(f"search scan length is: {len(search_result_list_b)}")
#     search_result_list = search_result_list_a + search_result_list_b + search_result_list_c + search_result_list_d
#     search_result_list = deduplicate_by_url_and_title(search_result_list)
#     print_and_log(f"Total search results obtained: {len(search_result_list)}")
#     print_and_log(f"[TIMER]Search duration: {time.time() - time_start_search:.2f} seconds")
#     return search_result_list


# 调用搜索获取结果
def single_search_step(step_cnt, question_index,original_question,all_history_context,temperature=0):
    need_rollout = False
    react_prompt = react_prompt_template.format(
            question=original_question,
            context=all_history_context
        )
    one_step_history = ""
    all_history_context+= f"\n\n===== ReAct Step {step_cnt+1} =====\n\n"
    one_step_history+= f"\n\n===== ReAct Step {step_cnt+1} =====\n\n"
    react_begin_time = time.time()
    react_response = get_model_output(react_prompt,temperature,choosed_model=plan_model,timeout=60)
    print_and_log(f"[TIMER]react duration: {time.time() - react_begin_time:.2f} seconds")
    print_and_log(f"ReAct response for this step: {react_response}\n\n")
    react_standard_output = parse_react_output(react_response)
    need_rollout = False
    is_end = False
    print_and_log(f"Parsed ReAct output: {react_standard_output}")
    if True:
        if react_standard_output.get("action") == "final_answer": # 如果是最终答案
            print_and_log(f"Final Answer: {react_standard_output.get('answer', '')}")
            is_end = True
        elif (react_standard_output.get("action","") == "search" and len(react_standard_output.get("query", "")) > 0) \
            or (len(react_standard_output.get("query", "")) > 0 and react_standard_output.get("wiki", "") != ""): # 如果需要搜索
            all_history_context += f"---\n\n<think> {react_standard_output.get('thought', '')} </think>\nGoal: {react_standard_output.get('goal', '')}\nQuery: {react_standard_output.get('query', '')}\n"
            one_step_history += f"---\n\n<think> {react_standard_output.get('thought', '')} </think>\nGoal: {react_standard_output.get('goal', '')}\nQuery: {react_standard_output.get('query', '')}\n"
            if react_standard_output.get("wiki", "") != "":
                all_history_context += f"Wiki: {react_standard_output.get('wiki', '')}\n"
                one_step_history += f"Wiki: {react_standard_output.get('wiki', '')}\n"
            search_goal = react_standard_output.get("goal", "")
            search_query = react_standard_output.get("query", "")
            print_and_log(f"Thought: {react_standard_output.get('thought', '')}")
            print_and_log(f"Search Goal: {search_goal}")
            print_and_log(f"Search Query: {search_query}")
            try:
                search_result_list = fetch_search(question_index,original_question,all_history_context,search_goal,search_query,react_standard_output.get("wiki", ""))
                try:
                    start_visit_time = time.time()
                    selected_idx = find_most_potential_url_for_visit(original_question, question_index, search_goal, search_query, search_result_list)
                    if DEBUG_MODE:
                        print_and_log(f"Selected indices for Current Goal {search_goal}, Query {search_query}: 【{selected_idx}】")
                    
                    def fetch_one(original_question, question_index, search_goal, search_query, search_result_list, search_idx):
                        # ✅ 这些变量会通过 Python 闭包自动捕获
                        response = download_and_read_html_for_subquestion(
                            original_question,
                            question_index,
                            search_goal,
                            search_query,
                            search_result_list,  # 注意：只读传递，实际修改在主线程
                            search_idx
                        )
                        return search_idx, response

                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(fetch_one, original_question, question_index, search_goal, search_query, search_result_list, idx) for idx in selected_idx]

                        for future in as_completed(futures):
                            try:
                                search_idx, visit_html_response = future.result()
                                # ✅ 在主线程中更新列表，绝对线程安全
                                search_result_list[search_idx]['url_detail'] = visit_html_response
                                if DEBUG_MODE:
                                    print_and_log(f"Visit HTML Response for idx {search_idx}: {visit_html_response}")
                            except Exception as e:
                                print_and_log(f"Failed to fetch idx")
                    print_and_log(f"[TIMER]Visit duration for all selected URLs: {time.time() - start_visit_time:.2f} seconds")
                except Exception as e:
                    print_and_log(f"[download_and_read_html] Error during download_and_read_html_for_subquestion: {e}")
                    pass
                # 总结单个搜索结果
                summary_begin_time = time.time()
                search_result_list = summarize_main_search_results(
                        question_idx=question_index,
                        main_question=original_question,
                        sub_question=search_goal,
                        query=search_query,
                        search_result_list=search_result_list
                )
                print_and_log(f"[TIMER]Summarization duration: {time.time() - summary_begin_time:.2f} seconds")

                # 获取了之后，如果信息超长，压缩一下得到history_dense
                subanswer_begin_time = time.time()
                answer_sub_question_response,avg_prob = gen_sub_question_answer_with_prob(search_goal, search_result_list, history_info=all_history_context,timeout=40)
                if avg_prob is not None and avg_prob<0.01:
                    need_rollout = True
                print_and_log(f"Answer to the goal in this search is: {answer_sub_question_response}, avg prob is {avg_prob}")
                try:
                    json_dict = json.loads(answer_sub_question_response)
                    answer_reserve = {
                        "answer": json_dict.get("answer", ""),
                        "answer_candidates": json_dict.get("answer_candidates", []),
                        "answer_reason":get_quick_output(short_compress_template.format(full_text=json_dict.get("reason", ""))) if "百科" not in json_dict.get("reason", "") else json_dict.get("reason", "")
                    }
                    answer_sub_question_response = json.dumps(answer_reserve, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass

                all_history_context += f"Answer to the goal in this search is: {answer_sub_question_response}\n\n----"
                one_step_history += f"Answer to the goal in this search is: {answer_sub_question_response}\n\n----"
                print_and_log(f"[TIMER]Sub-answer generation duration: {time.time() - subanswer_begin_time:.2f} seconds")
                print_and_log(f"[compressed]Answer to the goal in this search is: {answer_sub_question_response}, avg prob is {avg_prob}")
                return react_response, answer_sub_question_response, one_step_history, need_rollout, is_end

            except Exception as e:
                print_and_log(f"Error during search step: {e}")
    return react_response, "", all_history_context, need_rollout,is_end


# 调用搜索获取结果
def multi_search_step(step_cnt, question_index,original_question,all_history_context,temperature=0):
    need_rollout = False
    react_prompt = multi_react_prompt_template.format(
            question=original_question,
            context=all_history_context
        )
    one_step_history = ""
    all_history_context+= f"\n\n===== ReAct Step {step_cnt+1} =====\n\n"
    one_step_history+= f"\n\n===== ReAct Step {step_cnt+1} =====\n\n"
    react_begin_time = time.time()
    react_response = get_model_output(react_prompt,temperature,choosed_model=plan_model,timeout=60)
    print_and_log(f"[TIMER]react duration: {time.time() - react_begin_time:.2f} seconds")
    print_and_log(f"ReAct response for this step: {react_response}\n\n")
    react_standard_output = multi_parse_react_output(react_response)
    need_rollout = False # that is ok
    is_end = False
    print_and_log(f"Parsed ReAct output: {react_standard_output}")
    if True:
        if react_standard_output.get("action") == "final_answer": # 如果是最终答案
            print_and_log(f"Final Answer: {react_standard_output.get('answer', '')}")
            is_end = True
        elif (react_standard_output.get("action","") == "search" and len(react_standard_output.get("query", [])) > 0) \
            or (len(react_standard_output.get("query", "")) > 0 and react_standard_output.get("wiki", "") != ""): # 如果需要搜索
            all_history_context += f"---\n\n<think> {react_standard_output.get('thought', '')} </think>\nGoal: {react_standard_output.get('goal', '')}\nQuery: {react_standard_output.get('query', '')}\n"
            one_step_history += f"---\n\n<think> {react_standard_output.get('thought', '')} </think>\nGoal: {react_standard_output.get('goal', '')}\nQuery: {react_standard_output.get('query', '')}\n"
            if react_standard_output.get("wiki", "") != "":
                all_history_context += f"Wiki: {react_standard_output.get('wiki', '')}\n"
                one_step_history += f"Wiki: {react_standard_output.get('wiki', '')}\n"
            search_goal = react_standard_output.get("goal", "")
            search_querys = react_standard_output.get("query", [])
            search_result_list = []
            
            wiki_query = react_standard_output.get("wiki", "")
            try:
                # wikis = [wiki_query,""]
                # for idx, search_query in enumerate(search_querys):
                #     time_start_search = time.time()
                #     search_result_list += (fetch_search(question_index,original_question,all_history_context,search_goal,search_query,wikis[idx]))
                #     print_and_log(f"[TIMER]Search duration: {time.time() - time_start_search:.2f} seconds")
                wikis = [wiki_query] + [""] * (len(search_querys) - 1) if search_querys else []
                search_args = [
                    (question_index, original_question, all_history_context, search_goal, sq, wikis[i])
                    for i, sq in enumerate(search_querys)
                ]
                search_result_list = []
                time_start_search = time.time()
                if search_args:
                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(fetch_search, *args) for args in search_args]
                        
                        for future in futures:
                            try:
                                search_result_list.extend(future.result())
                            except Exception as e:
                                print_and_log(f"Error in one of the search queries: {e}")
                print_and_log(f"[TIMER]Search duration: {time.time() - time_start_search:.2f} seconds")
                try:
                    start_visit_time = time.time()
                    selected_idx = find_most_potential_url_for_visit(original_question, question_index, search_goal, "", search_result_list)
                    if DEBUG_MODE:
                        print_and_log(f"Selected indices for Current Goal {search_goal}, Query "": 【{selected_idx}】")
                    
                    def fetch_one(original_question, question_index, search_goal, search_query, search_result_list, search_idx):
                        # ✅ 这些变量会通过 Python 闭包自动捕获
                        response = download_and_read_html_for_subquestion(
                            original_question,
                            question_index,
                            search_goal,
                            search_query,
                            search_result_list,  # 注意：只读传递，实际修改在主线程
                            search_idx
                        )
                        return search_idx, response

                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(fetch_one, original_question, question_index, search_goal, "", search_result_list, idx) for idx in selected_idx]

                        for future in as_completed(futures):
                            try:
                                search_idx, visit_html_response = future.result()
                                # ✅ 在主线程中更新列表，绝对线程安全
                                search_result_list[search_idx]['url_detail'] = visit_html_response
                                if DEBUG_MODE:
                                    print_and_log(f"Visit HTML Response for idx {search_idx}: {visit_html_response}")
                            except Exception as e:
                                print_and_log(f"Failed to fetch idx")
                    print_and_log(f"[TIMER]Visit duration for all selected URLs: {time.time() - start_visit_time:.2f} seconds")
                except Exception as e:
                    print_and_log(f"[download_and_read_html] Error during download_and_read_html_for_subquestion: {e}")
                    pass
                # 总结单个搜索结果
                summary_begin_time = time.time()
                search_result_list = summarize_main_search_results(
                        question_idx=question_index,
                        main_question=original_question,
                        sub_question=search_goal,
                        query="",
                        search_result_list=search_result_list
                )
                print_and_log(f"[TIMER]Summarization duration: {time.time() - summary_begin_time:.2f} seconds")

                # 获取了之后，如果信息超长，压缩一下得到history_dense
                subanswer_begin_time = time.time()
                answer_sub_question_response, avg_prob = gen_sub_question_answer_with_prob(search_goal, search_result_list, history_info=all_history_context,timeout=40)
                if avg_prob is not None and avg_prob<0.01:
                    need_rollout = True
                try:
                    json_dict = json.loads(answer_sub_question_response)
                    answer_reserve = {
                        "answer": json_dict.get("answer", ""),
                        "answer_candidates": json_dict.get("answer_candidates", []),
                        "answer_reason":get_quick_output(think_compress_template.format(full_text=json_dict.get("reason", "")))
                    }
                    answer_sub_question_response = json.dumps(answer_reserve, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass

                all_history_context += f"Answer to the goal in this search is: {answer_sub_question_response}\n\n----"
                one_step_history += f"Answer to the goal in this search is: {answer_sub_question_response}\n\n----"
                print_and_log(f"[TIMER]Sub-answer generation duration: {time.time() - subanswer_begin_time:.2f} seconds")
                print_and_log(f"Answer to the goal in this search is: {answer_sub_question_response}, avg prob is {avg_prob}")
                return react_response, answer_sub_question_response, one_step_history, need_rollout, is_end

            except Exception as e:
                print_and_log(f"Error during search step: {e}")
    return react_response, "", all_history_context, need_rollout,is_end



def gen_answer_to_question(question_index, react_response, original_question, all_history_context):
    react_standard_output = parse_react_output(react_response)
    final_answer_response = react_standard_output.get('answer', '')
    if react_standard_output.get('answer', '')=='':
        # 回答最终的问题
        final_answer_prompt = final_answer_template.format(
            question=original_question,
            keyinfo_summary_global=all_history_context
        )
        final_answer_response = get_max_model_output(final_answer_prompt,timeout=30)
        if DEBUG_MODE:
            print_and_log("Final answer response:", final_answer_response)

        if 'Unknown' in final_answer_response:
            final_answer_prompt= strong_fallback_template.format(
                question=original_question,
                keyinfo_summary_global=all_history_context
            )
            final_answer_response = get_max_model_output(final_answer_prompt,timeout=30)
            if DEBUG_MODE:
                print_and_log("Final answer response (fallback):", final_answer_response)

    # 对于答案，校验一下名称是否相符，生成一个搜索的goal和query，来验证这个答案的格式信息对不对
    name_consistency_verification_prompt = name_consistency_verification_template.format(
        question=original_question,
        candidate_answer=final_answer_response
    )
    name_consistency_verification_response = get_model_output(name_consistency_verification_prompt,timeout=30,choosed_model=max_model)
    if DEBUG_MODE:
        print_and_log("Name consistency verification response:", name_consistency_verification_response)
    react_standard_output = parse_react_output(name_consistency_verification_response)
    print_and_log(f"Parsed ReAct output: {react_standard_output}")

    search_goal = react_standard_output.get("goal", "")
    search_query = react_standard_output.get("query", "")
    print_and_log(f"Thought: {react_standard_output.get('thought', '')}")
    print_and_log(f"Search Goal: {search_goal}")
    print_and_log(f"Search Query: {search_query}")
    if len(search_query)>0:
        final_answer_response = validate_search_for_answer(question_index,original_question, search_goal,search_query,all_history_context,timeout=30)
        all_history_context += f"---\n\nVerify the Answer Form. Thought: {react_standard_output.get('thought', '')}\nGoal: {react_standard_output.get('goal', '')}\nQuery: {react_standard_output.get('query', '')}\n"
        all_history_context += f"Answer to the goal in this verify search is: {final_answer_response}\n\n----"
    else:
        if len(react_standard_output.get("answer", "")) > 0:
            final_answer_response = react_standard_output.get("answer")
        else:
            final_answer_response = react_standard_output

    # 做答案的结构化处理，例如语言等
    final_structure_answer_prompt = final_structure_template.format(
        question=original_question,
        answer=final_answer_response,
        summary=all_history_context
    )
    final_structure_answer = get_max_model_output(final_structure_answer_prompt,timeout=30)
    final_answer = extract_final_answer(final_structure_answer)
    print_and_log(f"\n\n===== Single Rollout Structured Answer for Question {question_index}: {final_answer} =====\n\n")
    final_think = react_standard_output.get('thought', '')
    final_history = all_history_context
    return final_answer, final_think, final_history


def answer_single_question_with_single_query(original_question, question_index):
    single_rollout_begin = time.time()
    final_answer = "[ERROR]"
    print_and_log(f"\n\n===== Processing Question {question_index}: {original_question} =====\n\n",need_print=True)
    all_history_context_list = []
    all_history_context = ""  # 所有的原始的输入输出信息 think+observation+action
    one_step_history = ""
    step_cnt = 0
    MAX_STEP_CNT = 12
    
    for step_cnt in range(MAX_STEP_CNT):
        one_step_history = ""
        step_start_time = time.time()
        if step_start_time-single_rollout_begin>SINGLE_MAX_TIME: # 防止超时
            print_and_log(f"Single rollout for Question {question_index} is taking too long ({step_start_time-single_rollout_begin:.2f} seconds), breaking out of the loop.",need_print=True)
            break
        print_and_log(f"\n\n===== ReAct Step {step_cnt+1} =====\n\n",need_print=True)
        if step_cnt > MAX_STEP_CNT:
            break

        # 单次 rollout 就行了
        react_response, answer_sub_question_response, one_step_history_rt, need_rollout, is_end = \
            single_search_step(step_cnt,question_index, original_question, all_history_context)
        compressed_think = condense_think_context(one_step_history_rt)
        one_step_history += compressed_think
        all_history_context_list.append(one_step_history)

        if is_end:  # 有最终结果了
            break

        # 压缩上下文
        if len(all_history_context_list)>2:
            condense_begin = time.time()
            print_and_log(f"Context length before condensation: {len(' '.join(all_history_context_list[:-2]))}")
            compress_context = condense_context('\n'.join(all_history_context_list[:-2]))

            print_and_log(f"[TIMER]Context condensation duration: {time.time() - condense_begin:.2f} seconds, lenth of compress_context is {len(compress_context)}")
            all_history_context_list = [compress_context] + all_history_context_list[-2:]

        all_history_context = '\n'.join(all_history_context_list)

        print_and_log(f"[TIMER]Step {step_cnt} duration: {time.time() - step_start_time:.2f} seconds")

    final_time_begin = time.time()
    final_answer, final_think, final_history = gen_answer_to_question(question_index, react_response, original_question, all_history_context)
    print_and_log(f"[TIMER]Final single rollout answer generation duration: {time.time() - final_time_begin:.2f} seconds")
    print_and_log(f"[TIMER]Total single rollout processing duration for Question {question_index}: {time.time() - single_rollout_begin:.2f} seconds",need_print=True)
    return final_answer, final_think, final_history



def answer_single_question_with_multi_query(original_question, question_index):
    single_rollout_begin = time.time()
    final_answer = "[ERROR]"
    print_and_log(f"\n\n===== Processing Question {question_index}: {original_question} =====\n\n",need_print=True)
    all_history_context_list = []
    all_history_context = ""  # 所有的原始的输入输出信息 think+observation+action
    one_step_history = ""
    step_cnt = 0
    MAX_STEP_CNT = 12
    
    for step_cnt in range(MAX_STEP_CNT):
        one_step_history = ""
        step_start_time = time.time()
        if step_start_time-single_rollout_begin>SINGLE_MAX_TIME: # 防止超时
            print_and_log(f"Single rollout for Question {question_index} is taking too long ({step_start_time-single_rollout_begin:.2f} seconds), breaking out of the loop.",need_print=True)
            break
        print_and_log(f"\n\n===== ReAct Step {step_cnt+1} =====\n\n",need_print=True)
        if step_cnt > MAX_STEP_CNT:
            break

        # 单次 rollout 就行了
        react_response, answer_sub_question_response, one_step_history_rt, need_rollout, is_end = \
            multi_search_step(step_cnt,question_index, original_question, all_history_context)
        compressed_think = condense_think_context(one_step_history_rt)
        one_step_history += compressed_think
        all_history_context_list.append(one_step_history)

        if is_end:  # 有最终结果了
            break

        # 压缩上下文
        if len(all_history_context_list)>2:
            condense_begin = time.time()
            print_and_log(f"Context length before condensation: {len(' '.join(all_history_context_list[:-2]))}")
            compress_context = condense_context('\n'.join(all_history_context_list[:-2]))
            print_and_log(f"[TIMER]Context condensation duration: {time.time() - condense_begin:.2f} seconds, lenth of compress_context is {len(compress_context)}")
            all_history_context_list = [compress_context] + all_history_context_list[-2:]

        all_history_context = '\n'.join(all_history_context_list)

        print_and_log(f"[TIMER]Step {step_cnt} duration: {time.time() - step_start_time:.2f} seconds")

    final_time_begin = time.time()
    final_answer, final_think, final_history = gen_answer_to_question(question_index, react_response, original_question, all_history_context)
    print_and_log(f"[TIMER]Final single rollout answer generation duration: {time.time() - final_time_begin:.2f} seconds")
    print_and_log(f"[TIMER]Total single rollout processing duration for Question {question_index}: {time.time() - single_rollout_begin:.2f} seconds",need_print=True)
    return final_answer, final_think, final_history


multi_rollout_merger_template = """You are an expert AI judge tasked with selecting the single best final answer from multiple completed reasoning rollouts for a GAIA-style factual question.

You are given:
- The **original question**
- Several **candidate rollouts**, each containing a <think> rationale and an <answer>

Your job is to choose **one and only one** answer from the provided candidates, based on:
✅ Factual alignment with the original question  
✅ Correct language/script (e.g., Chinese characters for Chinese historical terms in Chinese questions)  
✅ Adherence to any format example (e.g., "Alibaba Group Limited" → must include legal suffix)  
✅ Absence of clear errors (e.g., wrong person, time, or romanization like "shunzhi")

Output Rules:
1. Start with   <think>...</think>  (max 2 sentences). Briefly state:
   - Whether any candidate(s) violate constraints?
   - Why the selected answer is the most compliant (even if imperfect)

2. Then output:
   - Exactly one <answer>...</answer> from the candidates (do NOT invent new text)
   - A concise <evidence>...</evidence> block that lists:
     • Which rollout(s) were considered
     • Key reasons for accepting/rejecting each (e.g., “Rollout 2: used English script for Chinese-era name”)
     • Reference to specific requirements in the original question

Now perform the merge:

Original Question: {question}

Candidate Rollouts:
{rollouts}

Your output <think>...</think>, <answer>...</answer>, and <evidence>...</evidence> are:
"""

import re

def robust_parse_rollout_output(raw_output: str):
    """
    Robustly parse model output that may contain:
      <think>...</think>
      <answer>...</answer>
      <evidence>...</evidence>
    
    Fallback strategy:
      - If all tags present → extract structurally
      - If ANY tag missing or parsing fails → treat entire raw_output as <answer>
    
    Returns:
      dict with keys: "think", "answer", "evidence"
      (all values are strings; empty string if not found and no fallback needed)
    """
    try:
        # Use non-greedy matching with DOTALL to handle multi-line content
        think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL | re.IGNORECASE)
        evidence_match = re.search(r'<evidence>(.*?)</evidence>', raw_output, re.DOTALL | re.IGNORECASE)

        # Check if at least <answer> is present (minimum requirement for structured mode)
        if answer_match:
            return {
                "think": (think_match.group(1).strip() if think_match else ""),
                "answer": answer_match.group(1).strip(),
                "evidence": (evidence_match.group(1).strip() if evidence_match else "")
            }
        else:
            # No <answer> tag found → fall back to raw output as answer
            return {
                "think": "",
                "answer": raw_output.strip(),
                "evidence": ""
            }

    except Exception:
        # Any unexpected error (e.g., regex engine issue) → safe fallback
        return {
            "think": "",
            "answer": raw_output.strip(),
            "evidence": ""
        }

def process_multiple_rollouts(original_question,question_index, rollout_num=3,max_workers=3,search_type='single'):
    global web_cache, wiki_cache, wiki_query_cache, ddgs_cache, google_cache, iqs_cache

    rollout_begin_time = time.time()
    answer_list = []
    think_list = []
    history_list = []

    def run_single_query(temperature):
        """每个任务只依赖原始问题和索引，不依赖 rollout_id"""
        final_answer, final_think, final_history = "", "", ""
        try:
            final_answer, final_think, final_history = answer_single_question_with_single_query(original_question, question_index)
        except Exception as e:
            pass
        return final_answer, final_think, final_history
    
    def run_multi_query(temperature):
        final_answer, final_think, final_history = "", "", ""
        try:
            final_answer, final_think, final_history = answer_single_question_with_multi_query(original_question, question_index)
        except Exception as e:
            pass
        return final_answer, final_think, final_history

    # 提交所有任务（无需 rollout_id 作为输入）
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        if search_type=='multi':
            futures += [executor.submit(run_multi_query, temperature=0.05*i) for i in range(1)]
        elif search_type=='all':
            futures += [executor.submit(run_multi_query, temperature=0.05*i) for i in range(1)]
            futures += [executor.submit(run_single_query, temperature=0.05*i) for i in range(1)]
        elif search_type=='final':
            futures += [executor.submit(run_multi_query, temperature=0.05*i) for i in range(1)]
            futures += [executor.submit(run_single_query, temperature=0.05*i) for i in range(2)]
        else:
            futures += [executor.submit(run_single_query, temperature=0.05*i) for i in range(1)]

        # 等待所有完成，并按提交顺序收集（as_completed 不保证顺序，但我们不需要顺序！）
        results = []
        for future in futures:  # 直接遍历 futures 保持提交顺序
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print_and_log(f"⚠️ Rollout failed: {e}")
                results.append(("ERROR", "Rollout crashed", ""))

    # 解包结果（顺序就是 Rollout 1, 2, ..., N）
    answer_list = [r[0] for r in results]
    think_list = [r[1] for r in results]
    history_list = [r[2] for r in results]

    # 构造 rollouts 字符串
    rollout_texts = []
    for i in range(rollout_num):
        rollout_texts.append(
            f"Rollout {i+1}:\n"
            f"<evidence>{history_list[i]}</evidence>\n"
            f"<think>{think_list[i]}</think>\n"
            f"<answer>{answer_list[i]}</answer>"
        )
    rollouts_str = "\n\n".join(rollout_texts)

    # 合并
    multi_rollout_merger_prompt = multi_rollout_merger_template.format(
        question=original_question,
        rollouts=rollouts_str
    )
    merge_begin = time.time()
    roll_out_answer = get_max_model_output(multi_rollout_merger_prompt,timeout=45)
    print_and_log(f"[TIMER]Multi-rollout merge duration: {time.time() - merge_begin:.2f} seconds")
    if DEBUG_MODE:
        print_and_log("Multi Rollout Merger Response:", roll_out_answer)
    # 解析
    merged_answer = robust_parse_rollout_output(roll_out_answer).get("answer", "")
    evidence = robust_parse_rollout_output(roll_out_answer).get("evidence", "")

    
    # 对于答案，校验一下名称是否相符，生成一个搜索的goal和query，来验证这个答案的格式信息对不对
    name_consistency_verification_prompt = name_consistency_verification_template.format(
        question=original_question,
        candidate_answer=merged_answer
    )
    verification_llm_begin = time.time()
    name_consistency_verification_response = get_model_output(name_consistency_verification_prompt,choosed_model=max_model,timeout=30)
    print_and_log(f"[TIMER]Name consistency verification llm duration: {time.time() - verification_llm_begin:.2f} seconds")
    if DEBUG_MODE:
        print_and_log("Name consistency verification response:", name_consistency_verification_response)
    react_standard_output = parse_react_output(name_consistency_verification_response)
    print_and_log(f"Parsed ReAct output: {react_standard_output}")

    search_goal = react_standard_output.get("goal", "")
    search_query = react_standard_output.get("query", "")
    print_and_log(f"Thought: {react_standard_output.get('thought', '')}")
    print_and_log(f"Search Goal: {search_goal}")
    print_and_log(f"Search Query: {search_query}")
    if len(search_query)>0:
        search_begin = time.time()
        final_answer_response = validate_search_for_answer(question_index,original_question, search_goal,search_query,evidence,timeout=30)
        print_and_log(f"[TIMER]Verification Search and answer duration: {time.time() - search_begin:.2f} seconds")
        evidence += f"---\n\nVerify the Answer Form. Thought: {react_standard_output.get('thought', '')}\nGoal: {react_standard_output.get('goal', '')}\nQuery: {react_standard_output.get('query', '')}\n"
        evidence += f"Answer to the goal in this verify search is: {final_answer_response}\n\n----"
    else:
        if len(react_standard_output.get("answer", "")) > 0:
            final_answer_response = react_standard_output.get("answer")
        else:
            final_answer_response = react_standard_output

    # 做答案的结构化处理，例如语言等
    final_structure_time = time.time()
    final_structure_answer_prompt = final_structure_template.format(
        question=original_question,
        answer=final_answer_response,
        summary=evidence
    )
    final_structure_answer = get_max_model_output(final_structure_answer_prompt,timeout=30)
    print_and_log(f"[TIMER]Final structure answer generation duration: {time.time() - final_structure_time:.2f} seconds")
    final_answer = extract_final_answer(final_structure_answer)
    print_and_log(f"\n\n===== Final Structured Answer for Question {question_index}: {final_answer} =====\n\n",need_print=True)
    print_and_log(f"[TIMER]Total Answer multi-rollout processing duration for Question {question_index}: {time.time() - rollout_begin_time:.2f} seconds",need_print=True)
    try:
        if web_cache.get(question_index):
            web_cache[question_index] = {}
        if wiki_cache.get(question_index):
            wiki_cache[question_index] = {}
        if wiki_query_cache.get(question_index):
            wiki_query_cache[question_index] = {}
        if ddgs_cache.get(question_index):
            ddgs_cache[question_index] = {}
        if google_cache.get(question_index):
            google_cache[question_index] = {}
        if iqs_cache.get(question_index):
            iqs_cache[question_index] = {}
    except Exception as e:
        pass
    return final_answer



# df_question = pd.read_json('question_new.jsonl', lines=True)
# question = df_question.loc[78, 'question']
# final_answer = process_multiple_rollouts(question, 1,rollout_num=1,max_workers=3,search_type='multi')


