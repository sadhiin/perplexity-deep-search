# Given user query generate list of search queries
generate_search_queries_prompt = """Given the user query: '{user_query}', generate a set of well-structured search queries to retrieve the most relevant information.

Guidelines:
- Identify key components of the query and determine if multiple searches are required to cover different aspects.
- Generate a logical sequence of search queries that refine and expand the results progressively.
- Ensure that the total number of search queries does not exceed {MAX_QUERY_GENERATIONS}.
- Use variations in phrasing, synonyms, and alternative search approaches where applicable to maximize coverage.
- Today's date is {current_date} for your reference if needed.

Output Format:
- Provide each search query on a new line without any additional text, explanations, or headers.
- Do no give triple backticks or any other formatting, just the query itself."""


# Enhanced search query generation prompt optimized for SearchQueryLLM
enhanced_search_queries_prompt = """Research Topic: {user_query}

Context: {context}

Task: Generate {max_queries} strategic and diverse search queries to comprehensively research this topic. Each query should target different aspects, perspectives, or dimensions of the research question.

Search Strategy Guidelines:
1. **Core Query**: Create one direct, comprehensive query using the main keywords
2. **Perspective Queries**: Generate queries from different angles (technical, business, social, historical)
3. **Specific Queries**: Include queries targeting recent developments, statistics, case studies, or expert opinions
4. **Comparative Queries**: If applicable, include queries comparing alternatives or showing evolution over time

Quality Criteria:
- Use specific, targeted keywords rather than vague terms
- Include temporal indicators when relevant (recent, 2024, latest, current)
- Vary query structure and approach to maximize diverse results
- Avoid redundant or overly similar queries
- Each query should be 3-15 words for optimal search engine performance

Output Format: Return each search query on a separate line with no numbering, bullets, or additional formatting."""


# Enhanced query refinement prompt optimized for SearchQueryLLM
enhanced_refine_queries_prompt = """Original Research Question: {user_query}

Previous Search Queries:
{previous_queries}

Search Results Analysis:
{search_results_summary}

Task: Based on the search results obtained so far, identify information gaps and generate {max_queries} refined search queries to fill these gaps.

Refinement Strategy:
1. **Gap Analysis**: What key information is missing from the current results?
2. **Deeper Dive**: Which aspects need more detailed exploration?
3. **Alternative Angles**: What perspectives or viewpoints haven't been covered?
4. **Recent Updates**: Are there newer developments or recent changes to explore?
5. **Specificity**: Can we get more specific data, statistics, or case studies?

Refinement Guidelines:
- Avoid repeating queries that already provided good results
- Target the specific information gaps identified
- Use different terminology or phrasing from previous queries
- Consider domain-specific or technical terms that might yield better results
- Include queries for contrasting viewpoints or alternative perspectives

Output Format: Return each refined search query on a separate line with no numbering, bullets, or additional formatting."""


# Given user query, search queries and search results, generate refined search queries
refine_search_queries_prompt = """For the given user query: `{user_query}`, we previously generated search queries and obtained the following search results:

Search Queries:
```
{search_queries}
```

Search Results:
```
{search_results}
```

Task:
Evaluate the search results to identify new terms, insights, or related concepts that could enhance search effectiveness. If needed, generate refined search queries to improve relevance and depth.

Guidelines:
- Identify new keywords, insights, or related concepts from the search results that may warrant additional searches.
- Generate refined search queries that enhance clarity, specificity, or breadth.
- Ensure the total number of refined queries does not exceed {MAX_QUERY_GENERATIONS}.
- Avoid redundant queries that do not meaningfully improve search outcomes.
- Today's date is {current_date} for your reference if needed.

Output Format:
- Provide each refined search query on a new line, without any additional text, explanations, or headers.
- Do no give triple backticks or any other formatting, just the query itself."""


# Enhanced final report prompt optimized for ThinkingLLM
enhanced_final_report_prompt = """#### Advanced Research Report Generation Task

You are a senior research analyst tasked with creating a comprehensive, well-structured research report. Your expertise lies in synthesizing complex information, identifying key insights, and presenting findings in a clear, actionable format.

#### Research Context
**Original Query**: {user_query}
**Research Scope**: Deep analysis based on multiple search queries and comprehensive source review

#### Source Material
**Search Results**:
{search_results}

#### Report Structure Requirements

**Phase 1: Individual Source Analysis**
For each search result, provide a concise summary enclosed in `<summary>` tags:
- Start with the source title
- Provide 1-2 sentences highlighting the key contribution of this source
- Note any unique insights, data points, or perspectives

**Phase 2: Comprehensive Research Report**
Generate a detailed markdown report enclosed in `<final_markdown_report>` tags with the following structure:

1. **Executive Summary** (2-3 paragraphs)
   - Key findings and main conclusions
   - Most significant insights discovered
   - Overall assessment of the research question

2. **Detailed Analysis** (Dynamic sections based on findings)
   - Organize findings into logical themes/sections
   - Present evidence-based insights with proper attribution
   - Include relevant statistics, quotes, or data points
   - Address multiple perspectives when available

3. **Key Insights & Implications**
   - Novel discoveries or surprising findings
   - Practical implications and applications
   - Potential impact or significance

4. **Conclusions & Future Considerations**
   - Direct answers to the original research question
   - Areas requiring further investigation
   - Recommended next steps or actions

#### Quality Standards
- **Evidence-Based**: Every claim must be supported by the provided sources
- **Balanced**: Present multiple viewpoints when they exist
- **Current**: Prioritize recent information and developments
- **Professional**: Use clear, professional language appropriate for decision-makers
- **Actionable**: Include practical implications and recommendations where relevant
- **Cited**: Use inline citations with source links [source title](url) for major claims

#### Output Format
1. First, provide all source summaries within `<summary>` tags
2. Then, provide the comprehensive report within `<final_markdown_report>` tags
3. Use proper markdown formatting for structure and readability
4. Maintain professional tone throughout

User Query:
```
{user_query}
```

Search Results:
```
{search_results}
```"""


# Given user query and search results, generate search result wise summary and final markdown report
final_report_prompt = """#### Task
Generate a concise and well-structured markdown report based on the given user query and retrieved search results. The report should synthesize key insights, highlight critical information, and present findings in a clear and actionable manner.

Additionally, provide an extremely brief 1-2 line summary for each search result, mentioning its title first. These summaries should be enclosed within `<summary>` and `</summary>` tags. After all summaries, generate the final markdown report enclosed within `<final_markdown_report>` and `</final_markdown_report>` tags.

The structure of the final report is not rigid and should be dynamically determined based on the user query. Sections and subsections should be organized logically to best present the information relevant to the query.

#### Input Parameters
- **User Query**: The original query provided by the user.
- **Search Results**: The retrieved information from the search process.

#### Output Structure
1. **Summaries of Search Results**
   - Each search result summary should start with its title.
   - Provide an extremely brief (1-2 line) summary for each result.
   - Enclose each summary within `<summary>` and `</summary>` tags.

   **Example Format:**
   ```
   <summary>
   "Title of the Search Result Page"
   Extremely brief summary of this search result page.
   </summary>
   ```

2. **Final Markdown Report**
   - After presenting all search result summaries, generate the final markdown report.
   - The structure of the report should be dynamically determined based on the user query.
   - Enclose the entire report within `<final_markdown_report>` and `</final_markdown_report>` tags.

   **Example Format:**
   ```
   <final_markdown_report>
   # Title
   ## Relevant Section Based on Query
   ...
   ## Another Relevant Section
   ...
   ## Additional Insights
   ...
   </final_markdown_report>
   ```

#### Guidelines
1. **Title & Introduction**
   - Begin with a clear, precise title that captures the report's focus.
   - Provide a brief introduction explaining the context and objective based on the user query.

2. **Dynamic Structure for Key Insights & Analysis**
   - Extract and present the most valuable insights in a structured format.
   - The report should adapt its sectioning based on the nature of the query.
   - Use comparisons, statistical insights, or noteworthy trends where applicable.
   - Keep content direct and to the point with clear subheadings.

3. **Recommendations (If Applicable)**
   - Provide actionable recommendations based on the insights gathered.
   - Suggest next steps or areas for further research if relevant.

4. **Conclusion**
   - Summarize key takeaways succinctly.
   - Reinforce the significance of findings in relation to the user's query.

#### Output Format
- The final report should be formatted in **Markdown**.
- Use appropriate **headings, bullet points, and code blocks** (if necessary) for clarity.
- Ensure the content is structured, professional, and to the point, avoiding unnecessary details.
- Present search result summaries first, followed by the dynamically structured final report.
- Cite references where necessary to support the findings in final report using the links (hrefs) of the search result pages. Do not include a references section in the report though, only cite links for claims within the report.

User Query:
```
{user_query}
```

Search Results:
```
{search_results}
```"""