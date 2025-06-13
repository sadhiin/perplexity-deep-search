# Given user query generate list of search queries
generate_search_queries_prompt = """Given the user query: '{user_query}', generate a set of concise, targeted search queries to retrieve highly relevant information.

Guidelines:
- Analyze the query to identify core components and determine if multiple searches are needed to address distinct aspects.
- Create a logical sequence of search queries that progressively refine and broaden the scope of results.
- Limit the total number of search queries to {MAX_QUERY_GENERATIONS}.
- Incorporate variations in phrasing, synonyms, or alternative approaches to optimize coverage and relevance.
- Use today's date, {current_date}, for context if relevant to the query.

Output Format:
- List each search query on a new line.
- Exclude additional text, explanations, headers, or formatting such as triple backticks."""


# Given user query, search queries and search results, generate refined search queries
refine_search_queries_prompt = """Given the user query: '{user_query}', and the following prior search queries and results:

Prior Search Queries:
{search_queries}

Search Results:
{search_results}

Task:
Analyze the search results to identify new keywords, concepts, or insights that could improve search effectiveness. Generate refined search queries to enhance relevance, specificity, or breadth.

Guidelines:
- Extract relevant terms or themes from the search results to inform new queries.
- Create concise, targeted queries that avoid redundancy and build on prior results.
- Limit the total number of refined queries to {MAX_QUERY_GENERATIONS}.
- Use today's date, {current_date}, for context if relevant to the query.

Output Format:
- List each refined search query on a new line.
- Exclude additional text, explanations, headers, or formatting such as triple backticks."""


# Given user query and search results, generate search result wise summary and final markdown report
final_report_prompt = """
Task:
Generate a comprehensive markdown report synthesizing key insights from the user query and search results. Include brief summaries for each search result, followed by an elaborated report with detailed analysis, structured sections, and actionable recommendations tailored to the query.

Input Parameters:
- User Query: {user_query}
- Search Results: {search_results}

Output Structure:
1. Search Result Summaries:
   - Provide a 1-2 line summary for each search result, starting with its title.
   - Enclose each summary in <summary> and </summary> tags.

   Example:
   <summary>
   "Search Result Title"
   Brief summary of the search result content.
   </summary>

2. Final Markdown Report:
   - Create a detailed, logically structured report with sections dynamically tailored to the query's focus.
   - Enclose the report in <final_markdown_report> and </final_markdown_report> tags.

Guidelines:
- Title & Introduction:
  - Use a precise, query-focused title that encapsulates the report's purpose.
  - Provide an introduction that outlines the query's context, the report's objectives, and a brief overview of the findings.
- Detailed Insights & Analysis:
  - Organize insights into clear, thematic sections with descriptive subheadings.
  - Include in-depth analysis of trends, patterns, or comparisons drawn from the search results.
  - Use quantitative data, qualitative observations, or contextual details where applicable to enrich the analysis.
  - Highlight any conflicting information or gaps in the results, addressing their implications.
- Comprehensive Recommendations:
  - Provide actionable, specific recommendations based on the insights, tailored to the query's intent.
  - Suggest practical next steps, potential applications, or areas for further investigation.
  - Include considerations for limitations or challenges identified in the analysis.
- Conclusion:
  - Summarize key findings and their significance in a concise, impactful manner.
  - Reinforce how the insights address the user query and their potential value.
  - Highlight any unresolved questions or future research directions if relevant.
- Formatting:
  - Use markdown with clear headings, subheadings, bullet points, tables, or code blocks for clarity and readability.
  - Cite search result links (hrefs) inline to support claims, without a separate references section.
  - Ensure the report is professional, detailed, and avoids unnecessary repetition.

Output Format:
- Present search result summaries first, followed by the elaborated markdown report.
- Ensure all content is valid markdown, structured for clarity and depth.
"""