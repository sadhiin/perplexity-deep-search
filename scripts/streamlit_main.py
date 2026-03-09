import streamlit as st

from workflow import build_graph

# Initialize session state
if "final_report_generated" not in st.session_state:
    st.session_state["final_report_generated"] = False
    st.session_state["stream_data"] = []
    st.session_state["final_markdown_report"] = ""
    st.session_state["search_results"] = []
    st.session_state["claim_confidences"] = []
    st.session_state["reasoning_trace"] = []
    st.session_state["analysis_summary"] = ""
    st.session_state["analysis_results"] = {}
    st.session_state["search_session_history"] = []
    st.session_state["is_followup"] = False

if "claim_confidences" not in st.session_state:
    st.session_state["claim_confidences"] = []
if "reasoning_trace" not in st.session_state:
    st.session_state["reasoning_trace"] = []
if "analysis_summary" not in st.session_state:
    st.session_state["analysis_summary"] = ""
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = {}
if "search_session_history" not in st.session_state:
    st.session_state["search_session_history"] = []


# Function to simulate streaming of data
def fetch_results_streaming(query):
    for chunk in build_graph().stream(
        {
            "user_query": query,
        }
    ):
        key = list(chunk.keys())[0]

        # Case where nodes don't return any data but just go to another step
        if chunk[key] is None:
            continue

        st.session_state["stream_data"].append(chunk)

        if key == "query_planner":
            st.session_state["search_results"] += chunk["query_planner"][
                "search_results"
            ]
        elif key == "should_refine_query":
            st.session_state["search_results"] += chunk["should_refine_query"][
                "search_results"
            ]
        elif key == "final_report_generator":
            st.session_state["final_report_generated"] = True
            st.session_state["final_markdown_report"] = chunk["final_report_generator"][
                "report_markdown"
            ]
            st.session_state["claim_confidences"] = chunk["final_report_generator"].get(
                "claim_confidences", []
            )
            st.session_state["reasoning_trace"] = chunk["final_report_generator"].get(
                "reasoning_trace", []
            )
            st.session_state["analysis_summary"] = chunk["final_report_generator"].get(
                "analysis_summary", ""
            )
            st.session_state["analysis_results"] = chunk["final_report_generator"].get(
                "analysis_results", {}
            )
            st.session_state["search_session_history"] = chunk["final_report_generator"].get(
                "search_session_history", []
            )
            st.session_state["is_followup"] = chunk["final_report_generator"].get(
                "is_followup", False
            )
        yield


# UI Setup
st.set_page_config(page_title="Deep Research", layout="wide")
st.title("Deep Research")
query = st.text_input(
    "Enter your research query:",
)

if query:
    tab1, tab2, tab3 = st.tabs(["Steps Taken", "Final Report", "Fetched Sources"])

    with tab1:
        output_placeholder = st.empty()

        if not st.session_state["final_report_generated"]:
            for _ in fetch_results_streaming(query):
                output_placeholder.empty()  # Clear previous content

                with output_placeholder.container():

                    for step in st.session_state["stream_data"]:
                        if "query_planner" in step:
                            with st.expander(
                                "Looking into initial set of search results",
                                expanded=False,
                            ):
                                st.markdown("**Searching**")
                                queries = step["query_planner"]["search_queries"]
                                cols = st.columns(len(queries))
                                for col, query in zip(cols, queries):
                                    with col:
                                        st.markdown(f"🔍 `{query}`")

                                st.markdown("**Results**")
                                results = step["query_planner"]["search_results"]
                                cols = st.columns(len(results))
                                for col, result in zip(cols, results):
                                    with col:
                                        st.markdown(
                                            f"[{result['title']}]({result['href']})"
                                        )

                        elif "should_refine_query" in step:
                            with st.expander(
                                "Refining search queries based on findings till now",
                                expanded=False,
                            ):
                                st.markdown("**Searching**")
                                queries = step["should_refine_query"]["search_queries"]
                                cols = st.columns(len(queries))
                                for col, query in zip(cols, queries):
                                    with col:
                                        st.markdown(f"🔍 `{query}`")

                                st.markdown("**Results**")
                                results = step["should_refine_query"]["search_results"]
                                cols = st.columns(len(results))
                                for col, result in zip(cols, results):
                                    with col:
                                        st.markdown(
                                            f"[{result['title']}]({result['href']})"
                                        )

                        elif "final_report_generator" in step:
                            with st.expander(
                                "Summarizing search results till now and generating final report",
                                expanded=False,
                            ):
                                for summary in step["final_report_generator"][
                                    "individual_page_summaries"
                                ]:
                                    st.markdown(f"- {summary}")

    with tab2:
        if st.session_state["final_report_generated"]:
            claim_confidences = st.session_state.get("claim_confidences", [])
            if claim_confidences:
                st.subheader("Claim Confidence")
                for entry in claim_confidences:
                    score_pct = int(round(entry.get("confidence_score", 0) * 100))
                    level = entry.get("confidence_level", "unknown").title()
                    st.markdown(
                        f"**{level} confidence ({score_pct}%):** {entry.get('claim', '')}"
                    )
                    evidence = entry.get("evidence") or []
                    if evidence:
                        st.caption("Evidence: " + "; ".join(evidence))
                    rationale = entry.get("rationale")
                    if rationale:
                        st.caption(f"Why: {rationale}")
                st.divider()
            reasoning_trace = st.session_state.get("reasoning_trace", [])
            if reasoning_trace:
                with st.expander("Thinking process", expanded=False):
                    for idx, step in enumerate(reasoning_trace, 1):
                        st.markdown(f"**Step {idx}.** {step}")
                st.divider()
            if st.session_state.get("is_followup"):
                st.caption(
                    "This query was detected as a follow-up to an existing research conversation."
                )
            analysis_summary = st.session_state.get("analysis_summary", "")
            analysis_results = st.session_state.get("analysis_results", {})
            if analysis_summary:
                st.subheader("Analyst Overview")
                st.markdown(analysis_summary)
                st.divider()
            key_insights = analysis_results.get("key_insights") or []
            if key_insights:
                st.subheader("Key Insights")
                for insight in key_insights:
                    st.markdown(f"- {insight}")
                st.divider()
            limitations = analysis_results.get("limitations") or []
            if limitations:
                st.subheader("Limitations Noted")
                for limitation in limitations:
                    st.markdown(f"- {limitation}")
                st.divider()
            search_history = st.session_state.get("search_session_history", [])
            if search_history:
                st.subheader("Recent research sessions")
                for entry in search_history:
                    title = entry.get("search_query", "Unnamed query")
                    count = entry.get("result_count", 0)
                    summary = entry.get("summary") or "No summary available."
                    timestamp = entry.get("created_at", "Unknown time")
                    st.markdown(
                        f"- {timestamp}: `{title}` ({count} results) — {summary}"
                    )
                st.divider()
            st.markdown(st.session_state["final_markdown_report"])

    with tab3:
        if st.session_state["final_report_generated"]:
            for result in st.session_state["search_results"]:
                st.write(f"- [{result['title']}]({result['href']})")
