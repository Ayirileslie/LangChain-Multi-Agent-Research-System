from src.agents.agents import build_search_agent, build_reader_agent, writer_chain, critic_chain


def truncate_text(text: str, max_chars: int = 5500) -> str:
    """Truncate long text to avoid token limit errors"""
    if not text:
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n... [Content truncated due to length limit] ..."
    return text

def run_research_pipeline(topic : str) -> dict:

    state = {}

    #search agent working 
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages" : [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = "\n\n".join(
        m.content for m in search_result["messages"]
        if m.__class__.__name__ in ("ToolMessage", "AIMessage") and m.content
    )
    print("\n search result ",state['search_results'])


    
    #step 2 - reader agent 
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent = build_reader_agent()
    search_results_str = str(state.get('search_results', ''))

    reader_result = reader_agent.invoke({
        "messages": [
            ("user", 
            f"Based on the following search results about '{'The impact of AI on the job market in 2026'}', "
            f"pick the most relevant URL(s) and scrape them for deeper content.\n\n"
            f"Search Results:\n{search_results_str}"
            )
        ]
    })

    state['scraped_content'] = reader_result['messages']

    print("\nscraped content: \n", state['scraped_content'])
 
    #step 3 - writer chain 
    # Before calling Reader Agent
    state['search_results'] = truncate_text(str(state.get('search_results', '')), 5500)

    # Before calling Writer / Critic
    state['scraped_content'] = truncate_text(str(state.get('scraped_content', '')), 4000)

    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n Final Report\n",state['report'])


    #critic report 

    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report":state['report']
    })

    print("\n critic report \n", state['feedback'])

    return state