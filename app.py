import os

from langchain_core.tools import tool
from pydantic import BaseModel
from typing_extensions import TypedDict,Annotated
import operator
from langchain_core.messages import AnyMessage

import re
import ast
import time
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
from langchain_core.messages import AIMessage,HumanMessage
from langgraph.graph import StateGraph,START,MessagesState,END
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.graph import StateGraph,START,MessagesState,END
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage

from langgraph.prebuilt import create_react_agent

from langchain_google_genai import GoogleGenerativeAIEmbeddings

import streamlit as st

import logging


logger = logging.getLogger("runs_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler("./running_logs.log", mode="a")
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)






tavily_api_key = os.getenv("TAVILY_API_KEY")


gemini_api_key = os.getenv("GOOGLE_API_KEY")



embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=gemini_api_key)


tavily_search = TavilySearch(max_results=1, api_key=tavily_api_key,topic="general",include_raw_content=True)






llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=gemini_api_key
 
)




class State(TypedDict):

    messages: Annotated[list[AnyMessage],operator.add]
    # running_summary:str = field(default=None)
    title: Annotated[list,operator.add]



format = {
    "subtopics": [
        {
            "title": "Subtopic Title",
            "search_queries": ["query1", "query2"]
        }
    ]
}

prompt = f"""
You are a deep research expert. Your job is to break a broad topic into several detailed subtopics.
For each subtopic, provide a maximum of **four** web search queries that can help collect relevant data.

Your output must strictly follow this JSON-like format:
{format}

Example:
If the topic is "climate change", one subtopic might be "effects on agriculture", and search queries could be:
["impact of climate change on agriculture", "climate change and crop yields"]

Goal: These search queries will be used to gather web data for generating a detailed report.

Now generate subtopics and search queries for the topic: "{{topic}}"
"""


query_generator_agent = create_react_agent(llm,tools=[],prompt=prompt)




chromadb.api.client.SharedSystemClient.clear_system_cache()


vector_db = Chroma(collection_name="research_data_2", embedding_function=embeddings)

# logging.basicConfig(filename="newfile.log",
#                         format='%(asctime)s %(message)s',
#                         filemode='w')


# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)  




def add_to_vectorDB(doc):
    if not doc:
        return False
        
    try:
        logger.log(logging.INFO,f"Adding document to vector DB: {doc.metadata.get('title', 'No title')}")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents([doc])
        logger.log(logging.INFO,f"Split into {len(splits)} chunks")
        vector_db.add_documents(splits)
        logger.log(logging.INFO,f"Successfully added document to vector DB")
        return True
    
    except Exception as e:
        logger.log(logging.INFO,f"Error adding document to vector DB: {e}")

        return False
    






def web_search(state:State):
    """
    Uses the latest message to extract subtopics and web searches, then adds raw content to the vector database
    """ 

    last_message = state['messages'][-1]
    pattern = r"```json\s*(.*?)\s*```"

    if isinstance(last_message,AIMessage):
        message_content = last_message.content   

        logger.log(logging.INFO,"starting pattern search ")
        subtopics_dict = re.search(pattern,message_content,re.DOTALL)
        logger.log(logging.INFO,f"found pattern {subtopics_dict}")

        if subtopics_dict:
            result = subtopics_dict.group(1)
                
            result = ast.literal_eval(result)

            for i,content in enumerate(result['subtopics']):

                title = content.get("title")

                if title:
                    metadata = {"title":title}
                    state['title'].append(title)
                else:
                    metadata = {"title":"no title"}

                for query in content.get('search_queries',"no search query"):

                    
                    logger.log(logging.INFO,f"starting search for search {i}, query: {query}")

                    try:
                        search_result = tavily_search.invoke({"query":query})

                        if search_result:
                            logger.log(logging.INFO,f"found search result {i}")

                            raw_content = search_result["results"][0].get("raw_content","No content")

                            if raw_content:
                                raw_content.replace("\n","")

                                docs = Document(page_content=raw_content,metadata=metadata)

                                add_to_vectorDB(docs)

                            else:
                                logger.log(logging.INFO,f"no raw content found for search {i}")

                    except Exception as e:
                        logger.log(logging.ERROR,f"unable to perform search, {e}")
                        return State['messages'].append(AIMessage(content=f"unable to perform search, error:{e}"))



                logger.log(logging.INFO,"sleeping for 6 seconds")
                time.sleep(6)

        else:
            return State['messages'].append(AIMessage(content="unable to extract subtopics"))
        
    else:
        return state['messages'].append(AIMessage(content="no AI message in messages"))

    try:
        db_size = len(vector_db.get()['documents'])
        result_text = f"added {db_size} elements to vector db"       


    except Exception as e:
        result_text = f"error finding size of vector db check if its initilaized {e}"



    return state['messages'].append(AIMessage(content=result_text))



summarizer_instructions = """
You are a specialized research assistant responsible for generating detailed, comprehensive research reports based on retrieved documents. Your reports must demonstrate academic rigor, analytical depth, and thorough coverage of all aspects of each topic.

REPORT STRUCTURE AND CONTENT REQUIREMENTS:
For each subject (e.g., historical figure, event, movement, or development), provide:

1. COMPREHENSIVE OVERVIEW (1-2 paragraphs):
   - Clear definition and significance of the subject
   - Temporal and geographical context
   - Brief introduction to key themes that will be explored

2. DETAILED ANALYSIS BY SUBTOPIC:
   Each subtopic should include:

   ## [Subtopic Title]

   **Historical Context:**
   - Thorough exploration of preceding events and conditions
   - Cultural, political, and social environment
   - Relevant ideological currents or intellectual foundations

   **Core Developments:**
   - Chronological progression of key events
   - Critical turning points and catalyst moments
   - Primary sources or documented evidence where applicable
   - Different perspectives or interpretations by scholars

   **Key Figures and Their Contributions:**
   - Biographical details relevant to their role
   - Specific actions, decisions, or works that proved influential
   - Relationships with other significant actors or institutions

   **Mechanisms of Change:**
   - Analysis of how and why developments occurred
   - Examination of power structures, resources, or tactical approaches
   - Assessment of resistance or support from different sectors

   **Short and Long-term Implications:**
   - Immediate effects on contemporaneous systems or populations
   - Lasting legacy and influence on subsequent developments
   - Changes to institutions, laws, cultural practices, or social norms
   - Global or regional ripple effects

   **Critical Analysis:**
   - Scholarly debates or competing interpretations
   - Methodological considerations in studying this topic
   - Gaps in historical knowledge or contested narratives

   **Connections to Broader Themes:**
   - Links to major historical processes (e.g., industrialization, globalization)
   - Relationship to theoretical frameworks (e.g., colonialism, nationalism)
   - Comparisons with similar developments in other contexts

3. VISUAL AND ORGANIZATIONAL ELEMENTS:
   - Chronological timelines of key events
   - Hierarchical relationships between actors or institutions
   - Geographic distributions or movements
   - Statistical data presented clearly when relevant

4. CONCLUDING SYNTHESIS:
   - Integration of subtopics into a coherent narrative
   - Assessment of overall historical significance
   - Enduring questions or areas for further research

FORMATTING AND STYLE REQUIREMENTS:
- Use **Markdown** formatting for structure and readability
- Employ formal academic language while maintaining clarity
- Include precise dates, locations, and proper names
- Maintain objective, evidence-based analysis
- Avoid presentism or anachronistic judgments
- Use footnotes for clarifications or supplementary information
- Organize content with clear headers, subheaders, and logical paragraph breaks
- Include bullet points for lists of events, factors, or components
- The output capability is limited to text only so dont display images or timelines

QUALITY STANDARDS:
- Prioritize depth over breadth
- Verify factual accuracy and consistency
- Address multiple perspectives or interpretations
- Acknowledge limitations of available evidence
- Maintain appropriate historical context throughout
- Ensure logical transitions between sections
- Avoid oversimplification of complex historical processes

The final report should function as a standalone, comprehensive academic resource that could serve as a foundation for further research, teaching materials, or policy analysis.
"""






def summarize_the_content(state:State):

    titles = state['title']
    
    full_content = ""

    for title in titles:
    

        if title:
            full_content += f"title: {title}\n"
            docs = vector_db.similarity_search(title)

            if docs:
                logger.log(logging.INFO,f"successfully extracted the docs based on title: {title}")
                for doc in docs:
                    if isinstance(doc,Document):
                        full_content += f"\n{doc.page_content.strip()}\n"

                    else:
                        full_content += "\nNo content\n"

            else:
                logger.log(logging.INFO,f"No docs found for {title}")
                
    
    summary = llm.invoke([SystemMessage(content=summarizer_instructions),
                            HumanMessage(content=full_content)])
    

    state['messages'].append(AIMessage(content=summary.content))
        
    return state



workflow = StateGraph(State)
workflow.add_node("query_generator", query_generator_agent)
workflow.add_node("web_search", web_search)
workflow.add_node("summarize",summarize_the_content)

workflow.add_edge(START, "query_generator")
workflow.add_edge("query_generator", "web_search")
workflow.add_edge("web_search", "summarize")

workflow.add_edge("summarize", END)



graph = workflow.compile()


st.title("Deep research")

user_input = st.text_input("Enter your topic to deep research")

if user_input:
    with st.spinner('Researching your topic... This may take a few minutes'):

        events = graph.invoke({"messages": [HumanMessage(content=user_input)]})

    st.success("Research Completed")
    st.markdown(events['messages'][-1].content)



