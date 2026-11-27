"""Grounded answer generation with Google Gemini."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.vector_store import VectorStore

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful, friendly chatbot for people who want to learn about Veer Surendra Sai University of Technology (VSSUT).

            For questions about VSSUT, answer using only the information in the supplied document excerpts. Read all relevant excerpts together, combine and organize their information, and answer the user's actual question in clear, natural language. Use short paragraphs, bullets, or numbered steps when they make the answer easier to follow.

            Do not add facts from your general knowledge or make assumptions. If the excerpts do not contain enough relevant information to answer, politely say that you could not find the information in the material available to you. If only part of the question can be answered, answer that part and briefly say which part you could not find information about.

            The user may also make casual conversation. Respond naturally and briefly to greetings, thanks, and other conversational messages. Do not invent university facts in casual replies. If a message mixes casual conversation with a VSSUT question, respond to both, grounding the university-related part in the excerpts.

            Do not cite sources or refer to excerpts, retrieval, metadata, similarity scores, or internal processing. Do not pretend to have information that was not provided.
            """.strip(),
        ),
        ("human", "Question:\n{question}\n\nDocument excerpts:\n{context}"),
    ]
)


class RAG:
    def __init__(self, vector_store: VectorStore, api_key: str, model_name: str):
        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
        )

    def ask(self, question: str, top_k: int = 5) -> tuple[str, list[dict]]:
        matches = self.vector_store.search(question, top_k)
        if not matches:
            return "No documents have been indexed. Add files and run ingestion first.", []

         # Keep source metadata out of the model context. It remains attached to
        # each match and is formatted as citations by the CLI/UI layer.
        context = "\n\n---\n\n".join(match["document"] for match in matches)

        response = (PROMPT | self.llm).invoke(
            {"question": question, "context": context}
        )

        content = response.content
        if isinstance(content, str):
            answer = content.strip()
        elif isinstance(content, list):
            # Gemini may return text as content blocks instead of one string.
            text_parts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            ]
            answer = "\n".join(text_parts).strip()
        else:
            answer = str(content).strip()
        return answer, matches