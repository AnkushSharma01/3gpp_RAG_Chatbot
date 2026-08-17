import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from src.guardrails import enforce_guardrails
from src.acronym_expander import expand_telecom_query 


def create_rag_chain(retrievers):
    dense_retriever, bm25_retriever = retrievers

    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.0,
        api_key=os.getenv("MISTRAL_API_KEY"),
    )

    def rag_chain(inputs: dict):
        query = inputs["question"]
        expanded_query = expand_telecom_query(query) 

        dense_docs = dense_retriever.invoke(expanded_query) 
        bm25_docs = bm25_retriever.invoke(expanded_query) if bm25_retriever else [] 

        seen = set()
        combined_docs = []
        for doc in dense_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined_docs.append(doc)

        context_str = "\n\n".join(
            [
                f"[Spec: {d.metadata.get('spec_number', '3GPP')} | Clause: {d.metadata.get('clause', 'N/A')}] {d.page_content}"
                for d in combined_docs
            ]
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert 3GPP Telecom AI Assistant. Answer the"
                " user's question ONLY using the provided 3GPP document"
                " context. If the answer cannot be derived from the context,"
                " state that information is insufficient. Do not make up"
                " specifications.",
            ),
            (
                "user",
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            ),
        ])

        chain = prompt | llm
        raw_response = chain.invoke({"context": context_str, "question": query})

        validated_ans = enforce_guardrails(
            query=query, response=raw_response.content, context=context_str
        )

        return {"response": validated_ans, "sources": combined_docs}

    return rag_chain