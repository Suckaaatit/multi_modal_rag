import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.core.node_parser import SentenceSplitter
from rank_bm25 import BM25Okapi
import re

class RAGEngine:
    def __init__(self, api_key):
        Settings.llm = MistralAI(model="open-mistral-7b", api_key=api_key, temperature=0.1)
        Settings.embedding = MistralAIEmbedding(model_name="mistral-embed", api_key=api_key)
        Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=25)

        self.db_client = chromadb.PersistentClient(path="./chroma_db")
        self.index = None
        self._bm25 = None
        self._nodes = []

    def _sanitize(self, name): return re.sub(r'[^a-zA-Z0-9]', '_', name)[:60]

    def index_documents(self, nodes, file_name):
        c_name = self._sanitize(file_name)
        try: self.db_client.delete_collection(c_name)
        except: pass
        
        collection = self.db_client.get_or_create_collection(c_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
        
        self.index = VectorStoreIndex(nodes, storage_context=storage_ctx, embed_model=Settings.embedding)
        # Build BM25 over plain text for hybrid retrieval
        self._nodes = list(nodes)
        texts = [n.text if hasattr(n, 'text') else str(n) for n in self._nodes]
        tokenized = [self._tokenize(t) for t in texts]
        if any(len(tok) for tok in tokenized):
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None
        return self.index

    def query(self, query_text):
        if not self.index: return "Upload a file first.", []
        # Vector candidates
        retriever = self.index.as_retriever(similarity_top_k=10)
        vec_nodes = retriever.retrieve(query_text)

        # BM25 candidates
        bm25_nodes = []
        if self._bm25 and self._nodes:
            scores = self._bm25.get_scores(self._tokenize(query_text))
            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:10]
            bm25_nodes = [self._nodes[i] for i, _ in ranked]

        # RRF fuse
        fused = self._rrf_fuse(vec_nodes, bm25_nodes, k=60, top_k=10)
        nodes = fused if fused else vec_nodes

        # Quote-aware evidence boosting: if user provides a long span or quotes, prefer nodes containing it
        exact_nodes = self._find_quote_matches(query_text)
        if exact_nodes:
            # Put exact/fuzzy matches first, followed by remaining fused nodes
            seen = set(id(n) for n in exact_nodes)
            tail = [n for n in nodes if id(n) not in seen]
            nodes = (exact_nodes + tail)[:10]

        # If we have exact nodes, restrict context to those to keep citations precise
        context_basis = exact_nodes if exact_nodes else nodes

        context_str = "\n\n".join([f"[Page {n.metadata.get('page','?')}]: {n.text}" for n in context_basis])

        prompt = f"""
        You are a precise assistant. Answer ONLY using the Context. For every fact, include a page citation like [Page X].
        If the answer is not in Context, respond: "I couldn't find that information in the document."

        Context:
        {context_str}

        Question: {query_text}
        """

        answer = str(Settings.llm.complete(prompt))
        # Strict citation filtering: only cite pages from evidence actually used in Context
        evidence_pages = sorted({str(n.metadata.get('page','?')) for n in context_basis if n is not None})
        if evidence_pages:
            # If model forgot to include citations, append concise Sources with only evidence pages
            if "[Page" not in answer:
                answer = answer + "\n\nSources: " + ", ".join(f"[Page {p}]" for p in evidence_pages if p)
            else:
                # Trim any hallucinated citations by appending authoritative list
                answer = answer + "\n\n(Verified pages: " + ", ".join(f"[Page {p}]" for p in evidence_pages if p) + ")"
        return answer, nodes

    def _tokenize(self, text: str):
        return [w for w in re.split(r"[^A-Za-z0-9+@._-]+", text.lower()) if w]

    def _rrf_fuse(self, vec_nodes, bm25_nodes, k=60, top_k=10):
        # Assign ranks
        ranks = {}
        def add_ranks(nodes_list, start_rank=1):
            for rank, n in enumerate(nodes_list, start_rank):
                key = id(n)
                ranks.setdefault(key, {"node": n, "score": 0.0})
                ranks[key]["score"] += 1.0 / (k + rank)
        add_ranks(vec_nodes, 1)
        add_ranks(bm25_nodes, 1)
        fused = sorted(ranks.values(), key=lambda x: x["score"], reverse=True)
        return [e["node"] for e in fused[:top_k]]

    def _find_quote_matches(self, query_text: str):
        """Return nodes that contain the query phrase (exact or fuzzy token overlap).
        Prioritize exact substring matches; then add high-overlap nodes.
        """
        if not self._nodes:
            return []
        q = query_text.strip()
        if len(q) < 12 and ' ' not in q:
            # Too short to enforce quote logic
            return []

        q_norm = self._normalize(q)
        q_tokens = set(self._tokenize(q))
        exact = []
        fuzzy = []
        for n in self._nodes:
            t = getattr(n, 'text', str(n))
            t_norm = self._normalize(t)
            if q_norm and q_norm in t_norm:
                exact.append(n)
            else:
                # simple token overlap ratio
                t_tokens = set(self._tokenize(t))
                if not q_tokens or not t_tokens:
                    continue
                overlap = len(q_tokens & t_tokens) / float(len(q_tokens))
                if overlap >= 0.6:  # threshold can be tuned
                    fuzzy.append((overlap, n))
        # Sort fuzzy by overlap desc
        fuzzy_sorted = [n for _, n in sorted(fuzzy, key=lambda x: x[0], reverse=True)]
        # Deduplicate preserving order
        seen = set()
        ordered = []
        for n in exact + fuzzy_sorted:
            i = id(n)
            if i not in seen:
                seen.add(i)
                ordered.append(n)
        return ordered[:10]

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()
