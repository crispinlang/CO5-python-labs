from whoosh import index
from II_index import index_dir
from whoosh.qparser import QueryParser, MultifieldParser, OrGroup, FuzzyTermPlugin, PhrasePlugin


ix = index.open_dir(index_dir)

def run_search_experiment(ix, query_str, mode="MULTIFIELD"):
    """
    Executes a search using different logic modes without reopening the index.
    
    Modes:
    - 'AND'       : Single-field (content), Exact match, All terms required.
    - 'OR'        : Multi-field, Exact match, Any term required.
    - 'MULTIFIELD': Multi-field, Exact match, All terms required (Default).
    - 'FUZZY'     : Multi-field, Approximate match (typo tolerant).
    """
    
    # open the searcher
    with ix.searcher() as s:
        if mode == "AND":
            qp = QueryParser("body", schema=ix.schema)
            
        elif mode == "OR":
            qp = QueryParser("body", schema=ix.schema, group=OrGroup)

        elif mode == "MULTIFIELD":
            # The "Standard" Google-like search: Title + Content, AND logic
            qp = MultifieldParser(["title", "content"], schema=ix.schema, group=OrGroup)
            
        elif mode == "FUZZY":
            # Typo Tolerant: Search ALL fields but allow errors
            qp = MultifieldParser(["title", "body"], schema=ix.schema)
            qp.add_plugin(FuzzyTermPlugin())

        # searching
        q = qp.parse(query_str)
        results = s.search(q, limit=5)

        # output
        print(f"\n" + "="*40)
        print(f"MODE: {mode} | QUERY: '{query_str}'")
        print(f"Parsed as: {q}")
        print(f"Total Hits: {len(results)}")
        print("-" * 40)

        for i, hit in enumerate(results):
            print(f"  {i+1}. [{hit.score:.4f}] {hit.get('title', 'content')}")


## testing:
#run_search_experiment(ix, "acid fluid", mode="AND")
#run_search_experiment(ix, "acid fluid", mode="OR")
#run_search_experiment(ix, "acid fluid", mode="MULTIFIELD")
run_search_experiment(ix, "fluiids~1", mode="FUZZY")

















# ##################################################
# # Setting up the default AND logic retrieval
# with ix.searcher() as s:
#     qp = QueryParser("body", schema=ix.schema)
#     q_AND = qp.parse(search_term)

#     results = s.search(q_AND, limit=1)

#     print("\n" + "="*30 + "\n")

#     print(f"Found {len(results)} results for AND query logic: '{q_AND}'")

#     for hit in results:
#         print(f"Matched: {hit['title']}")
#         print(f"Score:   {hit.score}")
#         print("---")
#     print("\n" + "="*30 + "\n")

# ##################################################
# # Setting up the OR logic retrieval
# with ix.searcher() as s:
#     qp = QueryParser("body", schema=ix.schema, group=OrGroup)
#     q_OR = qp.parse(search_term) 

#     results = s.search(q_OR, limit=1)

#     print(f"Found {len(results)} results for OR query logic: '{q_OR}'")

#     for hit in results:
#         print(f"Matched: {hit['title']}")
#         print(f"Score:   {hit.score}")
#         print("---")
#     print("\n" + "="*30 + "\n")

# ##################################################
# # Setting up the multifield logic retrieval
# # mf = multifield

# with ix.searcher() as s:
#     mf_qp = MultifieldParser(["title", "body"], schema=ix.schema)
#     q_MF = mf_qp.parse(search_term)

#     results_mf = s.search(q_MF, limit=1)
#     print(f"Found {len(results_mf)} results for MF query LOGIC: '{q_MF}'")

#     for hit in results_mf:
#         print(f"Matched: {hit['title']}")
#         print(f"Score:   {hit.score}")
#         print("---")

#     print("\n" + "="*30 + "\n")

# ##################################################
# # Setting up the FUZZY retreival add-on
# # mf_f_qp = multifield fuzzy searcher

# with ix.searcher() as s:
#     mf_qp.add_plugin(FuzzyTermPlugin()) 
#     q_MF_fuzzy = mf_qp.parse(misspelled_search_term)

#     results_mf_fuzzy = s.search(q_MF_fuzzy, limit=1)
#     print(f"Found {len(results_mf_fuzzy)} results for query: '{q_MF_fuzzy}'")

#     for hit in results_mf_fuzzy:
#         print(f"Matched: {hit.get('title', 'Untitled')}")
#         print(f"Score:   {hit.score}")
#         print("---")
