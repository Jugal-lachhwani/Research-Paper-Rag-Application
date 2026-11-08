import json
import os

base_dir = "/home/jugal/Documents/DataScience/Research-Paper-Rag-Application"
pdf_urls_path = os.path.join(base_dir, "Experiment_pipeline", "pdf_urls.json")
# Let's check if pdf_urls.json is in root or Experiment_pipeline
if not os.path.exists(pdf_urls_path):
    pdf_urls_path = os.path.join(base_dir, "pdf_urls.json")

qrels_path = os.path.join(base_dir, "open_ragbench/pdf/arxiv/qrels.json")
queries_path = os.path.join(base_dir, "open_ragbench/pdf/arxiv/queries.json")

try:
    with open(pdf_urls_path, 'r') as f:
        pdf_urls = json.load(f)
    
    first_100_doc_ids = list(pdf_urls.keys())[:100]
    first_100_doc_ids_set = set(first_100_doc_ids)
    
    with open(qrels_path, 'r') as f:
        qrels = json.load(f)
        
    qrels_100 = {}
    for q_id, q_data in qrels.items():
        if q_data['doc_id'] in first_100_doc_ids_set:
            qrels_100[q_id] = q_data
            
    with open(queries_path, 'r') as f:
        queries = json.load(f)
        
    queries_100 = {}
    for q_id in qrels_100.keys():
        queries_100[q_id] = queries[q_id]
        
    out_qrels = os.path.join(base_dir, "qrels_100.json")
    out_queries = os.path.join(base_dir, "queries_100.json")
    out_urls = os.path.join(base_dir, "pdf_urls_100.json")
    
    with open(out_qrels, 'w') as f:
        json.dump(qrels_100, f, indent=4)
        
    with open(out_queries, 'w') as f:
        json.dump(queries_100, f, indent=4)
        
    with open(out_urls, 'w') as f:
        first_100_urls = {k: pdf_urls[k] for k in first_100_doc_ids}
        json.dump(first_100_urls, f, indent=4)
        
    print(f"Successfully processed!")
    print(f"Found {len(qrels_100)} relevant qrels.")
    print(f"Found {len(queries_100)} relevant queries.")
    print(f"Saved files to:")
    print(f"- {out_qrels}")
    print(f"- {out_queries}")
    print(f"- {out_urls}")
    
except Exception as e:
    print(f"Error: {e}")
